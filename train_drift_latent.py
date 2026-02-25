import os
import argparse
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL

from config import load_config
from dataset.cache_dataset import CachedFolder
from model.LightningDiT.lightningdit import DitGen
from features import Flatten, Coordinate, UnfoldFeatures
from utils.ema import EMA
from utils.ckpt_utils import save_ckpt, load_last_ckpt, get_run_name, ckpt_epoch_numbers
from utils.fid import eval_fid, visualize_imagenet_samples
from utils.logging_utils import WandbLogger
from utils.distributed_utils import is_main_process, get_rank, get_world_size


def get_lr(step, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


class ClassLabelDataset(Dataset):

    def __init__(self, num_classes=1000, samples_per_class=50):
        self.labels = []
        for c in range(num_classes):
            self.labels.extend([c] * samples_per_class)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.long)


def build_vae_decoder(device):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def make_eval_generator(model, vae, cfg_scale, device):
    cache_scaling = 0.18125

    @torch.inference_mode()
    def generator_fn(batch):
        if isinstance(batch, (list, tuple)):
            labels = batch[0].to(device)
        else:
            labels = batch.to(device)
        output = model(labels, cfg_scale=cfg_scale)
        latents = output["samples"]
        images = vae.decode(latents / cache_scaling).sample
        images = ((images + 1) / 2).clamp(0, 1)
        return images

    return generator_fn


def main(args):
    cfg = load_config(args.config)

    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = (rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1
        is_main = True

    dataset_name = cfg.dataset.get("name", "imagenet256_cache")
    run_name = args.run_name or cfg.wandb.get("name", None) or "dit_B2_flatten"
    run_id = get_run_name(run_name)

    logger = WandbLogger()
    logger.setup_wandb(
        project=cfg.wandb.get("project", "drift-flow"),
        entity=cfg.wandb.get("entity", None) or None,
        name=run_id,
        config=cfg,
    )
    if is_main:
        print(f"Run ID: {run_id}")

    cached_path = args.cached_path
    dataset = CachedFolder(root=cached_path)
    if is_main:
        print(f"Loaded cached latent dataset: {len(dataset)} samples")

    batch_size = args.batch_size or cfg.train.get("total_batch_size", 64)

    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    dec_cfg = cfg.model.decoder_config
    generator = DitGen(**dec_cfg).to(device)

    if world_size > 1:
        generator = DDP(generator, device_ids=[local_rank])

    if is_main:
        n_params = sum(p.numel() for p in generator.parameters()) / 1e6
        print(f"Generator parameters: {n_params:.1f}M")

    ema_decay = cfg.train.get("ema_decay", 0.999)
    gen_model_raw = generator.module if world_size > 1 else generator
    ema = EMA(gen_model_raw, decay=ema_decay)
    if is_main:
        print(f"EMA (eval only): decay={ema_decay}")

    input_shape = tuple(cfg.model.get("input_shape", [4, 32, 32]))
    feat_params = cfg.model.get("feature_params", {})

    feature_extractors = []
    if feat_params.get("has_global", False):
        feature_extractors.append(Flatten(input_shape))
    if feat_params.get("has_local", False):
        feature_extractors.append(Coordinate(input_shape))
    for uf_dict in feat_params.get("unfold_feature_dicts", []):
        feature_extractors.append(UnfoldFeatures(
            input_shape,
            patch_size=uf_dict.get("patch_size", 2),
            ds_steps=uf_dict.get("ds_steps", 1),
            use_mean_std=uf_dict.get("use_mean_std", True),
        ))
    if is_main:
        print(f"Feature extractors: {[f.name() for f in feature_extractors]}")

    opt_cfg = cfg.get("optimizer", {})
    base_lr = opt_cfg.get("lr", 2e-4)
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=base_lr,
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    start_step = 0
    load_dict = cfg.train.get("load_dict", {})
    resume_run_id = args.resume_run_id or load_dict.get("run_id", "")
    if resume_run_id:
        if is_main:
            print(f"Resuming from run: {resume_run_id}")
        ckpt = load_last_ckpt(resume_run_id)
        gen_model_raw.load_state_dict(ckpt["generator"])
        ema.model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("global_step", 0)
        if is_main:
            print(f"Resumed from step {start_step}")

    vae = None

    num_classes = cfg.train.get("n_classes", 1000)
    eval_dataset = ClassLabelDataset(num_classes=num_classes, samples_per_class=50)

    train_cfg = cfg.train
    fwd = train_cfg.get("forward_dict", {})
    attn_cfg = fwd.get("attn_dict", {})

    Nc = train_cfg.get("n_class_labels", 8)
    N_neg = fwd.get("recon", 8)
    cfg_scale = train_cfg.get("min_cfg_scale", 1.0)
    warmup_steps = train_cfg.lr_schedule.get("warmup_steps", 5000)
    total_steps = train_cfg.get("n_steps", 30000)
    clip_grad = train_cfg.lr_schedule.get("clip_grad", 2.0)
    save_every = train_cfg.get("save_per_step", 5000)
    eval_every = train_cfg.get("eval_gen_per_step", 5000)
    eval_fid_samples = train_cfg.get("eval_fid_samples", 50000)
    eval_batch_size = train_cfg.get("eval_bsz_per_gpu", 64)

    contra_dict = dict(
        kernel_type=attn_cfg.get("kernel_type", "attn_new"),
        sample_norm=attn_cfg.get("sample_norm", True),
        scale_dist_normed=attn_cfg.get("scale_dist_normed", True),
        R_list=attn_cfg.get("R_list", [0.02, 0.05, 0.2]),
    )

    global_step = start_step
    n_dataset = len(dataset)
    steps_per_epoch = n_dataset // (batch_size * world_size)

    if is_main:
        print(f"\n{'='*60}")
        print(f"Pipeline summary:")
        print(f"  Config file:          {args.config}")
        print(f"  Run ID:               {run_id}")
        print(f"  Cached latent path:   {cached_path}")
        print(f"  Latent shape:         {list(input_shape)}")
        print(f"  Dataset size:         {n_dataset}")
        print(f"  Batch size (per GPU): {batch_size}")
        print(f"  Steps per epoch:      {steps_per_epoch}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Start step:           {start_step}")
        print(f"  Warmup steps:         {warmup_steps}")
        print(f"  Save every:           {save_every} steps")
        print(f"  Eval FID every:       {eval_every} steps ({eval_fid_samples} samples)")
        print(f"  Nc={Nc}, N_neg={N_neg} (positives from batch)")
        print(f"  CFG alpha:            {cfg_scale} (1.0 = no CFG)")
        print(f"  EMA decay (eval):     {ema_decay}")
        print(f"  Grad clip:            {clip_grad}")
        print(f"  Loss config:          {contra_dict}")
        print(f"{'='*60}\n")

    gen_model = generator.module if world_size > 1 else generator
    epoch = 0

    while global_step < total_steps:
        if world_size > 1:
            sampler.set_epoch(epoch)

        for batch_idx, (latents, labels) in enumerate(dataloader):
            if global_step >= total_steps:
                break

            lr = get_lr(global_step, base_lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            latents = latents.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            unique_labels = labels.unique()
            if len(unique_labels) < Nc:
                selected = unique_labels
            else:
                perm = torch.randperm(len(unique_labels), device=device)[:Nc]
                selected = unique_labels[perm]

            pos_list = []
            gen_list = []
            valid_classes = []

            for c in selected:
                c_val = c.item()
                mask = (labels == c)
                target_latents = latents[mask]
                if target_latents.shape[0] == 0:
                    continue

                rand_idx = torch.randint(0, target_latents.shape[0], (1,), device=device)
                pos_list.append(target_latents[rand_idx])

                class_cond = torch.full((N_neg,), c_val, dtype=torch.long, device=device)
                gen_output = gen_model(class_cond, cfg_scale=cfg_scale)
                generated_latents = gen_output["samples"]
                gen_list.append(generated_latents)

                valid_classes.append(c_val)

            if len(valid_classes) == 0:
                continue

            target_batch = torch.stack(pos_list, dim=0)       # [B_valid, 1, 4, 32, 32]
            gen_batch = torch.stack(gen_list, dim=0)           # [B_valid, N_neg, 4, 32, 32]

            total_loss = torch.zeros(len(valid_classes), device=device)
            all_info = {}

            for feat in feature_extractors:
                loss, info = feat(
                    target=target_batch,
                    recon=gen_batch,
                    contra_dict=contra_dict,
                )
                total_loss = total_loss + loss
                for k, v in info.items():
                    all_info[f"{feat.name()}/{k}"] = v

            avg_loss = total_loss.mean()

            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_grad)
            optimizer.step()

            ema.update(gen_model)

            global_step += 1

            # ── Logging ──
            log_payload = {
                "train/loss": avg_loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/step": global_step,
            }
            for k, v in all_info.items():
                if isinstance(v, torch.Tensor):
                    log_payload[f"train/{k}"] = v.item()
                else:
                    log_payload[f"train/{k}"] = v
            logger.log_dict(log_payload, step=global_step)
            logger.set_step(global_step)

            if is_main and global_step % 100 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                log_str = (f"[epoch {epoch}][step {global_step}/{total_steps}] "
                           f"loss={avg_loss.item():.6f} lr={cur_lr:.2e}")
                print(log_str)

            if global_step % save_every == 0:
                ckpt_dict = {
                    "generator": gen_model.state_dict(),
                    "ema": ema.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                save_ckpt(run_id, global_step, ckpt_dict, max_ckpts=5)
                if is_main:
                    print(f"Saved checkpoint at step {global_step}")

            if global_step % eval_every == 0:
                if is_main:
                    print(f"\n{'─'*40}")
                    print(f"FID evaluation at step {global_step}...")

                if vae is None:
                    vae = build_vae_decoder(device)
                    if is_main:
                        print("Loaded SD-VAE decoder for FID evaluation")

                ema_gen_fn = make_eval_generator(ema.model, vae, cfg_scale, device)
                visualize_imagenet_samples(
                    generator=ema_gen_fn,
                    logger=logger,
                    log_prefix=f"step_{global_step}",
                )

                ema_fid_result = eval_fid(
                    generator=ema_gen_fn,
                    cond_dataset=eval_dataset,
                    logger=logger,
                    total_samples=eval_fid_samples,
                    gpu_batch_size=eval_batch_size,
                    log_prefix=f"EMA_{global_step}",
                    dataset=dataset_name,
                    log_folder_name="eval",
                )
                if is_main:
                    fid_val = ema_fid_result.get("fid", float("nan"))
                    isc_val = ema_fid_result.get("isc_mean", float("nan"))
                    prec_val = ema_fid_result.get("precision", 0)
                    recall_val = ema_fid_result.get("recall", 0)
                    logger.log_dict({
                        "eval/ema_fid": fid_val,
                        "eval/ema_isc": isc_val,
                        "eval/ema_precision": prec_val,
                        "eval/ema_recall": recall_val,
                    }, step=global_step)
                    print(f"EMA FID @ step {global_step}: {fid_val:.2f}  "
                          f"IS: {isc_val:.2f}  P: {prec_val:.3f}  R: {recall_val:.3f}")
                    print(f"{'─'*40}\n")

                if world_size > 1:
                    dist.barrier()

        if is_main:
            print(f"Epoch {epoch} complete, global_step={global_step}/{total_steps}")
        epoch += 1

    if global_step > 0:
        ckpt_dict = {
            "generator": gen_model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        }
        save_ckpt(run_id, global_step, ckpt_dict, max_ckpts=5)
        if is_main:
            print(f"Saved final checkpoint at step {global_step}")

    if is_main:
        print(f"\nFinal FID evaluation at step {global_step}...")
    if vae is None:
        vae = build_vae_decoder(device)
    ema_gen_fn = make_eval_generator(ema.model, vae, cfg_scale, device)
    visualize_imagenet_samples(
        generator=ema_gen_fn,
        logger=logger,
        log_prefix=f"final_{global_step}",
    )
    final_fid = eval_fid(
        generator=ema_gen_fn,
        cond_dataset=eval_dataset,
        logger=logger,
        total_samples=eval_fid_samples,
        gpu_batch_size=eval_batch_size,
        log_prefix=f"EMA_final_{global_step}",
        dataset=dataset_name,
        log_folder_name="eval",
    )
    if is_main:
        fid_val = final_fid.get("fid", float("nan"))
        isc_val = final_fid.get("isc_mean", float("nan"))
        logger.log_dict({
            "eval/final_ema_fid": fid_val,
            "eval/final_ema_isc": isc_val,
        }, step=global_step)
        print(f"Final EMA FID: {fid_val:.2f}  IS: {isc_val:.2f}")

    logger.finish()


def get_args():
    parser = argparse.ArgumentParser("Drifting Model Training")

    parser.add_argument('--config', type=str, default='configs/dit_B2_flatten.yaml',
                        help='Path to YAML config file')

    parser.add_argument('--cached_path', type=str, required=True,
                        help='Path to cached SD-VAE latents (output of cache_latent.py)')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size per GPU (0 = use config value)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume_run_id', type=str, default='',
                        help='Run ID to resume from (overrides config load_dict.run_id)')
    parser.add_argument('--run_name', type=str, default='',
                        help='Run name (overrides config wandb.name)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
