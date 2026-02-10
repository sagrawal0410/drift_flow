import yaml
from utils.misc import EasyDict
import torch.optim as optim
from utils.logging_utils import WandbLogger
from utils.misc import set_seed
from utils.misc import add_weight_decay
from model.resnet import get_resnet_spec
from diffusers import AutoencoderKL

def _dict_to_easydict(d):
    """
    Recursively convert a dictionary to an EasyDict.
    """
    if not isinstance(d, dict):
        return d
    
    for k, v in d.items():
        d[k] = _dict_to_easydict(v)
    
    return EasyDict(d)

def load_config(config_path):
    """
    Load a YAML configuration file and convert it to an EasyDict.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return _dict_to_easydict(config_dict) 


def null_prepost():
    def pre_fn(x, label_list=None):
        return x

    def post_fn(x, label_list=None):
        return ((x + 1) / 2).clamp(0, 1)

    return pre_fn, post_fn

def vae_prepost():
    def pre_fn(x, label_list=None):
        return x
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").cuda()
    def post_fn(x, label_list=None):
        gen = vae.decode(x / 0.18215).sample
        gen = (gen + 1) / 2
        gen = gen.clamp(0, 1)
        return gen
    return pre_fn, post_fn

def get_prepost_for_dataset(dataset_name):
    """
    Select preprocess/postprocess functions based on dataset.
    - Default: identity pre, simple post scaling to [0,1]
    - imagenet256_cache: inputs are VAE latents (scaled). Pre is identity; Post decodes latents to images.
      Uses the same scaling factor as the cache reader.
    - imagenet256_latent: inputs are RGB images. Pre encodes to VAE latents with stop-grad (online cache);
      Post decodes latents to images.
    """
    # Default pre/post
    if dataset_name not in ["imagenet256_cache", "imagenet256_latent"]:
        return null_prepost()

    # Construct VAE once per process and freeze
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").cuda()
    vae.eval()
    vae.requires_grad_(False)

    # Compile the vae decode function if running PyTorch >= 2.0 for faster decoding
    import torch
    vae.decode = torch.compile(vae.decode)  # or use 'default'
    vae.encode = torch.compile(vae.encode)

    # Note: cache_dataset scales by 0.18125, keep consistent for decoding cached latents
    cache_scaling = 0.18125
    online_scaling = 0.18215

    if dataset_name == "imagenet256_cache":
        def pre_fn(x, label_list=None):
            return x.cuda()
        def post_fn(x, label_list=None):
            gen = vae.decode(x.cuda() / cache_scaling).sample
            gen = (gen + 1) / 2
            return gen.clamp(0, 1)
        return pre_fn, post_fn

    # dataset_name == "imagenet256_latent"
    def pre_fn(x, label_list=None):
        # x: images in [-1,1], B,C,H,W on CUDA; encode to latents with stop-grad
        import torch
        with torch.no_grad():
            latents = vae.encode(x.cuda()).latent_dist.sample()
            latents = latents * online_scaling
        return latents

    def post_fn(x, label_list=None):
        with torch.no_grad():
            gen = vae.decode(x.cuda() / online_scaling).sample
            gen = (gen + 1) / 2
            return gen.clamp(0, 1)

    return pre_fn, post_fn

def build_model_dict(config, model_class):
    """
    Build models and datasets.
    """
    print("Building model...")

    model = model_class(**config.model)

    print("Building dataset...")

    from dataset import get_dataset

    eval_dataset = get_dataset(config.dataset.name, split="val", tar_local=config.dataset.get("tar_local", False))
    train_dataset = get_dataset(config.dataset.name, split="train", tar_local=config.dataset.get("tar_local", False))

    # Select preprocess/postprocess based on dataset
    pre_fn, post_fn = get_prepost_for_dataset(config.dataset.name)
    config.train.preprocess_fn = pre_fn
    config.train.postprocess_fn = post_fn

    beta1 = config.optimizer.get("beta1", 0.9)
    beta2 = config.optimizer.get("beta2", 0.999)
    optimizer = optim.AdamW(
        add_weight_decay(
            model, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, 
        ),
        betas=(beta1, beta2)
    )

    # Setup logger
    logger = WandbLogger()
    w_cfg = config.wandb
    logger.setup_wandb(
        config=config,
        **w_cfg,
    )
    return EasyDict(
        model=model,
        optimizer=optimizer,
        logger=logger,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        dataset_name=config.dataset.name,
        train=config.train,
    )


def compose_pre_post(ds_list):
    '''
    Args:
        ds_list: a list of tuples (pre_fn, post_fn); each will be applied in order
    Returns:
        (pre, post): a tuple of functions that preprocesses and postprocesses the data
        pre: a function that preprocesses the data
        post: a function that postprocesses the data
    '''
    def pre(x, label_list=None):
        for pre_fn in ds_list:
            x = pre_fn[0](x, label_list)
        return x
    
    def post(x, label_list=None):
        for post_fn in ds_list[::-1]:
            x = post_fn[1](x, label_list)
        return x
    return pre, post