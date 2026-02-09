# %%
import torch 
import torch.nn as nn 
from utils.persistence import persistent_class
import torch.nn.functional as F
from utils.feats import unfold_feats, group_features
from utils.misc import custom_compile
from einops import rearrange

@persistent_class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

@persistent_class
class WrappedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, stride=1, n_blocks=2):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )
        layer_list = [ResidualBlock(hidden_channels, hidden_channels, stride)] + [ResidualBlock(hidden_channels, hidden_channels, 1)] * (n_blocks - 1)
        self.middle = nn.Sequential(*layer_list)
        self.out_block = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.in_block(x)
        x = self.middle(x)
        x = self.out_block(x)
        return x

from math import prod
@persistent_class
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, wrapped=False, channels=[64, 128, 256, 512, 512], strides=[1,2,2,2,2], n_blocks=4, resolution=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.channels = channels
        self.resolution = resolution
        if wrapped:
            self.layers = nn.ModuleList([WrappedResBlock(in_channels, in_channels, channel, stride, n_blocks) for (channel, stride) in zip(channels, strides)])
            last_dim = (resolution // prod(strides)) ** 2 * in_channels
        else:
            self.layers = nn.ModuleList([ResidualBlock(old_channel, new_channel, stride) for (old_channel, new_channel, stride) in zip([in_channels] + channels[:-1], channels, strides)])
            last_dim = (resolution // prod(strides)) ** 2 * channels[-1]
        
        self.fc = nn.Linear(last_dim, out_channels)
    
    def get_encoder(self, id):

        '''
        Return the encoder for the given id, which maps input to [res // (2 ** (id + 1))] x [res // (2 ** (id + 1))]
        Args:
            id: the id of the encoder
        '''

        def stride_until(goal):
            '''
            Return first i such that prod(strides[:i]) >= goal
            '''
            if goal == 1:
                return -1
            prod = 1
            for (i, x) in enumerate(self.strides):
                prod *= x
                if prod >= goal:
                    return i
            return len(self.strides) - 1
        return nn.Sequential(*self.layers[stride_until(2 ** id) + 1 : stride_until(2 ** (id + 1)) + 1])
    
    def get_activations(self, x):

        '''
        Returns a list of activations for each layer
        Activations are stored as a list of tuples (name, activation). 
        Args:
            x: input tensor
        Returns:
            activations: dict of {name: activation}
        '''
        # print("!!!", x.shape)
        activations = {}
        for (i, layer) in enumerate(self.layers):
            x = layer(x)
            activations[f"layer{i}"] = x
        activations["fc"] = self.fc(torch.flatten(x, 1))
        return activations

    def forward(self, x, debug=False):
        for (i, layer) in enumerate(self.layers):
            x = layer(x)
            if debug:
                print(f"layer{i}", x.shape, x.mean(), x.std())

        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

# --- Model Definition ---
class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18/34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout after first activation
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class LatentResNet(nn.Module):
    """Custom ResNet optimized for 4-channel latent inputs (4x32x32)"""
    def __init__(self, layers, num_classes, in_channels=4, base_channels=64, dropout_prob=0.0, block=BasicBlock):
        super(LatentResNet, self).__init__()
        self.in_channels = base_channels
        
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2, dropout_prob=dropout_prob)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_prob=0.0):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_prob))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels):
        '''
        Args:
            x: (B, C, H, W)
            labels: (B, )
        Returns:
            loss: (1,)
            info: dict
        '''

        info = dict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        info['scale_1'] = x.std().mean()
        x = self.layer2(x)
        info['scale_2'] = x.std().mean()
        x = self.layer3(x)
        info['scale_3'] = x.std().mean()
        x = self.layer4(x)
        info['scale_4'] = x.std().mean()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        loss = F.cross_entropy(x, labels)
        info['acc'] = (x.argmax(dim=1) == labels).float().mean()
        info['loss'] = loss.mean()
        return loss, info
        
    def activations(self, x, num_acti=3, min_res=16, use_std=False, use_mean=False, unfold_kernel_list=[], patch_group_size=1):
        '''
        x: [B, C, H, W]
        num_acti: int
        Returns: 
            list of activations, each of shape [B, F, D] (where F = HW, D=C);
        '''
        assert len(x.shape) == 4
        assert x.shape[-1] == x.shape[-2]
        if x.shape[-1] < min_res:
            x = F.interpolate(x, size=min_res, mode='bicubic')

        res = {}
        
        def push(x, name):
            nonlocal res
            reshaped = rearrange(x, 'b c h w -> b (h w) c')
            res[name] = group_features(x, patch_group_size)
            if use_mean:
                res[f"{name}_mean"] = reshaped.mean(dim=1, keepdim=True)
            if use_std:
                if reshaped.shape[1] == 1:
                    res[f"{name}_std"] = reshaped
                else:
                    res[f"{name}_std"] = reshaped.std(dim=1, keepdim=True)
            for kernel in unfold_kernel_list:
                if x.shape[-1] > 1 and x.shape[-2] > 1:
                    unfolded, corr = unfold_feats(x, kernel, return_corr=True) # [B, kernel, kernel, H, W, C]
                    res[f"{name}_unfold_{kernel}"] = rearrange(unfolded, 'b k l h w c -> b (h w) (k l c)')
                    res[f"{name}_corr_{kernel}"] = corr
            if len(res) >= num_acti:
                # slice the dict to return only num_acti keys
                res = {k: res[k] for k in list(res.keys())[:num_acti]}
            return len(res) >= num_acti
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if push(x, 'conv1'):
            return res
        
        x = self.layer1(x)
        if push(x, 'layer1'):
            return res
        
        x = self.layer2(x)
        if push(x, 'layer2'):
            return res
        
        x = self.layer3(x)
        if push(x, 'layer3'):
            return res
        
        x = self.layer4(x)
        push(x, 'layer4')
        return res

def build_latent_resnet(**kwargs):
    """
    Builder function for ResNet models that supports both standard kwargs and checkpoint loading.
    
    Args:
        load_dict (dict, optional): Dictionary containing checkpoint loading information with keys:
            - run_id (str): The run ID for checkpoint loading
            - epoch (str): The epoch to load ("latest" or specific epoch number)
            - load_entry (str, optional): Entry in checkpoint to load (default: "model")
        **kwargs: Standard ResNet constructor arguments (in_channels, out_channels, etc.)
    
    Returns:
        ResNet model instance
    """
    load_dict = kwargs.pop('load_dict', None)
    
    if load_dict is not None and load_dict.get('run_id', ''):
        # Load from checkpoint
        from utils.ckpt_utils import load_ckpt_epoch
        from utils.misc import EasyDict
        
        # Set default values
        load_dict = EasyDict(load_dict)
        if not hasattr(load_dict, 'load_entry'):
            load_dict.load_entry = 'model'

        loaded = load_ckpt_epoch(
            run_id=load_dict.run_id, 
            epoch=load_dict.epoch,
        )[load_dict.load_entry]
        model = LatentResNet(**kwargs)
        model.load_state_dict(loaded if isinstance(loaded, dict) else loaded.state_dict())
        return model
    else:
        # Create new model with provided kwargs
        return LatentResNet(**kwargs)

def get_resnet_spec(arch_name):
    if arch_name == 'resnet18':
        return BasicBlock, [2, 2, 2, 2]
    elif arch_name == 'resnet34':
        return BasicBlock, [3, 4, 6, 3]
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

def resnet18_cifar_wrapped(num_classes=10):
    """ResNet-18 model for CIFAR dataset"""
    return ResNet(in_channels=3, out_channels=num_classes, wrapped=True, strides=[1,2,2,2,2], n_blocks=4, resolution=32, channels=[64, 128, 256, 512, 512])

def resnet18_cifar(num_classes=10):
    """ResNet-18 model for CIFAR dataset"""
    return ResNet(in_channels=3, out_channels=num_classes, wrapped=False, strides=[1,2,2,2], n_blocks=3, resolution=32, channels=[64, 128, 256, 512])

if __name__ == "__main__":
    # model = resnet18_cifar(10)
    # model(torch.randn(1, 3, 32, 32), debug=True)
    # s = model.get_activations(torch.randn(1, 3, 32, 32))
    # for (name, act) in s.items():
    #     print(name, act.shape)
    model = build_latent_resnet(layers=[2, 2, 2, 2], num_classes=10, in_channels=4, base_channels=128, dropout_prob=0.0)
    acti = model.activations(torch.randn(1, 4, 32, 32), num_acti=20, min_res=16, use_std=True, use_mean=True, unfold_kernel_list=[], patch_group_size=1)
    for (name, act) in acti.items():
        print(name, act.shape)