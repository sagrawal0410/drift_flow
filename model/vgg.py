import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange
import torch.nn.functional as F
from itertools import islice

class VGGPerceptualFeatures(nn.Module):
    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')):
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        self.layer_name_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
        }
        self.layer_names = layers
        self.selected_layers = [self.layer_name_map[l] for l in self.layer_names]
        self.inv_layer_name_map = {v: k for k, v in self.layer_name_map.items()}

        for param in self.vgg.parameters():
            param.requires_grad = False

    def activations(self, x, num_acti=4, min_res=32, use_mean=False, use_std=False):
        if x.shape[-1] < min_res:
            x = F.interpolate(x, size=(min_res, min_res), mode='bicubic')
        
        res = {}
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.selected_layers:
                name = self.inv_layer_name_map[i]
                reshaped = rearrange(x, 'b c h w -> b (h w) c')
                res[name] = reshaped

                if use_mean:
                    res[f'{name}_mean'] = reshaped.mean(dim=1, keepdim=True)

                if use_std:
                    if reshaped.shape[1] == 1:
                        res[f'{name}_std'] = reshaped
                    else:
                        res[f'{name}_std'] = reshaped.std(dim=1, keepdim=True)

        return dict(islice(res.items(), num_acti))
    
    def acti_names(self, use_mean=False, use_std=False):
        final_names = []
        for name in self.layer_names:
            final_names.append(name)
            if use_mean:
                final_names.append(f'{name}_mean')
            if use_std:
                final_names.append(f'{name}_std')
        return final_names

def vgg16_perceptual(random_init=False, **kwargs):
    model = VGGPerceptualFeatures(**kwargs)
    if random_init:
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(weights_init)

    return model

