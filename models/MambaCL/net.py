from torch import nn
from .cl_module import CLNet
import torch.nn.functional as F
import torch
from models.MambaCL.Mamba_backbone import Backbone_VSSM



class CLCDNet(nn.Module):
    def __init__(self, feature_size=64, n_class=2, img_chan=3, in_chan=32, projection_dim=128, temperature=0.5, image_size=512, mean=(0.5,), std=(0.229, 0.224, 0.225), use_denseCL=False):
        super(CLCDNet, self).__init__()


        self.encoder = Backbone_VSSM(pretrained='checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth')


        self.feature_size = feature_size
        self.n_class = n_class
        self.img_chan = img_chan
        self.in_chan = in_chan
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_denseDL = use_denseCL
        self.clnet = CLNet(backbone=self.encoder, in_chan=self.in_chan, n_class=self.n_class,
                           feature_size=self.feature_size, projection_dim=self.projection_dim,
                           temperature=self.temperature, image_size=self.image_size, mean=self.mean, std=self.std,
                           use_denseCL=self.use_denseDL) #swin


    def forward(self, x0, x1, x2, label):
        if self.use_denseDL:
            cl_logits, targets, dense_logits, dense_label, n, cd_map1, cd_map2 = self.clnet(x0, x1, x2, label)
            return cl_logits, targets, dense_logits, dense_label, n, cd_map1, cd_map2
        else:
            cl_logits, targets, cd_map1, cd_map2 = self.clnet(x0, x1, x2, label)
            return cl_logits, targets, cd_map1, cd_map2


class Predict(nn.Module):
    def __init__(self, encoder, decoder, feature_size=1000, n_class=2, img_chan=3, in_chan=32, projection_dim=128, temperature=0.5, image_size=512, mean=(0.5,), std=(0.229, 0.224, 0.225)):
        super(Predict, self).__init__()

        self.encoder = encoder
        self.cls = decoder
        self.in_chan = in_chan
        self.n_class = n_class

    def forward(self,x1,x2):
        size = x1.shape[2:]
        f1, f2 = self.encoder(x1), self.encoder(x2)
        f1 = F.interpolate(f1, size=size, mode='bicubic', align_corners=True)
        f2 = F.interpolate(f2, size=size, mode='bicubic', align_corners=True)
        cd_map = self.cls(f1, f2)
        return cd_map


