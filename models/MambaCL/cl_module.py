# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
from einops import rearrange


class CLNet(nn.Module):
    """ 
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    Link: https://arxiv.org/abs/2002.05709
    Implementation: https://github.com/google-research/simclr
    """
    def __init__(self, backbone, in_chan=2048, n_class=2, feature_size=1000, projection_dim=128, temperature=0.5,
                 image_size=224, mean=(0.5,), std=(0.229, 0.224, 0.225), use_denseCL=False):
        super().__init__()
        self.projection_dim = projection_dim
        self.feature_size = feature_size
        self.temperature = temperature
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.backbone = backbone
        self.in_chan = in_chan
        self.n_class = n_class
        self.use_denseDL = use_denseCL
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.in_chan, self.feature_size),
            nn.ReLU(inplace=True)
        )
        self.projector = Projector(feature_size, hidden_dim=feature_size, out_dim=projection_dim)
        self.decoder = Classifier(in_chan=self.in_chan, n_class=self.n_class)
        self.augment = T.Compose([
                T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
                ])
        self.denseDL= dense_projection_head(in_channel=self.in_chan, out_channel=projection_dim, temperature=self.temperature,dense_size=9, num_neg=50)

    def forward(self, x0, x1, x2, label):
        size = x0.shape[2:]

        x0, x1, x2 = self.backbone(x0), self.backbone(x1), self.backbone(x2)
        f0, f1, f2 = x0, x1, x2

        feat0 = F.interpolate(x0, size=size, mode='bicubic', align_corners=True)
        feat1 = F.interpolate(x1, size=size, mode='bicubic', align_corners=True)
        feat2 = F.interpolate(x2, size=size, mode='bicubic', align_corners=True)


        x0, x1, x2 = self.avg_pool(x0).squeeze(2).squeeze(2), self.avg_pool(x1).squeeze(2).squeeze(2), self.avg_pool(x2).squeeze(2).squeeze(2)
        x0, x1, x2 = self.fc(x0), self.fc(x1), self.fc(x2)

        z0, z1, z2 = self.projector(x0), self.projector(x1), self.projector(x2)

        cl_logits, targets = nt_xent_loss(z0, z1, z2, self.temperature)

        cd_map1 = self.decoder(feat1, feat2)
        cd_map2 = self.decoder(feat1, feat2)

        if self.use_denseDL:
            dense_logits, dense_label, n = self.denseDL(f0, f1, f2, label)
            return cl_logits, targets, dense_logits, dense_label, n, cd_map1, cd_map2
        else:
            return cl_logits, targets, cd_map1, cd_map2
    
    @torch.no_grad()
    def eval(self):
        super().eval()
        self.backbone = nn.Sequential(self.backbone, self.projector.layer1)


def nt_xent_loss(z0, z1, z2, temperature=0.5):
    """ NT-Xent loss """
    z0 = F.normalize(z0, dim=1)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device
    representations = torch.cat([z0, z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    l_pos = l_pos[:N]
    r_pos = torch.diag(similarity_matrix, -N)
    r_pos = r_pos[:N]

    positives = torch.cat([l_pos, r_pos]).view(2 * N, -1)

    diag = torch.eye(3*N, dtype=torch.bool, device=device)
    diag[:N,N:2*N] = diag[N:2*N,:N] = diag[:N,:N]
    diag[2*N:,:] = True

    negatives = similarity_matrix[~diag].view(2*N, -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64)
    return logits, labels

class dense_projection_head(nn.Module):
    def __init__(self, in_channel=32, out_channel=128, temperature=0.5, dense_size=9, num_neg=50):
        super(dense_projection_head, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.temperature = temperature
        self.dense_size = dense_size
        self.num_neg = num_neg

        self.avgpooling  = nn.AdaptiveAvgPool2d(self.dense_size)
        self.maxpooling = nn.AdaptiveMaxPool2d(self.dense_size)
        self.conv1x1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=3, padding=1)

    def forward(self, f0, f1, f2, label):

        z0, z1, z2 = f0, f1, f2
        z0, z1, z2 = self.conv1x1(z0), self.conv1x1(z1), self.conv1x1(z2)
        z0, z1, z2 = self.avgpooling(z0), self.avgpooling(z1), self.avgpooling(z2)

        label = self.maxpooling(label)
        dense_logits, dense_label, n = self.denseCL_loss(z0, z1, z2, label)
        return dense_logits, dense_label, n

    def denseCL_loss(self, z0, z1, z2, label):
        N,C,H,W = z0.size()
        device = z0.device

        z0 = F.normalize(z0, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z0 = rearrange(z0, 'N C H W -> (N H W) C')
        z1 = rearrange(z1, 'N C H W -> (N H W) C')
        z2 = rearrange(z2, 'N C H W -> (N H W) C')
        label = rearrange(label, 'N C H W -> (N H W) C').squeeze()

        representations = torch.cat([z0, z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N*H*W)
        l_pos = l_pos[:N*H*W]
        r_pos = torch.diag(similarity_matrix, -N*H*W)
        r_pos = r_pos[:N*H*W]
        l_neg = torch.diag(similarity_matrix, 2*N*H*W)
        r_neg = torch.diag(similarity_matrix, N*H*W)
        r_neg = r_neg[N*H*W:]

        label_bool = label == 1
        l_pos = l_pos[label_bool]
        r_pos = r_pos[label_bool]
        l_neg = l_neg[label_bool]
        r_neg = r_neg[label_bool]

        n = l_pos.size()[0]
        n = np.array(n)
        n = torch.from_numpy(n).to(device)

        diag = torch.eye(3*N*H*W, dtype=torch.bool, device=device)
        diag[:N*H*W, N*H*W:2*N*H*W] = diag[:N*H*W, 2*N*H*W:] = diag[N*H*W:2*N*H*W, :N*H*W] = diag[N*H*W:2*N*H*W, 2*N*H*W:] = diag[:N*H*W, :N*H*W]
        diag[2*N*H*W:, :] = True

        positives = torch.cat([l_pos, r_pos]).unsqueeze(1)
        negatives_temp = torch.cat([l_neg, r_neg]).unsqueeze(1)
        label_bool = torch.cat([label_bool, label_bool], dim=0)
        negatives = similarity_matrix[~diag].view(2*N*H*W,-1)

        negatives = negatives[label_bool,:]
        negatives,_ = torch.sort(negatives, dim=1, descending=True)
 
        negatives = negatives[:, :self.num_neg]

        dense_logits = torch.cat([positives, negatives_temp, negatives], dim=1)
        dense_logits /= self.temperature

        dense_labels = torch.zeros(2*n.item(), device=device, dtype=torch.int64)

        return dense_logits, dense_labels, n


class Projector(nn.Module):
    """ Projector for SimCLR v2 """
    def __init__(self, in_dim, hidden_dim=64, out_dim=128):
        super().__init__()
        
        self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer3 = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim),
                    nn.BatchNorm1d(out_dim, eps=1e-5, affine=True),
                    )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 


class Classifier(nn.Module):
    def __init__(self, in_chan=64, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.SyncBatchNorm(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.head(x)
        return x