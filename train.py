# coding=utf-8
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.functional import F
from tqdm import tqdm

from models.MambaCL.net import CLCDNet,Predict
from train_options import parser
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder, calMetric_iou
from loss.losses import cross_entropy
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed_tool import setup, cleanup, run_distribute, reduce_mean


opt = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def main_worker(rank, world_size):
    print(f"Running basic DistributedDataParallel example on rank {rank}.")
    setup(rank, world_size)
    batchsize = 2
    train_set = TrainDatasetFromFolder(opt.path_img0, opt.path_img1, opt.path_img2, opt.path_lab)
    val_set = ValDatasetFromFolder(opt.path_val_img1, opt.path_val_img1, opt.path_val_img2, opt.path_val_lab)

    train_sampler = DistributedSampler(dataset=train_set,num_replicas=world_size,rank=rank)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batchsize,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=1,
                            shuffle=True,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=True)

    netCD = CLCDNet(feature_size=opt.feature_size, n_class=opt.n_class, img_chan=opt.img_chan,
                    in_chan=opt.in_chan, projection_dim=opt.projection_dim,
                    temperature=opt.temperature, image_size=opt.image_size, use_denseCL=opt.use_denseCL).to(rank)
    netCD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netCD)
    netCD = DDP(netCD, device_ids=[rank],find_unused_parameters=True)

    optimizerCD = optim.Adam(netCD.parameters(), lr=0.0001, betas=(0.9, 0.999))
    CDcriterion = cross_entropy().to(rank, dtype=torch.float)

    NUM_EPOCHS = opt.num_epochs
    val_batchsize = 1
    mloss = 0

    
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = AverageMeter('Loss', ':.4e')
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'CD_loss': 0}
        train_sampler.set_epoch(epoch)
        netCD.train()
    
        for image0, image1, image2, label in train_bar:
            running_results['batch_sizes'] += batchsize
    
            if torch.cuda.is_available():
                image0 = image0.cuda(rank, non_blocking=True)
                image1 = image1.cuda(rank, non_blocking=True)
                image2 = image2.cuda(rank, non_blocking=True)
                label = label.cuda(rank, non_blocking=True)
    
            label = torch.argmax(label, 1).unsqueeze(1).float()
    
            if opt.use_denseCL:
                cl_logits, targets, dense_logits, dense_label, n, result1, result2 = netCD(image0, image1, image2,label)
                denseCL_loss = F.cross_entropy(dense_logits, dense_label) / (2 * n)
            else:
                cl_logits, targets, result1, result2 = netCD(image0, image1, image2, label)
            
            cl_loss = F.cross_entropy(cl_logits, targets) / (2 * batchsize)
            CD_loss = CDcriterion(result2, label)
    
            loss = 0.05 * denseCL_loss + 0.05 * cl_loss + CD_loss
    
            reduced_loss = reduce_mean(loss, world_size)
            losses.update(reduced_loss.item(), image0.size(0))
    
            netCD.zero_grad()
            loss.backward()
            optimizerCD.step()
    
            # loss for current batch before optimization
            running_results['CD_loss'] += loss.item() * batchsize
    
            train_bar.set_description(
                desc='[%d/%d] CD_Loss: %.4f, a=%.4f, b=%d ' % (
                    epoch, NUM_EPOCHS, running_results['CD_loss'] / running_results['batch_sizes'],running_results['CD_loss'], running_results['batch_sizes']
                ))
    
        if rank==0:
            netCD.eval()
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                inter, unin = 0, 0
                valing_results = {'CD_loss': 0, 'batch_sizes': 0}
    
                for image0, image1, image2, label in val_bar:
                    valing_results['batch_sizes'] += val_batchsize
    
                    image0 = image0.to(rank, dtype=torch.float)
                    image1 = image1.to(rank, dtype=torch.float)
                    image2 = image2.to(rank, dtype=torch.float)
                    label = label.to(rank, dtype=torch.float)
    
                    label = torch.argmax(label, 1).unsqueeze(1).float()
                    if opt.use_denseCL:
                        _, _, _, _, _, _, result = netCD(image0, image1, image2, label)
                    else:
                        _, _, _, result = netCD(image0, image1, image2, label)
    
                    prob = torch.argmax(result, 1).unsqueeze(1).float()
    
                    gt_value = (label > 0).float()
                    prob = (prob > 0).float()
                    prob = prob.cpu().detach().numpy()
                    gt_value = gt_value.cpu().detach().numpy()
    
                    gt_value = np.squeeze(gt_value)
                    result = np.squeeze(prob)
    
                    intr, unn = calMetric_iou(gt_value, result)
                    inter = inter + intr
                    unin = unin + unn
                    # loss for current batch before optimization
                    valing_results['IoU'] = (inter * 1.0 / unin)
    
                    val_bar.set_description(
                        desc='IoU: %.4f' % (valing_results['IoU']))
    
                    # save model parameters
                val_loss = valing_results['IoU']
                df = pd.DataFrame({'Epoch': epoch, 'IoU': val_loss}, index=[0])
                df.to_csv(opt.sta_dir, index=False, mode='a', header=False)
    
                if val_loss > mloss or epoch == 1:
                    mloss = val_loss
                    torch.save(netCD.state_dict(), opt.model_dir + 'netCD_epoch_%d.pth' % (epoch + opt.pre_num))

    cleanup()

if __name__ == '__main__':
    run_distribute(main_worker, 1)