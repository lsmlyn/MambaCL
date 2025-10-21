# coding=utf-8
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data
from models.MambaCL.net import CLCDNet,Predict
from train_options import parser
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder, calMetric_iou
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed_tool import setup, cleanup, run_distribute, reduce_mean
from PIL import Image


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
    test_set = TestDatasetFromFolder(opt.path_test_img1, opt.path_test_img2, opt.path_test_lab)
    test_loader = DataLoader(dataset=test_set,
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

    # #
    netCD.load_state_dict(torch.load("epochs/LEVIR/mambacl_S9_M50/netCD_epoch_79.pth"))
    netCD = Predict(encoder=netCD.module.encoder, decoder=netCD.module.clnet.decoder, n_class=2,in_chan=32).to(rank)
    netCD.eval()

    save_dir = "/results/LEVIR/mambacl/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    test_bar = tqdm(test_loader)
    inter = 0
    unin = 0
    test_results = { 'batch_sizes': 0, 'IoU': 0, 'f1': 0}

    for image1, image2, label, image_name in test_bar:
        test_results['batch_sizes'] += 1
        # print(test_results['batch_sizes'])

        image1 = image1.cuda(rank, non_blocking=True)
        image2 = image2.cuda(rank, non_blocking=True)
        label = label.cuda(rank, non_blocking=True)

        label = torch.argmax(label, 1).unsqueeze(1).float()

        result = netCD(image1, image2)

        prob = torch.argmax(result, 1).unsqueeze(1)

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
        test_results['IoU'] = (inter * 1.0 / unin)


        test_bar.set_description(
            desc='IoU: %.4f' % ( test_results['IoU'] ))

        result = result*255
        result = Image.fromarray(result.astype('uint8'))
        result.save(save_dir + image_name[0])


    cleanup()

if __name__ == '__main__':
    run_distribute(main_worker, 1)