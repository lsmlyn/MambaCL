import argparse

#training options
parser = argparse.ArgumentParser(description='Train Change Detection Models')

# training parameters
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=2, type=int, help='batchsize for validation')

parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--img_chan', default=3, type=int, help='channel of input image')
parser.add_argument('--image_size', default=512, type=int, help='size of input image')
parser.add_argument('--in_chan', default=32, type=int, help='channel of output image of encoder')
parser.add_argument('--feature_size', default=64, type=int, help='channel of input image')
parser.add_argument('--projection_dim', default=128, type=int, help='dim of projection')
parser.add_argument('--temperature', default=0.7, type=int, help='temperature')
parser.add_argument('--n_hybridloss', default=0.1, type=int, help='n_hybridloss')

parser.add_argument('--gpu_id', default="2", type=str, help='which gpu to run.')
parser.add_argument('--crop_size', default=512, type=str, help='which gpu to run.')
parser.add_argument('--num_workers', default=24, type=int, help='num_workers')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--niter', type=int, default = 50, help='momentum term of adam')
parser.add_argument('--pretrained', type=int, default=0, help='use pretrained model')
parser.add_argument('--pre_num', type=int, default=0, help='pre_num')

parser.add_argument('--use_denseCL', type=int, default=1, help='use denseCL')

# path for loading data
parser.add_argument('--path_img0', default='Data/LEVIR-CD/LEVIR-CD-512/train/A', type=str, help='path_img0')
parser.add_argument('--path_img1', default='Data/LEVIR-CD/LEVIR-CD-512/train/A', type=str, help='path_img1')
parser.add_argument('--path_img2', default='Data/LEVIR-CD/LEVIR-CD-512/train/B', type=str, help='path_img2')
parser.add_argument('--path_lab', default='Data/LEVIR-CD/LEVIR-CD-512/train/label', type=str, help='path_lab')
parser.add_argument('--path_val_img1', default='Data/LEVIR-CD/LEVIR-CD-512/val/A', type=str, help='path_val_img1')
parser.add_argument('--path_val_img2', default='Data/LEVIR-CD/LEVIR-CD-512/val/B', type=str, help='path_val_img2')
parser.add_argument('--path_val_lab', default='Data/LEVIR-CD/LEVIR-CD-512/val/label', type=str, help='path_val_lab')
parser.add_argument('--path_test_img1', default='Data/LEVIR-CD/LEVIR-CD-512/test/A', type=str, help='path_test_img1')
parser.add_argument('--path_test_img2', default='Data/LEVIR-CD/LEVIR-CD-512/test/B', type=str, help='path_test_img2')
parser.add_argument('--path_test_lab', default='Data/LEVIR-CD/LEVIR-CD-512/test/label', type=str, help='path_test_lab')


parser.add_argument('--model_dir', default='epochs/LEVIR/mambacl_S9_M50/', type=str, help='model_dir')
parser.add_argument('--sta_dir', default='epochs/LEVIR/mambacl_S9_M50/mambacl_S9_M50.txt', type=str, help='sta_dir')
parser.add_argument('--save_dir', default='epochs/LEVIR/mambacl_S9_M50/', type=str, help='save_dir')