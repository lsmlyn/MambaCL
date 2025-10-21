#coding=utf-8
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os

def Convert2Binary(image):
    image = image.convert('L')
    for x in range(512):
        for y in range(512):
            # 如果像素值为255，则将其设置为1，否则设置为0
            if image.getpixel((x, y)) == 255:
                image.putpixel((x, y), 1)
            else:
                image.putpixel((x, y), 0)
    return image

def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result

def get_transform(convert=True, normalize=False, aug=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if aug:
        transform_list += [transforms.RandomApply([transforms.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8)]
        transform_list += [transforms.RandomApply([transforms.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8)]
        transform_list += [transforms.RandomGrayscale(p=0.2)]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, Image_dir0, Image_dir1, Image_dir2, Label_dir):
        super(TrainDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir0)
        self.image_filenames0 = [join(Image_dir0, x) for x in datalist if is_image_file(x)]
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.data_augment = trainImageAug(crop=False, augment = True, angle = 30)
        self.img_transform0 = get_transform(convert=True, normalize=True, aug=True)
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):
        image0 = Image.open(self.image_filenames0[index]).convert('RGB')
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index]).convert('L')
        image0, image1, image2, label = self.data_augment(image0, image1, image2, label)
        image0, image1, image2 = self.img_transform0(image0), self.img_transform(image1), self.img_transform(image2)
        # label = Convert2Binary(label)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        return image0, image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)

    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        image0, image1, image2, label = tuple(zip(*batch))

        image0 = torch.stack(image0, dim=0)
        image1 = torch.stack(image1, dim=0)
        image2 = torch.stack(image2, dim=0)
        label = torch.as_tensor(label)
        return image0, image1, image2, label


class TrainDatasetFromFolder_suixi(Dataset):
    def __init__(self, Image_dir0, Image_dir1, Image_dir2, Label_dir):
        super(TrainDatasetFromFolder_suixi, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir0)
        self.image_filenames0 = [join(Image_dir0, x) for x in datalist if is_image_file(x)]
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.data_augment = trainImageAug(crop=False, augment = True, angle = 30)
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):
        image0 = Image.open(self.image_filenames0[index]).convert('RGB')
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index]).convert('L')
        image0, image1, image2, label = self.data_augment(image0, image1, image2, label)
        image0, image1, image2 = self.img_transform(image0), self.img_transform(image1), self.img_transform(image2)
        # label = Convert2Binary(label)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        return image0, image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)

    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        image0, image1, image2, label = tuple(zip(*batch))

        image0 = torch.stack(image0, dim=0)
        image1 = torch.stack(image1, dim=0)
        image2 = torch.stack(image2, dim=0)
        label = torch.as_tensor(label)
        return image0, image1, image2, label


class ValDatasetFromFolder(Dataset):
    def __init__(self, Image_dir0, Image_dir1, Image_dir2, Label_dir):
        super(ValDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames0 = [join(Image_dir0, x) for x in datalist if is_image_file(x)]
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.img_transform0 = get_transform(convert=True, normalize=True, aug=True)
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):
        image0 = Image.open(self.image_filenames0[index]).convert('RGB')
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index]).convert('L')
        image0, image1, image2 = self.img_transform0(image0), self.img_transform(image1), self.img_transform(image2)
        # label = Convert2Binary(label)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        return image0, image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)


class ValDatasetFromFolder_suixi(Dataset):
    def __init__(self, Image_dir0, Image_dir1, Image_dir2, Label_dir):
        super(ValDatasetFromFolder_suixi, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames0 = [join(Image_dir0, x) for x in datalist if is_image_file(x)]
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):
        image0 = Image.open(self.image_filenames0[index]).convert('RGB')
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index]).convert('L')
        image0, image1, image2 = self.img_transform(image0), self.img_transform(image1), self.img_transform(image2)
        # label = Convert2Binary(label)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        return image0, image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)

class TestDatasetFromFolder(Dataset):
    def __init__(self, Image_dir1, Image_dir2, Label_dir):
        super(TestDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.lab_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):

        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        label = Image.open(self.label_filenames[index]).convert('L')
        # label = Convert2Binary(label)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        image_name =  self.image_filenames1[index].split('/', -1)
        image_name = image_name[len(image_name)-1]

        return image1, image2, label, image_name

    def __len__(self):
        return len(self.image_filenames1)

class TestDatasetFromFolder_BU(Dataset):
    def __init__(self,Image_dir0, Image_dir1, Image_dir2, Label_dir):
        super(TestDatasetFromFolder_BU, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames0 = [join(Image_dir0, x) for x in datalist if is_image_file(x)]
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.transform0 = get_transform(convert=True, normalize=True, aug=True)
        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.lab_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):
        image0 = Image.open(self.image_filenames0[index]).convert('RGB')
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        image0 = self.transform(image0)
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        label = Image.open(self.label_filenames[index]).convert('L')
        # label = Convert2Binary(label)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        image_name =  self.image_filenames1[index].split('/', -1)
        image_name = image_name[len(image_name)-1]

        return image0, image1, image2, label, image_name

    def __len__(self):
        return len(self.image_filenames1)

class trainImageAug(object):
    def __init__(self, crop = True, augment = True, angle = 30):
        self.crop =crop
        self.augment = augment
        self.angle = angle

    def __call__(self, image0, image1, image2, mask):
        if self.crop:
            w = np.random.randint(0,256)
            h = np.random.randint(0,256)
            box = (w, h, w+256, h+256)
            image0 = image0.crop(box)
            image1 = image1.crop(box)
            image2 = image2.crop(box)
            mask = mask.crop(box)
        if self.augment:
            prop = np.random.uniform(0, 1)
            if prop < 0.15:
                image0 = image0.transpose(Image.FLIP_LEFT_RIGHT)
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif prop < 0.3:
                image0 = image0.transpose(Image.FLIP_TOP_BOTTOM)
                image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
                image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            elif prop < 0.5:
                image0 = image0.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                image1 = image1.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                image2 = image2.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                mask = mask.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))

        return image0, image1, image2, mask