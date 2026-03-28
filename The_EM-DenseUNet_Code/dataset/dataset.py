import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import dataset.tfs as T
import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from glob import glob

class SegmentationPresetTrain:
    def __init__(self,rcrop_size=224, hflip_prob=0.5, vflip_prob=0.5,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):


        trans = [T.RandomResize(rcrop_size,rcrop_size)] 
        if hflip_prob > 0:  # 水平翻转
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:  # 垂直翻转
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(rcrop_size),         # 随机裁剪,需要保证一个batch 的 size相等
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),  # normalization
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


# 验证集预处理

class SegmentationPresetVal:
    def __init__(self,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),rcrop_size=224):
        self.transforms = T.Compose([
            T.Resize(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),  # normalization
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

# 测试集预处理
class SegmentationPresetTest:
    def __init__(self,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),rcrop_size=224):
        self.transforms = T.Compose([
            T.Resize(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),  # normalization
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class MyDataset(Dataset):
    def __init__(self, imgs_path, transform=None, normalize_to_01=True):
        # 直接查找 imgs_path/images 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'images', '*.*g'))
        self.transform = transform
        self.normalize_to_01 = normalize_to_01
        
        # 如果需要标准化，计算数据集的统计信息
        if self.normalize_to_01:
            self.mean, self.std = self._compute_dataset_stats()

    def _compute_dataset_stats(self):
        """计算数据集的均值和标准差"""
        print("正在计算数据集统计信息...")
        pixel_sum = torch.zeros(3)
        pixel_squared_sum = torch.zeros(3)
        num_pixels = 0
        
        # 采样部分图片来计算统计信息（避免内存问题，适合4090显卡）
        sample_size = min(200, len(self.imgs_path))  # 4090显存大，可以采样更多
        sample_indices = np.random.choice(len(self.imgs_path), sample_size, replace=False)
        
        for idx in sample_indices:
            image = Image.open(self.imgs_path[idx]).convert('RGB')
            image_tensor = torch.tensor(np.array(image)).float() / 255.0  # 转换到[0,1]
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
            
            pixel_sum += image_tensor.sum(dim=[1, 2])
            pixel_squared_sum += (image_tensor ** 2).sum(dim=[1, 2])
            num_pixels += image_tensor.shape[1] * image_tensor.shape[2]
        
        mean = pixel_sum / num_pixels
        std = torch.sqrt(pixel_squared_sum / num_pixels - mean ** 2)
        
        print(f"计算得到的均值: {mean.tolist()}")
        print(f"计算得到的标准差: {std.tolist()}")
        
        return mean.tolist(), std.tolist()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        mask = Image.open(self.imgs_path[index].replace('images', 'masks').replace('.jpg', '.png')) 
        
        # 看情况选择是否要转化为RGB和L模式
        # image = image.convert('RGB')
        # mask = mask.convert('L')
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 标准化到0-1正态分布（减去均值，除以标准差）
        if self.normalize_to_01 and isinstance(image, torch.Tensor):
            for c in range(image.shape[0]):  # 对每个通道分别标准化
                image[c] = (image[c] - self.mean[c]) / self.std[c]
        
        # 修复标签值：将255映射为1（保持原有逻辑）
        # 确保在transform之后处理，此时mask已经是tensor
        if isinstance(mask, torch.Tensor):
            mask[mask == 255] = 1
            
        return image, mask



class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
