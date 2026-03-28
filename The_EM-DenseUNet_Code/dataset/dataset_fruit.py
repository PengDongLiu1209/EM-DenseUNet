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


        trans = [T.RandomResize(rcrop_size, rcrop_size)] 
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
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'images', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        mask = Image.open(self.imgs_path[index].replace('images', 'masks').replace('.jpg', '.png')) 
        #看情况选择是否要转化为RGB和L模式
        # image = image.convert('RGB')
        # mask = mask.convert('L')
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 修复标签值：将255映射为1（二值分割任务）
        # 确保在transform之后处理，此时mask已经是tensor
        mask[mask == 255] = 1
            
        return image, mask

class MyDataset_StrawDI_Db1(Dataset):
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'img', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        mask = Image.open(self.imgs_path[index].replace('img', 'label').replace('.jpg', '.png')) 
        #看情况选择是否要转化为RGB和L模式
        # image = image.convert('RGB')
        # mask = mask.convert('L')
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 对mask进行二值化：大于0的值全部变为1
        mask[mask > 0] = 1
        return image, mask
    
class MyDataset_Fruit_Segment(Dataset):
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'images', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        mask = Image.open(self.imgs_path[index].replace('images', 'masks').replace('.jpg', '.png')) 
        #看情况选择是否要转化为RGB和L模式
        # image = image.convert('RGB')
        # mask = mask.convert('L')
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 修复标签值：将255映射为1（二值分割任务）
        # 确保在transform之后处理，此时mask已经是tensor
        mask[mask == 255] = 1
        return image, mask

class MyDataset_FruitSeg30(Dataset):
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'images', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        mask = Image.open(self.imgs_path[index].replace('images', 'label').replace('.jpg', '_mask.png')) 
        #看情况选择是否要转化为RGB和L模式
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        # 修复标签值：将255映射为1（二值分割任务）
        # 确保在transform之后处理，此时mask已经是tensor
        #如果掩码标注文件肉眼清晰可见，则mask[mask == 255] = 1放在if语句外，每次都执行
        mask[mask == 255] = 1
        return image, mask

class MyDataset_Fruit_5(Dataset):
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'imgs', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        mask = Image.open(self.imgs_path[index].replace('imgs', 'masks').replace('.jpg', '.png')) 
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            # 确保在transform之后处理，此时mask已经是tensor
            #如果掩码标注文件肉眼不可见，则mask[mask == 255] = 1放在if语句内，不保证每次都执行
            mask[mask == 255] = 1
        return image, mask

class MyDataset_Fruit_9(Dataset):
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'imgs', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        mask = Image.open(self.imgs_path[index].replace('imgs', 'masks').replace('.jpg', '.png')) 
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            # 确保在transform之后处理，此时mask已经是tensor
        #如果掩码标注文件肉眼清晰可见，则mask[mask == 255] = 1放在if语句外，每次都执行
        mask[mask == 255] = 1
        return image, mask

class MyDataset_COCO_12(Dataset):
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'imgs', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        image = image.convert('RGB')  # 确保图像为RGB模式
        mask = Image.open(self.imgs_path[index].replace('imgs', 'masks').replace('.jpg', '.png')) 
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            # 确保在transform之后处理，此时mask已经是tensor
        #如果掩码标注文件肉眼清晰可见，则mask[mask == 255] = 1放在if语句外，每次都执行
        mask[mask == 255] = 1
        return image, mask

class MyDataset_Fruit_13(Dataset):
    def __init__(self,imgs_path,transform=None):
        # 直接查找 imgs_path/iimagesmgs 下的图片
        self.imgs_path = glob(os.path.join(imgs_path, 'imgs', '*.*g'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index])  # 读取图片
        #image = image.convert('RGB')  # 确保图像为RGB模式
        mask = Image.open(self.imgs_path[index].replace('imgs', 'masks').replace('.jpg', '.png')) 
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            # 确保在transform之后处理，此时mask已经是tensor
        #如果掩码标注文件肉眼清晰可见，则mask[mask == 255] = 1放在if语句外，每次都执行
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
    def __len__(self):
        return len(self.data) 
       
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    

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



if __name__ == '__main__':
    # 测试样例：检查数据集是否能正确读取图片和标签
    # base_dir = "F:/data_demo"
    # split = "blue"
    # 训练时：传True，做数据增强
    base_dir = "F:/data_demo/data_new_blue"
    # 训练数据集
    dataset = MyDataset(base_dir, True)
    test_base_dir = "F:/data_demo/data_new_new_test"
    test_dataset = MyDataset(test_base_dir, False)
    
    img, mask = dataset[0]  # 触发 __getitem__，会有打印输出
