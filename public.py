from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from PIL import Image
import SimpleITK as sitk
import numpy as np

# 公共参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['background', 'PS', 'FH']
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def image_reader(img_path, isMask):
    img = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(img)  # (横断面Depth, 冠状面Height, 矢状面Width)

    if isMask:
        img_rgb = Image.fromarray(nda, 'L')  # 矩阵数据转灰度图
        img_array = nda
    else:
        img_array = np.transpose(nda, (1, 2, 0))  # 转置为(冠状面Height, 矢状面Width, 横断面Depth)
        img_rgb = Image.fromarray(img_array, 'RGB')  # 转RGB

    return img_rgb, img_array


class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform, augmentation=None, preprocessing=None):
        self.image_paths = image_dir
        self.mask_paths = mask_dir
        self.transform = transform

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img_name = self.image_paths[i]
        mask_name = self.mask_paths[i]

        _, img = image_reader(img_name, False)
        _, mask = image_reader(mask_name, True)

        if self.transform:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        mask = mask.type(torch.LongTensor)
        mask_onehot = F.one_hot(mask, len(CLASSES))  # onehot编码
        mask_stack = torch.stack((mask_onehot[:, :, 0], mask_onehot[:, :, 1], mask_onehot[:, :, 2]))  # 三个标签onehot在第一维
        mask_stack = mask_stack.type(torch.IntTensor)

        return img, mask_stack

    def __len__(self):
        return len(self.image_paths)
