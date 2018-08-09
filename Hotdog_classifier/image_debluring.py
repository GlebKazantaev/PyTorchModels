from torchvision import datasets, transforms
from torch.utils.data import Dataset
from skimage import io, transform

import torch.nn as nn
import numpy as np
import torch
import os


class BlurryDataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.root_dir = root_dir
        self.pref = "train" if train else "test"
        self.transform = transform

        self.original_images_path = self.root_dir #$+ "/{}/hotdog/".format(self.pref)
        self.blurred_images_path = self.root_dir #+ "/{}/not_hotdog/".format(self.pref)

        self.dataset_len = 1 #len(os.listdir(self.hot_dog_path)) + len(os.listdir(self.not_hot_dog_path))

        # Load data set
        self.images = []
        self.labels = {}
        for x in os.listdir(self.hot_dog_path):
            self.images.append(self.hot_dog_path + x)
            self.labels.update({self.hot_dog_path + x:1})

        for x in os.listdir(self.not_hot_dog_path):
            self.images.append(self.not_hot_dog_path + x)
            self.labels.update({self.not_hot_dog_path + x: 0})

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        img_name = os.path.join(self.images[idx])
        image = io.imread(img_name)
        #print(img_name, " ", image.shape)
        label = np.array([float(self.labels[img_name])])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out




def main():
    pass






if __name__ == "__main__":
    main()