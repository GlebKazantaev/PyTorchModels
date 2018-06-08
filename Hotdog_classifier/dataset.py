import os
import torch
import numpy as np

from skimage import io, transform
from torch.utils.data import Dataset


class HotDogsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, train, transform=None):
        self.root_dir = root_dir #+ ("/train" if train else "/test")
        self.pref = "train" if train else "test"
        self.transform = transform

        self.hot_dog_path = self.root_dir + "/{}/hotdog/".format(self.pref)
        self.not_hot_dog_path = self.root_dir + "/{}/not_hotdog/".format(self.pref)

        self.dataset_len = len(os.listdir(self.hot_dog_path)) + len(os.listdir(self.not_hot_dog_path))

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


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        res = {**sample}
        res['image'] = img
        return res


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        #print(image.shape, " ", label)
        res = {**sample}
        res['image'] = torch.from_numpy(image.transpose((2, 0, 1)))
        return res


class HotDogsDatasetEval(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_len = len(os.listdir(self.root_dir))

        # Load data set
        self.images = []
        self.names = []
        for x in os.listdir(self.root_dir):
            self.images.append(self.root_dir + '/' + x)
            self.names.append(x)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        img_name = os.path.join(self.images[idx])
        image = io.imread(img_name)
        sample = {'image': image, 'name': self.names[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample