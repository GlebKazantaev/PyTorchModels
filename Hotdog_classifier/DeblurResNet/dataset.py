import os
import torch
import random

from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def random_crop_image(images, h, w):
    if type(images) is not list:
        images = [images]

    img_w, img_h = images[0].size
    for img in images:
        _w, _h = img.size
        assert _w == img_w and _h == img_h

    rand_h, rand_w = random.randint(0, img_h - h), random.randint(0, img_w - w)

    crop_images = []
    for img in images:
        crop_images.append(transforms.functional.crop(img, rand_h, rand_w, h, w))

    if len(crop_images) == 1:
        return crop_images[0]
    return crop_images


def split_images(images, h, w):
    if type(images) is not list:
        images = [images]

    img_w, img_h = images[0].size
    for img in images:
        _w, _h = img.size
        assert _w == img_w and _h == img_h

    #random.randint(0, img_h - h), random.randint(0, img_w - w)

    crop_images = [[] for x in range(len(images))]
    id = 0
    for img in images:
        cur_h, cur_w = 0, 0
        while cur_h + h <= img_h:
            cur_w = 0
            while cur_w + w <= img_w:
                crop_images[id].append(transforms.functional.crop(img, cur_h, cur_w, h, w))
                cur_w += w
            cur_h += h
        id += 1

    if len(images) == 1:
        return crop_images[0][0]
    return crop_images


class DeblurDataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.pref = "train" if train else "test"
        self.root_dir = os.path.join(root_dir, self.pref)
        self.transform = transform

        image_paths = os.listdir(self.root_dir)

        # Load data set
        self.in_images = []
        self.ref_images = []

        for img_path in image_paths:
            blur_images_path = os.path.join(self.root_dir, img_path, "blur")
            blur_gamma_images_path = os.path.join(self.root_dir, img_path, "blur_gamma")
            sharp_images_path = os.path.join(self.root_dir, img_path, "sharp")

            blur_images_list = os.listdir(blur_images_path)
            blur_gamma_images_list = os.listdir(blur_gamma_images_path)
            sharp_images_list = os.listdir(sharp_images_path)

            for in_image, ref_image in zip(blur_images_list, sharp_images_list):
                self.in_images.append(os.path.join(blur_images_path, in_image))
                self.ref_images.append(os.path.join(sharp_images_path, ref_image))

            for in_image, ref_image in zip(blur_gamma_images_list, sharp_images_list):
                self.in_images.append(os.path.join(blur_images_path, in_image))
                self.ref_images.append(os.path.join(sharp_images_path, ref_image))

        self.dataset_len = len(self.in_images)
        print("Dataset {} has {} images".format(self.pref, self.dataset_len))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        in_img_name = self.in_images[idx]
        ref_img_name = self.ref_images[idx]

        in_image = Image.open(in_img_name).convert('RGB')
        ref_image = Image.open(ref_img_name).convert('RGB')

        #in_image, ref_image = random_crop_image([in_image, ref_image], 256, 256)
        splited = split_images([in_image, ref_image], 256, 256)
        in_image, ref_image = splited[0], splited[1]

        sample = {'image': in_image, 'reference': ref_image}
        if self.transform:
            for id in range(len(in_image)):
                #sample['image'][id].save('img_{}.bmp'.format(id))
                #sample['reference'][id].save('reference_{}.bmp'.format(id))
                sample['image'][id] = self.transform(in_image[id])
                sample['reference'][id] = self.transform(ref_image[id])

        return {'image': torch.stack([crop for crop in sample['image']]),
                'reference': torch.stack([crop for crop in sample['reference']])}
