import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from numpy.core.fromnumeric import transpose
from scipy import ndimage, misc
from PIL import Image
from torch.utils.data import Dataset


h, w = 512, 512


def random_background(img_pil):
    img_pil = img_pil.convert('RGBA')

    key = np.random.choice([0, 1, 2, 3])

    if key == 0:
        bg = Image.new('RGBA', img_pil.size, (255,) * 4)  # White image
    elif key == 1:
        bg = Image.new('RGBA', img_pil.size, (0, 0, 0, 255))  # Black image
    elif key == 2:
        img_np = np.array(img_pil)
        mean_color = img_np.mean((0, 1))
        bg = Image.new('RGBA', img_pil.size,
                       (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]), 255))  # mean color
    elif key == 3:
        bg = Image.new('RGBA', img_pil.size,
                       (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255))  # random color

    img_pil = Image.blend(img_pil, bg, 0.25)  # blend
    img_out = img_pil.convert('RGB')

    return img_out


def random_gamma(img_pil):
    gain = 1
    gamma_range = [0.7, 1.8]
    min_gamma = gamma_range[0]
    max_gamma = gamma_range[1]
    gamma = np.random.rand() * (max_gamma - min_gamma) + min_gamma

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img_out = img_pil.point(gamma_map)

    return img_out


class CornDataset(Dataset):
    ''' returns images, labels(GT) - general dataset for class-segmentation'''
    def __init__(self, root_dir='./', type_="train", size=None, transform=None):
        self.root_dir = root_dir
        self.type = type_

        # get image, foreground and instance list
        image_list = glob.glob(os.path.join(self.root_dir, self.type, '*.jpg'))
        image_list.sort()
        self.image_list = image_list
        print("# image files: ", len(image_list))

        label_list = glob.glob(os.path.join(self.root_dir, self.type, '*_overall.png'))
        label_list.sort()
        self.label_list = label_list
        print("# label(bg/plant/disease) files: ", len(label_list))

        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        self.jitter = transforms.ColorJitter(brightness=0.1,
                                             contrast=0.1,
                                             saturation=0.1,
                                             hue=0.1)

        print('Corn Dataset(Teacher) created [{}]'.format(self.type))

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image and foreground
        image = Image.open(self.image_list[index]).convert('RGB')
        image = image.resize((h, w), resample=Image.Resampling.BILINEAR)

        width, height = image.size

        sample['image'] = image.copy()
        sample['im_name'] = self.image_list[index]

        label_map = skimage.io.imread(self.label_list[index])  # := instance map
        label_map = cv2.resize(label_map, (h, w), interpolation=cv2.INTER_NEAREST)
        class_ids = np.unique(label_map)[1:]  # no background

        label_all = np.zeros((h, w), dtype=np.uint8)

        for idx in class_ids:
            if idx == 10:
                mask_plant = (label_map == idx)
                label_all[mask_plant] = 1

            elif idx == 20:
                mask_disease = (label_map == idx)
                label_all[mask_disease] = 2

        # ---------------------------- data augmentation ----------------------------
        if 'train' in self.type:
            # ---------------------------- random hflip ----------------------------
            # 1.original
            if random.random() > 0.5:
                # FLIP_TOP_BOTTOM
                sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
                label_all = np.fliplr(label_all)

            # ----------------------------  random vflip ----------------------------
            # 1.original
            if random.random() > 0.5:
                # FLIP_LEFT_RIGHT
                sample['image'] = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
                label_all = np.flipud(label_all)

            # ----------------------------  rotate 90 - clockwise ----------------------------
            # 1.original
            if random.random() > 0.5:
                img_np = np.array(sample['image'])
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
                sample['image'] = Image.fromarray(img_np)
                label_all = cv2.rotate(label_all, cv2.ROTATE_90_CLOCKWISE)

            # ---------------------------- rotate 90 - counterclockwise ----------------------------
            # 1.original
            if random.random() > 0.5:
                img_np = np.array(sample['image'])
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                sample['image'] = Image.fromarray(img_np)
                label_all = cv2.rotate(label_all, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ---------------------------- random background ----------------------------
            # 1.original
            if random.random() > 0.5:
                sample['image'] = random_background(img_pil=sample['image'])

            # ---------------------------- random gamma ----------------------------
            # 1.original
            if random.random() > 0.5:
                sample['image'] = random_gamma(img_pil=sample['image'])

            # ---------------------------- random grayscaling ----------------------------
            # 1.original
            #grayscaler = transforms.RandomGrayscale(p=1.0)
            #if random.random() > 0.9:
            #    sample['image'] = grayscaler(sample['image'])


        if 'train' in self.type:
            # 1.original
            label_all = Image.fromarray(np.uint8(label_all))
            sample['label_all'] = label_all

        else:
            # 1.original
            label_all = Image.fromarray(np.uint8(label_all))
            sample['label_all'] = label_all

        # transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample