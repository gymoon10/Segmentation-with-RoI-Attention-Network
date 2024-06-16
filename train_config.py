"""
Set training options.
"""
import copy
import os

import torch
from utils import transforms as my_transforms

DATASET_DIR = 'C:/Users/Kim/Desktop/SegRoI/DATA/Labeled'  # train(student&teacher) dir
DATASET_NAME = 'CornDataset'

#DATASET_DIR = 'F:/cityscapes/teacher'  # train(student&teacher) dir
#DATASET_NAME = 'CityscapesDataset'


args = dict(
    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='C:/Users/Kim/Desktop/SegRoI/save/weight',

    # resume training network
    resume_path=None,
    # pretrained_path='D:/CropMatch/train_ce/erfnet3/weight/best_disease_model_387.pth',
    #pretrained_path='C:/Users/Kim/Desktop/SegRoI/save/weight/pre74.pth',
    pretrained_path=None,

    train_dataset={
        'name': DATASET_NAME,
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'train',
            'size': None,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'label_all'],
                        'type': [torch.FloatTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 3,  # 3, 6
        'workers': 0
    },

    val_dataset={
        'name': DATASET_NAME,
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'label_all'],
                        'type': [torch.FloatTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 0
    },

    model={
        'name': "TransAttUNet",
        'kwargs': {
            'num_classes': 3,  # 3 for bg/plant/disease
        }
    },


    lr=1e-3,
    # lr=0.00555,
    # lr=1e-1 ,
    n_epochs=800,  # every x epochs, report train & validation set

)


def get_args():
    return copy.deepcopy(args)
