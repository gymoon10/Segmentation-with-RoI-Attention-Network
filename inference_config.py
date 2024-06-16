"""
Set training options.
"""
import copy
import os

import torch
from utils import transforms as my_transforms

DATASET_DIR = 'C:/Users/Kim/Desktop/SegRoI/DATA/Labeled'  # train(student&teacher) dir
DATASET_NAME = 'CornDataset'


args = dict(
    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='D:/SegRoI/save/inference',

    # resume training network
    # pretrained_path='D:/SegRoI/save/HRNet/weight/best_disease_model_462.pth',
    # pretrained_path='D:/SCRS2/save/train_ce/weight/best_disease_model_579.pth',
    # pretrained_path='D:/SegRoI/save/UNet/weight2/best_disease_model_600.pth',
    # pretrained_path='D:/SegRoI/save/pretrained2/weight/best_disease_model_319.pth',
    pretrained_path='D:/SCRS2/save/train_ce/erfnet_original/weight/best_disease_model_650.pth' ,


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
        'batch_size': 3,  # 3
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
        'name': "ERFNet_Semantic_Original",
        'kwargs': {
            'num_classes': 3,  # 3 for bg/plant/disease
        }
    },


    lr=1e-3,
    n_epochs=1,  # every x epochs, report train & validation set

)


def get_args():
    return copy.deepcopy(args)
