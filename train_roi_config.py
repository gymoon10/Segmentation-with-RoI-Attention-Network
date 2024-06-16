"""
Set training options.
"""
import copy
import os

import torch
from utils import transforms as my_transforms

DATASET_DIR = 'C:/Users/iml/Desktop/CropMatch/DATA/Labeled'  # train(student&teacher) dir
DATASET_NAME = 'CornDataset'

#DATASET_DIR = 'F:/cityscapes/teacher'  # train(student&teacher) dir
#DATASET_NAME = 'CityscapesDataset'


args = dict(
    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='D:/SegRoI/save/weight',

    # resume training network
    resume_path=None,
    pretrained_path='D:/SegRoI/save/pretrained2/weight/best_disease_model_746.pth',
    # pretrained_path='D:/SegRoI/save/weight/best_disease_model_166.pth',

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
        'name': "ERFNet_Semantic_RoI",
        'kwargs': {
            'num_classes': 3,  # 3 for bg/plant/disease
        }
    },


    lr=0.0005,  # default:1e-3
    n_epochs=800,  # every x epochs, report train & validation set

)


def get_args():
    return copy.deepcopy(args)
