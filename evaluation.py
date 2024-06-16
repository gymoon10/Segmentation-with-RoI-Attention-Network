"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import inference_config

from criterions.loss import CriterionCE  #, CriterionMatching

from datasets import get_dataset
from models import get_model, ERFNet_Semantic_Original
from utils.utils import AverageMeter, Logger, Visualizer  # for CVPPP

import numpy as np
from PIL import Image
from torchvision.utils import save_image


torch.backends.cudnn.benchmark = True

args = inference_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

def calc_dic(n_objects_gt, n_objects_pred):
    return np.abs(n_objects_gt - n_objects_pred)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([-1]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([-1]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice


def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)

    return min(_dice1, _dice2)

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# val dataloader (student)
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set criterion
criterion_val = CriterionCE()
criterion = CriterionCE()

criterion_val = torch.nn.DataParallel(criterion_val).to(device)
criterion = torch.nn.DataParallel(criterion).to(device)


def main():
    # init
    start_epoch = 0
    best_iou_plant = 0
    best_iou_disease = 0
    best_iou_both = 0

    args['model']['name'] = 'GAUNet'
    args['pretrained_path'] = 'C:/Users/Kim/Desktop/Access-Revision/weight/gaunet.pth'

    # set model (student)
    model = get_model(args['model']['name'], args['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)
    if args['pretrained_path']:
        state = torch.load(args['pretrained_path'])
        path = args['pretrained_path']
        print(f'load model from - {path}')
        model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()

    # print the network information
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    # print(model)
    print("The number of parameters: {}".format(num_params))

    for epoch in range(start_epoch, args['n_epochs']):
        print('Starting epoch {}'.format(epoch))

        dice_plant = []
        dice_disease = []

        sbd_plant = []
        sbd_disease = []

        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(val_dataset_it)):
                image = sample['image']  # (N, 3, 512, 512)
                label = sample['label_all'].squeeze(1)  # (N, 512, 512)

                output = model(image)  # (N, 4, h, w)

                pred = output[0, :, :, :].unsqueeze(0).type(torch.float32).cuda()
                pred = pred.detach().max(dim=1)[1]

                pred_plant = (pred == 1).cpu().detach() * 1
                pred_disease = (pred == 2).cpu().detach() * 1

                gt_plant = (label == 1) * 1
                gt_disease = (label == 2) * 1

                dice_plant.append(calc_dice(gt_plant.numpy(), pred_plant.numpy()))
                dice_disease.append(calc_dice(gt_disease.numpy(), pred_disease.numpy()))

                sbd_plant.append(calc_sbd(gt_plant.numpy(), pred_plant.numpy()))
                sbd_disease.append(calc_sbd(gt_disease.numpy(), pred_disease.numpy()))

            print('Dice-Plant :', np.mean(dice_plant))
            print('Dice-Disease :', np.mean(dice_disease))
            print('____________________________________________')
            print('SBD-Plant :', np.mean(sbd_plant))
            print('SBD-Disease :', np.mean(sbd_disease))

if __name__ == '__main__':
    main()






