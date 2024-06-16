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

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader (student)
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

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

# Logger
logger = Logger(('train', 'val', 'val_iou_plant', 'val_iou_disease'), 'loss')


def calculate_iou(instance, pred, label):
    intersection = ((label == instance) & (pred == instance)).sum()
    union = ((label == instance) | (pred == instance)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        # for j in range(3):
        #     r = r | (bitget(c, 0) << 2-j)
        #     g = g | (bitget(c, 1) << 2-j)
        #     b = b | (bitget(c, 2) << 2-j)
        #     c = c >> 3

        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def decode_target(mask):
    """decode semantic mask to RGB image"""
    cmap = voc_cmap()
    return cmap[mask]



def main():
    # init
    start_epoch = 0
    best_iou_plant = 0
    best_iou_disease = 0
    best_iou_both = 0

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

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=1e-4)

    def lambda_(epoch):
        return pow((1 - ((epoch) / args['n_epochs'])), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_, )

    for epoch in range(start_epoch, args['n_epochs']):
        print('Starting epoch {}'.format(epoch))

        # validation
        loss_val_meter = AverageMeter()
        iou1_meter, iou2_meter = AverageMeter(), AverageMeter()

        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(val_dataset_it)):
                image = sample['image']  # (N, 3, 512, 512)
                label = sample['label_all'].squeeze(1)  # (N, 512, 512)

                save_image(image, os.path.join(args['save_dir'], '%d_rgb.png' % i))
                gt_both_output_seg = decode_target(label[0].type(torch.uint8).cpu().numpy()).astype(np.uint8)
                Image.fromarray(gt_both_output_seg).save(
                    os.path.join(args['save_dir'], '%d_label.png' % i))

                output = model(image)  # (N, 4, h, w)

                loss = criterion_val(output, label,
                                     iou=True, meter_plant=iou1_meter, meter_disease=iou2_meter)
                loss = loss.mean()
                loss_val_meter.update(loss.item())

                preds_iou = output.detach().max(dim=1)[1].cpu()
                # iou_1 = calculate_iou(1, label, preds_iou)
                iou_2 = calculate_iou(2, label, preds_iou)

                preds = output.detach().max(dim=1)[1].cpu().numpy()
                preds = decode_target(preds[0]).astype(np.uint8)
                Image.fromarray(preds).save(os.path.join(args['save_dir'], '%d_pred.png' % i))

        val_loss, val_iou_plant, val_iou_disease = loss_val_meter.avg, iou1_meter.avg, iou2_meter.avg
        print('===> val loss: {:.5f}, val iou-plant: {:.5f}, val iou-disease: {:.5f}'.format(val_loss, val_iou_plant,
                                                                                             val_iou_disease))


if __name__ == '__main__':
    main()






