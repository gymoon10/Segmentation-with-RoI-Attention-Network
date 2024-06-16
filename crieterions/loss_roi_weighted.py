"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w), c=num_classes
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss


class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss


criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
criterion_ce_weighted = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=0)
criterion_l1 = nn.L1Loss()


class CriterionCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, class_label, prediction_roi, class_label_roi, prediction_fin, pixel_weight,
                iou=False, meter_plant=None, meter_disease=None):
        '''prediction : seg model output (N, 3, 512, 512) *3 for bg/plant/disease
           instances_all : GT plant-mask (N, 512, 512)'''

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        loss = 0

        pixel_weight, _ = torch.max(pixel_weight, dim=1)
        pixel_weight = torch.pow(pixel_weight.detach(), 3).cuda()

        ce_loss2 = criterion_ce_weighted(prediction_roi, class_label_roi.type(torch.long).cuda(), pixel_weight)

        for b in range(0, batch_size):

            # cross entropy loss (original)
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss1 = criterion_ce(pred, gt_label)

            # cross entropy loss (roi)
            # pred = prediction_roi[b, :, :, :]
            # pred = pred.unsqueeze(0)
            # pred = pred.type(torch.float32).cuda()
            #
            # gt_label = class_label_roi[b].unsqueeze(0)  # (1, h, w)
            # gt_label = gt_label.type(torch.long).cuda()
            #
            # ce_loss2 = criterion_ce(pred, gt_label)

            # cross entropy loss (fin)
            pred_fin = prediction_fin[b, :, :, :]
            pred_fin = pred_fin.unsqueeze(0)
            pred_fin = pred_fin.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss3 = criterion_ce(pred_fin, gt_label)

            # total loss
            loss = loss + ce_loss1 + ce_loss3

            if iou:
                pred = pred_fin.detach().max(dim=1)[1]

                pred_plant = (pred == 1)
                pred_disease = (pred == 2)

                gt_plant = (class_label[b].unsqueeze(0) == 1)
                gt_disease = (class_label[b].unsqueeze(0) == 2)

                meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
                meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))

        loss = loss + ce_loss2

        return loss


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
