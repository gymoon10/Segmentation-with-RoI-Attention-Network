"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics.functional import pairwise_cosine_similarity


criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
criterion_l1 = nn.L1Loss()


class CriterionCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, class_label, iou=False, meter_plant=None, meter_disease=None):
        '''prediction : seg model output (N, 3, 512, 512) *3 for bg/plant/disease
           instances_all : GT plant-mask (N, 512, 512)'''

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        loss = 0

        for b in range(0, batch_size):

            # cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            # total loss
            loss = loss + ce_loss

            if iou:
                pred = pred.detach().max(dim=1)[1]

                pred_plant = (pred == 1)
                pred_disease = (pred == 2)

                gt_plant = (class_label[b].unsqueeze(0) == 1)
                gt_disease = (class_label[b].unsqueeze(0) == 2)

                meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
                meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))

        return loss


# class CriterionMatching(nn.Module):
#     def __init__(self, num_classes=3, ths_prob=0.6, ths_sim=0.9):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ths_prob = ths_prob
#         self.ths_sim = ths_sim
#
#     def forward(self, outputs, class_labels, embeddings, labels_down, outputs_aug, embeddings_aug, labels_down_aug,
#                 iou=False, meter_plant=None, meter_disease=None):
#
#         batch_size, height, width = outputs.size(
#             0), outputs.size(2), outputs.size(3)
#
#         loss = 0
#         loss_ce_total = 0
#         loss_matching_total = torch.tensor(0.0).cuda()
#
#         for b in range(0, batch_size):
#
#             # ----------------- cross entropy loss -----------------
#             pred = outputs[b, :, :, :]
#             pred = pred.unsqueeze(0)
#             pred = pred.type(torch.float32).cuda()
#
#             gt_label = class_labels[b].unsqueeze(0)  # (1, 512, 512)
#             gt_label = gt_label.type(torch.long).cuda()
#
#             loss_ce_total += criterion_ce(pred, gt_label)
#
#             if iou:
#                 pred = pred.detach().max(dim=1)[1]
#
#                 pred_plant = (pred == 1)
#                 pred_disease = (pred == 2)
#
#                 gt_plant = (class_labels[b].unsqueeze(0) == 1)
#                 gt_disease = (class_labels[b].unsqueeze(0) == 2)
#
#                 meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
#                 meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))
#
#             # ----------------- feature matching loss for self-consistency -----------------
#             emb = embeddings[b]
#             emb_size, emb_dim = (emb.shape[1], emb.shape[2]), emb.shape[0]
#             emb_flat = emb.view(emb_dim, -1)  # (16, 256*256)
#
#             emb_aug = embeddings_aug[b]
#             emb_flat_aug = emb_aug.view(emb_dim, -1)  # (16, 256*256)
#
#             mask = labels_down[b]  # (256, 256)
#             mask_aug = labels_down_aug[b]
#
#             for class_idx in range(2, self.num_classes):
#                 mask_class = (mask == class_idx).flatten()
#                 mask_class_aug = (mask_aug == class_idx).flatten()
#
#                 emb_mask = emb_flat[:, mask_class]  # (16, m pixels)
#                 emb_mask_aug = emb_flat_aug[:, mask_class_aug]  # (16, j pixels)
#
#                 # do not need to normalize
#                 similarities = pairwise_cosine_similarity(emb_mask.T, emb_mask_aug.T)  # (m pixels x j pixels)
#                 distances = 1 - similarities  # values between [0, 2] where 0 means same vectors
#                 print('matching-score :', distances.mean())
#
#                 loss_matching_total += distances.mean()
#
#         loss = (1 * loss_ce_total) + (2 * loss_matching_total)
#         #print('CE :', loss_ce_total)
#         #print('CE :', loss_ce_total.dtype)
#         #print('Matching :', loss_matching_total)
#
#         return loss, loss_ce_total, loss_matching_total


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
