
"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet_original as erfnet


class ERFNet_Semantic_Original(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        print('Creating Original Semantic ERFNet with {} classes'.format(num_classes))
        # num_classes=[3, 1], 3 for instance-branch (1 sigma) & 1 for seed branch
        # num_classes=[4, 1], 4 for instance-branch (2 sigma) & 1 for seed branch

        # shared encoder
        self.encoder = erfnet.Encoder()  # Encoder(3+1)
        self.decoder = erfnet.Decoder(self.num_classes)

        dim_in = 16
        feat_dim = 16
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_):

        feat_enc = self.encoder(input_)  # (N, 128, 64, 64)
        output, feat_dec = self.decoder.forward(feat_enc)  # (N, 3, 512, 512) / (N, 16, 256, 256)

        return output#, feat_dec
