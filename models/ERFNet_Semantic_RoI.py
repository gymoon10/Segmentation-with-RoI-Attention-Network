
"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet_roi as network


class ERFNet_Semantic_RoI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        print('Creating ERFNet-RoI with {} classes'.format(num_classes))

        # shared model
        self.encoder = network.Encoder()  # Encoder(3+1)
        self.decoder = network.Decoder(self.num_classes)

        self.aux_decoder = network.AuxDecoder(self.num_classes)

    def forward(self, image, image_roi=None, feat_image=None):

        if image_roi == None:
            feat_enc = self.encoder(image)  # (N, 128, 64, 64)
            output, feat_dec = self.decoder.forward(feat_enc)  # (N, 3, 512, 512) / (N, 16, 512, 512)

            return output#, feat_dec

        elif image_roi != None and feat_image != None:
            feat_roi = self.encoder(image_roi)  # (N, 128, 64, 64)
            output_roi, feat_roi = self.decoder.forward(feat_roi)  # (N, 3, 512, 512) / (N, 16, 512, 512)

            output_fin = self.aux_decoder(feat_image, feat_roi)

            return output_roi, output_fin

