from typing import List
import torch.nn as nn
import logging
import numpy as np
import os
import torchvision
import torch

from DLIP.models.zoo.building_blocks.double_conv import DoubleConv
from DLIP.models.zoo.building_blocks.down_sample import Down
from DLIP.models.zoo.encoder.basic_encoder import BasicEncoder

class ResNetEncoder(BasicEncoder):
    def __init__(
        self,
        input_channels: int,
        encoder_type = 'resnet50',
        pretraining_weights = 'imagenet',
        classification_output = False,
    ):
        super().__init__(input_channels,classification_output)
        load_imagenet = False
        if pretraining_weights == 'imagenet':
            load_imagenet = True
        encoder_class = None
        weights = None
        load_weights_directly_into_encoder = False
        encoder_type = encoder_type.lower()
        # Its a resnet encoder!
        if encoder_type == 'resnet18':
            encoder_class = torchvision.models.resnet18
        if encoder_type == 'resnet34':
            encoder_class = torchvision.models.resnet34
        if encoder_type == 'resnet50':
            encoder_class = torchvision.models.resnet50
        if encoder_type == 'resnet101':
            encoder_class = torchvision.models.resnet101
        if encoder_type == 'resnet152':
            encoder_class = torchvision.models.resnet152
        if encoder_class != None:
            # we determined resnet encoder type. Now we put it into the backbone module list
            if load_imagenet:
                logging.info('Loading imagenet weights')
            encoder = encoder_class(pretrained=load_imagenet)
            encoder.fc = nn.Linear(in_features=encoder.fc.in_features,out_features=3,bias=True)
            self.backbone.append(encoder)
