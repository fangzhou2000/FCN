import os.path

import torch
import torchvision

def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained)
    return model
