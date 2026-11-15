#model.py

import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes):

  model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # imagenet weights, so we dont train from scratch

  for param in model.parameters():
    param.requires_grad=False  # freeze the whole backbone, only the new head learns

  model.fc=nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, num_classes)
  )  # new head, 512 -> 256 -> 25, one output per genre
  return model
