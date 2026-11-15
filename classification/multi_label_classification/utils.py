#utils.py

import os
import torch
from torchvision import transforms

BASE_DIR=os.path.dirname(os.path.abspath(__file__))  # folder this file sits in
DATA_DIR=os.path.join(BASE_DIR, "data")              # built by prepare_data.py

TRAIN_DIR=os.path.join(DATA_DIR, "train")
VAL_DIR=os.path.join(DATA_DIR, "val")
TEST_DIR=os.path.join(DATA_DIR, "test")

TRAIN_CSV=os.path.join(DATA_DIR, "labels_train.csv")
VAL_CSV=os.path.join(DATA_DIR, "labels_val.csv")
TEST_CSV=os.path.join(DATA_DIR, "labels_test.csv")
LABELS_JSON=os.path.join(DATA_DIR, "labels.json")     # the 25 genre names
MODEL_PATH=os.path.join(BASE_DIR, "model.pth")        # saved weights, made by main.py

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES=25
BATCH_SIZE=32
EPOCHS=10
LR=1e-4
NUM_WORKERS=0  # keep 0 so it also runs on windows without problems

def get_train_transform():
  return transforms.Compose([
      transforms.Resize((224,224)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485,0.456,0.406],
          std=[0.229,0.224,0.225]
      )
  ])

def get_val_transform():
  return transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485,0.456,0.406],
          std=[0.229,0.224,0.225]
      )
  ])
