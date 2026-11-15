# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import MoviePosterDataset
from model import get_resnet18
from train import train_per_epoch, evaluate
from utils import *

def main():
  if DEVICE == 'cpu':   # added a warning if running on CPU
    print("Warning: No GPU found. Training will be significantly slower on CPU.")

  train_dataset=MoviePosterDataset(TRAIN_DIR,TRAIN_CSV,get_train_transform())
  val_dataset=MoviePosterDataset(VAL_DIR,VAL_CSV,get_val_transform())
  test_dataset=MoviePosterDataset(TEST_DIR,TEST_CSV,get_val_transform())

  train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
  val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
  test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

  model=get_resnet18(NUM_CLASSES).to(DEVICE)
  criterion=nn.BCEWithLogitsLoss()
  optimizer=optim.Adam(model.fc.parameters(),lr=LR)  # only the new head trains

  for epoch in tqdm(range(EPOCHS), desc="Training Epochs"): # Added tqdm for epoch progress
    train_loss=train_per_epoch(model,train_loader,optimizer,criterion,DEVICE)
    val_acc,val_exact=evaluate(model,val_loader,DEVICE)

    print(f"epoch {epoch+1}/{EPOCHS}",
          f"train_loss={train_loss:.4f}",
          f"val_acc={val_acc:.4f}",
          f"val_exact={val_exact:.4f}")

  test_acc,test_exact=evaluate(model,test_loader,DEVICE)
  print(f"final test acc={test_acc:.4f} exact={test_exact:.4f}")

  torch.save(model.state_dict(),MODEL_PATH)  # so predict.py can load it later
  print("model saved to",MODEL_PATH)

if __name__=="__main__":
  main()
