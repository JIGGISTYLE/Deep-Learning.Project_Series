# train.py
import torch
import torch.nn as nn
from tqdm.auto import tqdm # Import tqdm here

def train_per_epoch(model,loader,optimizer,criterion,device):
  model.train()

  for m in model.modules():      # backbone is frozen, so keep batchnorm frozen too
    if isinstance(m,nn.BatchNorm2d):
      m.eval()

  total_loss=0

  for image,label in tqdm(loader, desc="Training Batch", leave=False): # Added tqdm to the loader
    image=image.to(device)
    label=label.to(device)
    optimizer.zero_grad()

    output=model(image)
    loss=criterion(output,label)
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
  avg_loss=total_loss/len(loader)

  return avg_loss

def evaluate(model,loader,device):
  model.eval()
  correct=0
  total=0
  exact=0
  rows=0

  with torch.no_grad():
    for image,label in tqdm(loader, desc="Evaluation Batch", leave=False): # Added tqdm to the loader
      image=image.to(device)
      label=label.to(device)
      output=torch.sigmoid(model(image))
      preds=(output>0.5).float()
      correct+=(preds==label).sum().item()            # right guess per genre
      total+=label.numel()
      exact+=(preds==label).all(dim=1).sum().item()   # all 25 genres right for that poster
      rows+=label.size(0)

    return correct/total, exact/rows
