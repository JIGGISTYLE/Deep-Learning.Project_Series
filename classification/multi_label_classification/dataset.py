# dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class MoviePosterDataset(Dataset):      # pytorch base class
  def __init__(self,image_dir,csv_file,transform=None):
    self.image_dir=image_dir
    self.data=pd.read_csv(csv_file)  # read csv
    self.transform=transform
    self.label_cols=list(self.data.columns[2:])  # the 25 label column names, skips Id and Genre
  def __len__(self):
    return len(self.data)  # length of data

  def __getitem__(self,idx):
    row=self.data.iloc[idx]
    img_path=os.path.join(self.image_dir, f"{row.iloc[0]}.jpg") # image file name is the Id
    image=Image.open(img_path).convert("RGB")
    label=torch.tensor(row.iloc[2:].values.astype("float32"))  # 25 labels as 0/1 floats

    if self.transform:
      image=self.transform(image)

    return image,label
