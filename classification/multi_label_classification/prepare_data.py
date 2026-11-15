# prepare_data.py (run only once)
# downloads nothing, it just splits Movies-Poster_Dataset into train/val/test

import os
import json
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder this file sits in
RAW_IMG_DIR = os.path.join(BASE_DIR, "Movies-Poster_Dataset", "Images")
RAW_LABELS = os.path.join(BASE_DIR, "Movies-Poster_Dataset", "train.csv")
OUT_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(OUT_DIR, exist_ok=True)
for s in ["train", "val", "test"]:
  os.makedirs(os.path.join(OUT_DIR, s), exist_ok=True)

df = pd.read_csv(RAW_LABELS)
df = df.drop_duplicates(subset="Id")  # 6 posters listed twice, keep one so they don't land in two splits

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)      # splitting 70% to train and 30% for temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)   # splitting temp into 15% val, 15% test

def copy_images(split_df, split):
  copied = 0
  for fname_id in split_df["Id"]:
    src = os.path.join(RAW_IMG_DIR, f"{fname_id}.jpg")
    dst = os.path.join(OUT_DIR, split, f"{fname_id}.jpg")
    if os.path.exists(src):
      shutil.copy(src, dst)
      copied += 1
    else:
      print(f"Warning: Image not found at {src}")
  print(f"{split}: copied {copied}/{len(split_df)} images")

copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

train_df.to_csv(os.path.join(OUT_DIR, "labels_train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "labels_val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "labels_test.csv"), index=False)

label_cols = list(df.columns[2:])  # the 25 genre names, skips Id and Genre
with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
  json.dump(label_cols, f, indent=2)  # so predict.py knows what each output number means

print("splits:", len(train_df), "train /", len(val_df), "val /", len(test_df), "test")
print("Dataset prepared successfully")
