<div align="center">

# 🎬 Movie Poster Genre Classification

### 🖼️ Predict multiple movie genres from a single poster using PyTorch & ResNet-18

**Multi-Label Classification · Computer Vision · Transfer Learning**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ResNet-18](https://img.shields.io/badge/Backbone-ResNet--18-6C5CE7?style=for-the-badge)
![CUDA](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

**📚 7,254 Posters · 🎭 25 Genres · 🧠 ~138K Trainable Parameters · 🏋️ 10 Epochs**

</div>

---

## 🌟 Overview

**Movie Poster Genre Classification** is a **multi-label image classifier**. Give it a movie poster, and it predicts which genres the movie belongs to.

A poster can belong to more than one genre — such as **Comedy + Drama** — so this is **multi-label** classification, not normal single-label classification. The model therefore has **25 outputs**, with a **sigmoid applied independently to each output**, rather than a softmax across all genres.

### 🎞️ Real posters from the dataset

<div align="center">
<table>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/laxmimerit/Movies-Poster_Dataset/master/Images/tt0086425.jpg" width="170" alt="Terms of Endearment poster"/></td>
<td align="center"><img src="https://raw.githubusercontent.com/laxmimerit/Movies-Poster_Dataset/master/Images/tt0086567.jpg" width="170" alt="WarGames poster"/></td>
<td align="center"><img src="https://raw.githubusercontent.com/laxmimerit/Movies-Poster_Dataset/master/Images/tt0086034.jpg" width="170" alt="Octopussy poster"/></td>
<td align="center"><img src="https://raw.githubusercontent.com/laxmimerit/Movies-Poster_Dataset/master/Images/tt0085549.jpg" width="170" alt="Flashdance poster"/></td>
</tr>
<tr>
<td align="center">🎭 Comedy · Drama</td>
<td align="center">🛸 Sci-Fi · Thriller</td>
<td align="center">💥 Action · Adventure · Thriller</td>
<td align="center">🎵 Drama · Romance · Music</td>
</tr>
</table>
<sub>Source: <a href="https://github.com/laxmimerit/Movies-Poster_Dataset">Movies-Poster_Dataset</a>. Captions are the dataset's ground-truth genre labels, not model predictions.</sub>
</div>

---

## ⚙️ How It Works

The classifier uses **transfer learning**: it keeps a pretrained ResNet-18 backbone frozen and trains a small classification head.

```text
                   🖼️ Movie Poster
                          │
                          ▼
              🧠 ResNet-18 (ImageNet)
                 🔒 Frozen Backbone
                          │
                          ▼
                    512 Features
                          │
                          ▼
                  Linear: 512 → 256
                          │
                          ▼
                         ReLU
                          │
                          ▼
                    Dropout (0.3)
                          │
                          ▼
                   Linear: 256 → 25
                          │
                          ▼
                25 Independent Logits
                          │
                          ▼
                 Sigmoid + 0.5 Cutoff
                          │
                          ▼
              🎭 Predicted Movie Genres
```

1. 📥 **Load the backbone:** Start with a pretrained **ResNet-18** trained on ImageNet and freeze the entire backbone.
2. 🧩 **Replace its final layer:** Use the head `512 → 256 → ReLU → Dropout(0.3) → 25`.
3. 🏋️ **Train the head only:** Optimize using `BCEWithLogitsLoss`.
4. 🎯 **Predict genres:** At inference, apply a sigmoid to each of the 25 outputs and select genres whose scores exceed **0.5**.

> 💡 **Why freeze the backbone?** Only about **138K parameters** are trained out of **11.3 million**. This keeps training fast and helps limit overfitting on a small dataset.

---

## 📚 Dataset

**Dataset:** [Movies-Poster_Dataset](https://github.com/laxmimerit/Movies-Poster_Dataset)

The dataset contains **7,254 movie posters** and **25 genre columns**.

### 📥 Download

```bash
git clone https://github.com/laxmimerit/Movies-Poster_Dataset.git
```

The downloaded dataset includes `train.csv` and an `Images/` directory.

### 🧾 CSV Structure

```csv
Id,Genre,Action,Adventure,Animation,Biography,Comedy,...
tt0086425,"['Comedy', 'Drama']",0,0,0,0,1,0,0,1,0,...
```

| Column | Meaning |
|:--|:--|
| `Id` | Movie poster image filename / ID |
| `Genre` | Human-readable genre list |
| Last 25 columns | Binary `0/1` labels for each genre |

### 🎭 All 25 Genres

`Action` · `Adventure` · `Animation` · `Biography` · `Comedy` · `Crime` · `Documentary` · `Drama` · `Family` · `Fantasy` · `History` · `Horror` · `Music` · `Musical` · `Mystery` · `N/A` · `News` · `Reality-TV` · `Romance` · `Sci-Fi` · `Short` · `Sport` · `Thriller` · `War` · `Western`

---

## 📂 Project Files

| File | Purpose |
|:--|:--|
| `prepare_data.py` | 🗂️ Runs **once**: splits the CSV into train/validation/test sets, copies images into `data/`, and writes label CSV files and `labels.json`. |
| `dataset.py` | 🖼️ `MoviePosterDataset`: opens an image and its 25 labels for the DataLoader. |
| `model.py` | 🧠 `get_resnet18()`: frozen ImageNet ResNet-18 with a new 25-output head. |
| `train.py` | 🏋️ `train_per_epoch()` handles one epoch; `evaluate()` computes accuracy and exact-match. |
| `utils.py` | ⚙️ Stores settings (batch size, epochs, learning rate) and file paths. |
| `main.py` | 🚀 Entry point: builds loaders, trains, evaluates on the test set, and saves `model.pth`. |
| `requirments.txt` | 📦 Required Python packages. |

**Important:** Run `main.py` **only at the end**, after the other components are ready.

---

## 🛠️ Setup

### 1️⃣ Install Dependencies

```bash
pip install -r requirments.txt
```

### 2️⃣ Set Up NVIDIA GPU Support

This project was trained on an **RTX 3050 Laptop GPU (4 GB)**. To use NVIDIA CUDA, install a CUDA-enabled build of PyTorch rather than relying on a potentially CPU-only installation.

```bash
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3️⃣ Verify CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

✅ Nothing else in the code needs to change: `utils.py` already selects `cuda` automatically when it is available.

---

## 🚀 How to Run

### Step 1 — Prepare the Data (once only)

```bash
python prepare_data.py
```

This creates the following structure:

```text
data/
├── train/            5069 images + labels_train.csv
├── val/              1086 images + labels_val.csv
├── test/             1087 images + labels_test.csv
└── labels.json       the 25 genre names
```

📊 **Split:** 70% training / 15% validation / 15% testing.

🧹 Duplicate posters are removed **before** splitting, so the same poster never appears in two splits.

### Step 2 — Train & Evaluate

```bash
python main.py
```

This trains the classifier for **10 epochs**, evaluates it on the test set, and saves the learned weights in **`model.pth`**.

---

## 📊 Results

### 🏋️ Final Training Log

```text
epoch 10  train_loss=0.2115  val_acc=0.9193  val_exact=0.0967

final test acc=0.9187   exact=0.0828
```

| Metric | Value |
|:--|--:|
| 🔻 Training loss, epoch 10 | `0.2115` |
| ✅ Validation per-label accuracy | `0.9193` (91.93%) |
| 🎯 Validation exact-match | `0.0967` (9.67%) |
| ✅ Final test per-label accuracy | `0.9187` (91.87%) |
| 🎯 Final test exact-match | `0.0828` (8.28%) |

### ⚠️ Understanding the Metrics

**Read the numbers carefully.** A poster has, on average, only **2.34 genres out of 25**. A model predicting **no genres at all** already gets **0.9066** per-label accuracy on validation and **0.9070** on test.

That means the model's **0.9187 test accuracy** is only a little higher than the all-zero baseline. **Per-label accuracy alone is therefore misleading.**

**Exact-match** is the stricter metric: a prediction counts as correct only when **all 25 genre labels** are correct for a poster. Here, **8.3% of test posters** were fully correct, compared with **0% exact-match** for the all-zero baseline.

> 🎯 **Takeaway:** The model has learned to make some meaningful multi-genre predictions, but its test accuracy should not be mistaken for 91.87% of posters being classified perfectly.

---

## 🔍 Things to Know

### ⚖️ 1. Severe Class Imbalance

The genre frequencies vary substantially:

| Genre | Examples |
|:--|--:|
| 🎭 Drama | **3,619** |
| 📺 Reality-TV | **2** |
| 📰 News | **21** |
| 🎞️ Short | **46** |

With a fixed **0.5 threshold**, very rare genres will almost never be predicted.

### 🔒 2. Frozen Backbone

The entire ImageNet backbone remains frozen. Training loss stops improving after around **3 epochs** because those pretrained visual features do not change.

**Potential improvement:** Unfreeze ResNet-18's `layer4` for fine-tuning.

### 🧊 3. BatchNorm Evaluation Mode

`train.py` keeps **BatchNorm in evaluation mode** during training. Otherwise, the frozen backbone's running statistics could keep changing and cause feature drift.

### 🎚️ 4. Genre-Specific Thresholds

A single **0.5 threshold** is a rough choice for an imbalanced multi-label problem.

**Potential improvement:** Choose a separate threshold for each genre to obtain noticeably better predictions.

---

<div align="center">

### 🎬 From Movie Posters to Multiple Genre Predictions

**PyTorch · ResNet-18 · Transfer Learning · Multi-Label Image Classification**

📚 [Original Dataset](https://github.com/laxmimerit/Movies-Poster_Dataset)

</div>
