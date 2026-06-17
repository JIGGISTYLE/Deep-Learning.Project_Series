# 🎬 Multi-Label Image Classification with ResNet18

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

This repository contains a modular PyTorch project for multi-label image classification. The current setup classifies movie posters into 25 different genres using a fine-tuned **ResNet18** model.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| 🗂️ `prepare_data.py` | Splits raw dataset (70/15/15) and organizes images into `train/`, `val/`, and `test/` directories. |
| 🗃️ `dataset.py` | Custom PyTorch `Dataset` (`MoviePosterDataset`) for loading images and multi-hot encoded labels. |
| 🛠️ `utils.py` | Defines data transformations (resizing, flipping, normalization) and global hyperparameters. |
| 🧠 `model.py` | Loads pre-trained ResNet18, freezes base layers, and adapts the fully connected layer. |
| ⚙️ `train.py` | Core training (`train_per_epoch`) and evaluation (`evaluate`) loops using `BCEWithLogitsLoss`. |
| 🚀 `main.py` | Entry point: initializes dataloaders, model, optimizer, and runs the training loop. |
| 📦 `requirments.txt` | Lists required Python dependencies. |

---

## 🛠️ Requirements

Install the necessary dependencies using:

```bash
pip install -r requirments.txt
