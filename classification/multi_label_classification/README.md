# Multi-Label Image Classification with ResNet18

This repository contains a PyTorch-based modular project for multi-label image classification. The current setup is configured to classify movie posters into 25 different genres/categories using a pre-trained ResNet18 model.

## Project Structure

* **`prepare_data.py`**: A script to organize the raw downloaded dataset. It reads the labels CSV, splits the data into `train`, `val`, and `test` sets (70/15/15), and copies the respective images into a structured `data/` directory.
* **`dataset.py`**: Contains the custom PyTorch `Dataset` class (`MoviePosterDataset`) to load images and their corresponding multi-hot encoded labels from the CSV files.
* **`utils.py`**: Defines data transformations (resizing, random horizontal flips, and ImageNet normalization) and stores global hyperparameters like `BATCH_SIZE`, `EPOCHS`, `LR`, and `NUM_CLASSES`.
* **`model.py`**: Loads a pre-trained ResNet18 model from `torchvision`, freezes the base layers, and replaces the fully connected layer to output the specified number of classes for fine-tuning.
* **`train.py`**: Contains the core training (`train_per_epoch`) and evaluation (`evaluate`) loops. It calculates Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`) for multi-label targets and computes accuracy based on a sigmoid threshold.
* **`main.py`**: The entry point of the application. It initializes the dataloaders, model, criterion, and optimizer, and runs the training loop over the specified number of epochs.
* **`requirments.txt`**: Lists the Python dependencies required to run the project.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirments.txt
