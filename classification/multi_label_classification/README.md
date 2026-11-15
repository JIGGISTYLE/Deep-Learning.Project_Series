# Movie Poster Genre Classification

A multi-label image classifier. Give it a movie poster, it predicts which genres the movie belongs to.

A poster can have more than one genre (for example *Comedy* + *Drama*), so this is **multi-label** classification, not normal single-label classification. That is why the model has 25 outputs with a sigmoid on each one instead of a softmax.

## How it works

1. Take a pretrained **ResNet-18** (trained on ImageNet), freeze the whole thing.
2. Cut off the last layer and put a small new head on top: `512 -> 256 -> ReLU -> Dropout(0.3) -> 25`.
3. Train only that head with `BCEWithLogitsLoss`.
4. At prediction time, run each of the 25 outputs through a sigmoid and mark it as a genre if the value is above 0.5.

Because the backbone is frozen, only about 138k parameters are trained out of 11.3 million. This keeps training fast and stops the model from overfitting on a small dataset.

## Dataset

**Movies-Poster_Dataset** - 7,254 movie posters with 25 genre columns.

```
git clone https://github.com/laxmimerit/Movies-Poster_Dataset.git
```

It has `train.csv` and an `Images/` folder. The CSV looks like this:

```
Id,Genre,Action,Adventure,Animation,Biography,Comedy,...
tt0086425,"['Comedy', 'Drama']",0,0,0,0,1,0,0,1,0,...
```

Column `Id` is the image file name, `Genre` is the readable text, and the last 25 columns are the 0/1 labels.

The 25 genres are: Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Musical, Mystery, N/A, News, Reality-TV, Romance, Sci-Fi, Short, Sport, Thriller, War, Western.

## Project files

| File | What it does |
| --- | --- |
| `prepare_data.py` | Runs **once**. Splits the CSV into train/val/test, copies images into `data/`, writes the label CSVs and `labels.json`. |
| `dataset.py` | `MoviePosterDataset` - opens one image + its 25 labels for the DataLoader. |
| `model.py` | `get_resnet18()` - frozen ImageNet ResNet-18 with a new 25-output head. |
| `train.py` | `train_per_epoch()` for one epoch, `evaluate()` for accuracy and exact-match. |
| `utils.py` | All settings (batch size, epochs, learning rate) and all file paths. |
| `main.py` | The entry point. Builds the loaders, trains, evaluates on test, saves `model.pth`. |
| `eval.py` | Loads `model.pth` and prints the full report: per-genre precision/recall/F1/AP, macro and micro scores, Hamming loss, and a threshold sweep. |
| `requirments.txt` | Python packages needed. |

`main.py` is run only at the very end, when everything else is already in place.

## Setup

```
pip install -r requirments.txt
```

**To train on an NVIDIA GPU** (this project was trained on an RTX 3050 Laptop, 4 GB), you need the CUDA build of PyTorch. The normal `pip install torch` gives you a CPU-only build that cannot see the GPU at all.

```
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Check it worked:

```
python -c "import torch; print(torch.cuda.is_available())"
```

It should print `True`. Nothing else in the code needs to change - `utils.py` already picks `cuda` automatically when it is available.

## How to run

**Step 1 - prepare the data (once only):**

```
python prepare_data.py
```

This creates:

```
data/
  train/   5069 images + labels_train.csv
  val/     1086 images + labels_val.csv
  test/    1087 images + labels_test.csv
  labels.json    the 25 genre names
```

The split is 70/15/15. Duplicate posters are dropped before splitting, so the same image never ends up in two splits.

**Step 2 - train:**

```
python main.py
```

This trains for 10 epochs, evaluates on the test set, and saves the weights to `model.pth`.

**Step 3 - check the model properly:**

```
python eval.py
```

This loads `model.pth` and prints the detailed report shown below. It does not train anything.

## Results

```
epoch 10  train_loss=0.2115  val_acc=0.9193  val_exact=0.0967

final test acc=0.9187   exact=0.0828
```

**Read these two numbers carefully.** The average poster has only 2.34 genres out of 25, so a model that predicts "no genres at all" already scores **0.9066** per-label accuracy on validation (`0.9070` on test). The 0.9187 above is barely above that, so per-label accuracy is not a useful number on its own.

`exact` is the honest metric - it means all 25 outputs were correct for that poster. 8.3% of test posters were fully correct. A model that predicts all zeros scores 0% exact, so this is the number that shows the model actually learned something.

## Things to know

- **Class imbalance is severe.** Drama has 3,619 examples but Reality-TV has 2, News has 21, and Short has 46. Those rare genres will basically never be predicted with a flat 0.5 threshold.
- **The backbone is fully frozen.** Training loss stops improving after about 3 epochs because the ImageNet features do not change. Unfreezing `layer4` is the simplest way to get a better score.
- **BatchNorm stays in eval mode** during training (`train.py`), otherwise the frozen backbone's running statistics keep changing and the "frozen" features drift.
- **A flat 0.5 threshold is rough** for imbalanced multi-label problems. Picking a threshold per genre gives noticeably better predictions.
