# Homework 2 - Image Classification (Transfer Learning)

Oxford-IIIT Pet (37 breeds) with a pre-trained MobileNetV3-Small (torchvision, `MobileNet_V3_Small_Weights.DEFAULT`, ImageNet-1K).
All experiments are in one file: `main.py`.

## Files
- `main.py` - data loading, training, feature extraction, evaluation, figures and tables
- `results/` - final run: metrics, plots, checkpoints, split file, log
- `results_v1/` - first run

## Environment
- Python 3.13.5
- torch 2.14.1 (CUDA 12.6 build), torchvision 0.29.1, scikit-learn 1.9.1, numpy 2.3.0, matplotlib 3.10.9, Pillow 10.4.0

```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install scikit-learn matplotlib numpy pillow
```

## Dataset
Oxford-IIIT Pet Dataset, Parkhi, Vedaldi, Zisserman, Jawahar, CVPR 2012.
Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ - license: Creative Commons Attribution-ShareAlike 4.0.
The dataset is **not** included in this repo. `main.py` downloads it automatically into `data/` (torchvision, ~800 MB).

Note (Windows): extracting `annotations.tar.gz` can fail on macOS `._*` files. If that happens, run
`tar -xzf annotations.tar.gz --exclude="._*"` inside `data/oxford-iiit-pet/` and run again.

Splits: official `trainval` (3680 imgs) is split 80/20 with stratification by breed (train 2944 / val 736);
official `test` (3669 imgs) is only used for the final evaluation. Indices are saved in `results/split_files.json`,
per-breed counts in `results/class_counts.csv`.

## How to reproduce
```
python main.py                                           # final run: all three approaches -> results/
python main.py --epochs 10 --no_cosine --out results_v1  # first run (10 epochs, constant LR)
python main.py --epochs 1 --out results_test             # quick test
```
This regenerates everything in `results/`: `results.json`, `results_table.md`, confusion matrices (`cm_*.png`),
prediction examples (`examples_*.png`), loss/accuracy curves (`curves.png`), comparison bar charts (`comparison.png`), best checkpoints (`full_best.pth`, `head_best.pth`),
and the split file.

## Setup of the three approaches
| | Full fine-tuning | Head training | Feature extraction + sklearn |
|---|---|---|---|
| Trainable | all layers | `classifier[0]` Linear(576,1024) and new `classifier[3]` Linear(1024,37); `features` frozen (BatchNorm kept in eval mode) | nothing in the CNN; sklearn classifier on 576-d pooled features |
| Loss / optimizer | CrossEntropy / Adam + cosine LR | CrossEntropy / Adam + cosine LR | sklearn classifiers |
| Learning rate | 1e-4 | 1e-3 | - |
| Batch size / epochs | 32 / 20 | 32 / 20 | - |
| Model selection | epoch with best validation accuracy | same | hyperparameter (C or n_estimators) with best validation accuracy; StandardScaler fit on train only |

Classifiers for approach 3: Logistic Regression, Linear SVM, Random Forest.

Preprocessing: resize 256, center crop 224, ImageNet mean/std (from the weights' own transforms).
Train-only augmentation: random crop 224, horizontal flip, color jitter. No class balancing (classes are almost balanced:
74-80 train images per breed, 88-100 test images per breed).

## Reproducibility
Seed 42 for Python, NumPy, PyTorch (and the train/val split). `cudnn.deterministic=True`, `cudnn.benchmark=False`,
`torch.use_deterministic_algorithms(True, warn_only=True)`. Some CUDA operations may still be non-deterministic
(PyTorch only warns for them), so tiny differences between runs are possible. Timing depends on the machine.
Inside one run the test set is used only once per selected model, never for checkpoint or hyperparameter selection.

## Two runs (hyperparameter tuning on validation)
- v1 (`results_v1/`): 10 epochs, constant learning rate. Test acc: full 0.830, head 0.859.
- v2 (`results/`, final): 20 epochs + cosine LR schedule, chosen because v1 was still improving at epoch 10.
  Test acc: full 0.845, head 0.869. The sklearn results are identical in both runs.
Checkpoints and hyperparameters were always selected on validation accuracy. However, the v2 run was started
after the v1 test results had been seen, so the test set was evaluated twice for the neural approaches and the
v2 test scores may be slightly optimistic. Both runs are reported.

## Selected checkpoints and hyperparameters (final run)
| Approach | Selected |
|---|---|
| Full fine-tuning | epoch 17 of 20 (`results/full_best.pth`), lr 1e-4 |
| Head training | epoch 16 of 20 (`results/head_best.pth`), lr 1e-3 |
| Logistic regression | C=0.01 (smallest value tried) |
| Linear SVM | C=0.01 |
| Random forest | n_estimators=300 (largest value tried) |
