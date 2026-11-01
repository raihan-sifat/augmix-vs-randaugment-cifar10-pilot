# CIFAR-10 Augmentation Pilot

This project investigates whether different image augmentation techniques can help an image classification model become more accurate and more robust. It uses **CIFAR-10**, a dataset of images from 10 categories, including cars, cats, dogs, and airplanes.

In simple terms, **image augmentation** creates different versions of training images to help a model recognize objects even when their appearance changes. This study compares three approaches:

- **Basic transformations (baseline):** Randomly crop and horizontally flip training images.
- **RandAugment:** Apply a randomly selected combination of image transformations.
- **AugMix:** Mix several augmented versions of an image with the original image.

All three approaches use the same **MobileNetV3-Small** model, adapted for CIFAR-10, to keep the comparison fair. The evaluation measures accuracy on normal images and on **corrupted images**, such as images affected by blur or noise, using CIFAR-10-C.

The main question is: **Which augmentation technique helps the model stay accurate when images are corrupted, and how much training time does it cost?**

This repository contains the experimental pilot described in [pilot_protocol.md](pilot_protocol.md), the long-term research plan, and the current paper draft. The sections below explain the environment, training, and evaluation steps.

## Environment

The project-local environment used for the current work is `.venv` with Python 3.14.5, torch 2.14.0+cu130, torchvision 0.29.0+cu130, and an NVIDIA GeForce GTX 1650 with 4 GB VRAM. Each final run saves exact software and hardware details in its config.

## Training

After the training duration is fixed in `pilot_protocol.md`, run each condition with the same epoch count, batch size, and worker count. The first run downloads CIFAR-10 if needed:

```powershell
.\.venv\Scripts\python.exe src\train.py --method baseline --seed 17 --epochs 50 --batch-size 128 --workers 2 --download --output-root results\final
.\.venv\Scripts\python.exe src\train.py --method randaugment --seed 17 --epochs 50 --batch-size 128 --workers 2 --output-root results\final
.\.venv\Scripts\python.exe src\train.py --method augmix --seed 17 --epochs 50 --batch-size 128 --workers 2 --output-root results\final
```

Repeat the three commands with `--seed 23`. The separate seed-99 one-epoch runtime pilots are not part of the accuracy comparison.

Each run creates a folder containing `config.json`, `epochs.csv`, `model.pt`, and `summary.json`.

## Evaluation

The clean CIFAR-10 test set is included when torchvision downloads CIFAR-10. For CIFAR-10-C, obtain the archive from [the official Zenodo record](https://zenodo.org/records/2535967), extract it under `data\CIFAR-10-C`, and record the downloaded file checksum and date. The evaluator uses the original 15 corruption types and ignores any extra arrays in expanded archive versions. Then run:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint-root results\final --data-root data --cifar-c-root data\CIFAR-10-C --output-root results\evaluation
```

Evaluation writes one CSV and JSON summary per checkpoint. Do not edit aggregate values by hand; generate paper tables from these outputs and preserve the raw files.

After evaluation, aggregate seed-level values and corruption-wise results with:

```powershell
.\.venv\Scripts\python.exe src\summarize.py --run-root results\final --evaluation-root results\evaluation --output-root results\analysis
```

This writes `condition_summary.csv`, `cifar10c_by_corruption.csv` (each type by severity), `cifar10c_by_type.csv` (averaged over severities and seeds), and `cifar10c_by_severity.csv` for the manuscript tables.

## Data and outputs

Downloaded datasets and model checkpoints are local research artifacts. Do not include them in a public source repository unless dataset and repository terms allow it. Share code, configurations, logs, result tables, and data acquisition instructions instead.
