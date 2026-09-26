# Reproducibility files

This archive accompanies **Training-Time Image Augmentation on MobileNetV3-Small: A Controlled CIFAR-10 Pilot**. It contains the scripts and run artifacts used for the reported two-seed comparison of crop-and-flip, RandAugment, and AugMix.

## Contents

- `src/`: training, evaluation, and aggregation scripts.
- `protocol/`: frozen pilot protocol.
- `results/final/`: six checkpoints, exact run configurations, epoch logs, and summaries.
- `results/evaluation/`: per-checkpoint clean CIFAR-10 and CIFAR-10-C evaluations in CSV and JSON formats.
- `results/analysis/`: aggregate condition, corruption, and severity tables used in the paper.

## Data

The CIFAR-10 and CIFAR-10-C datasets are not redistributed in this archive. CIFAR-10-C can be obtained from the [official Zenodo record](https://zenodo.org/records/2535967). The experiment used the original 15 corruption types at severities 1–5 from the `CIFAR-10-C.tar` archive, MD5 `56bf5dcef84df0e2308c6dcbcbbd8499`. Download and verify the archive before extracting it under `data/CIFAR-10-C`. The training script can download CIFAR-10 through torchvision.

## Reproduction

Use Python with PyTorch, torchvision, and NumPy. The recorded experiment environment was Windows 11, Python 3.14.5, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, NumPy 2.5.2, CUDA, and an NVIDIA GeForce GTX 1650 with 4 GB VRAM. Each run's `config.json` records its own software and hardware manifest.

From the archive root, run one training condition as follows; repeat with seeds 17 and 23 for all three methods:

```powershell
python src/train.py --method baseline --seed 17 --epochs 50 --batch-size 128 --workers 2 --download --output-root results/final
python src/train.py --method randaugment --seed 17 --epochs 50 --batch-size 128 --workers 2 --output-root results/final
python src/train.py --method augmix --seed 17 --epochs 50 --batch-size 128 --workers 2 --output-root results/final
```

For evaluation and aggregation, use:

```powershell
python src/evaluate.py --checkpoint-root results/final --data-root data --cifar-c-root data/CIFAR-10-C --output-root results/evaluation
python src/summarize.py --run-root results/final --evaluation-root results/evaluation --output-root results/analysis
```

The published results are descriptive results from two seeds on one model and one dataset. They are not inferential estimates of general performance. Timing observations are specific to the recorded system and showed substantial variation.

## Rights and license

The author-selected license for this archive is Creative Commons Attribution 4.0 International (CC BY 4.0). The datasets are excluded and remain subject to their source terms. The CC BY 4.0 attribution requirement applies to the paper and reproducibility materials in this record.
