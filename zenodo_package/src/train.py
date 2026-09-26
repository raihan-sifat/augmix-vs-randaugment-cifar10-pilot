"""Reproducible CIFAR-10 augmentation pilot training script."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
METHODS = ("baseline", "randaugment", "augmix")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_model() -> nn.Module:
    model = torchvision.models.mobilenet_v3_small(weights=None)
    # Keep more spatial information for 32x32 CIFAR images.
    model.features[0][0].stride = (1, 1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    return model


class AugmentedCIFAR(Dataset):
    def __init__(self, root: Path, method: str, download: bool):
        self.dataset = CIFAR10(root=str(root), train=True, download=download)
        self.method = method
        self.geometry = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.randaugment = transforms.RandAugment(num_ops=2, magnitude=9)
        self.augmix = transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.geometry(image)
        if self.method == "baseline":
            return self.normalize(image), label
        if self.method == "randaugment":
            return self.normalize(self.randaugment(image)), label
        # Full AugMix consistency training: clean view plus two independent
        # AugMix views; JSD loss is applied in the training loop.
        return (self.normalize(image),
                self.normalize(self.augmix(image)),
                self.normalize(self.augmix(image))), label


def jensen_shannon(logits_a, logits_b, logits_c):
    log_probs = [F.log_softmax(x, dim=1) for x in (logits_a, logits_b, logits_c)]
    probs = [x.exp() for x in log_probs]
    mean_prob = torch.stack(probs, dim=0).mean(dim=0)
    log_mean = mean_prob.clamp_min(1e-7).log()
    return sum(F.kl_div(log_mean, p, reduction="batchmean") for p in probs) / 3.0


def system_manifest(device: torch.device):
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this planned experiment; no CPU fallback was run.")
    run_name = f"{args.method}_seed{args.seed}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ds = AugmentedCIFAR(args.data_root, args.method, args.download)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker, generator=generator,
    )

    model = make_model().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config = {
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "optimizer": "SGD(momentum=0.9, weight_decay=0.0005)",
        "initial_lr": args.lr,
        "scheduler": "CosineAnnealingLR",
        "model": "torchvision MobileNetV3-Small; first convolution stride changed to 1; classifier output changed to 10",
        "parameters": parameter_count,
        "augmentation": {
            "shared": "RandomCrop(32, padding=4), RandomHorizontalFlip",
            "randaugment": "RandAugment(num_ops=2, magnitude=9)",
            "augmix": "AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0), two independent augmented views plus clean view, JSD weight=12",
        },
        "normalization_mean": MEAN,
        "normalization_std": STD,
        "system": system_manifest(device),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    epoch_log = run_dir / "epochs.csv"
    with epoch_log.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "epoch_seconds",
                                     "source_examples_per_second", "learning_rate_next_epoch"]).writeheader()
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in loader:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.method == "augmix":
                views = [x.to(device, non_blocking=True) for x in images]
                batch_n = labels.shape[0]
                logits = model(torch.cat(views, dim=0))
                clean_logits, aug1_logits, aug2_logits = logits.split(batch_n, dim=0)
                loss = (F.cross_entropy(clean_logits, labels)
                        + F.cross_entropy(aug1_logits, labels)
                        + F.cross_entropy(aug2_logits, labels)) / 3.0
                loss = loss + 12.0 * jensen_shannon(clean_logits, aug1_logits, aug2_logits)
                predicted = clean_logits.argmax(dim=1)
            else:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                predicted = logits.argmax(dim=1)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (predicted == labels).sum().item()
            seen += n
        scheduler.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        lr_now = scheduler.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "epoch_seconds": elapsed,
            "source_examples_per_second": seen / elapsed,
            "learning_rate_next_epoch": lr_now,
        }
        rows.append(row)
        with epoch_log.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
            f.flush()
        print(f"{run_name} epoch {epoch}/{args.epochs}: loss={row['train_loss']:.4f} "
              f"train_acc={row['train_accuracy']:.4f} seconds={elapsed:.1f} "
              f"source_img_s={row['source_examples_per_second']:.1f}", flush=True)

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_start
    torch.save(model.cpu().state_dict(), run_dir / "model.pt")
    summary = {
        "run": run_name,
        "training_seconds_including_epoch_data_loading": training_seconds,
        "training_minutes": training_seconds / 60.0,
        "source_examples_per_second_overall": len(ds) * args.epochs / training_seconds,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "final_epoch_train_accuracy": rows[-1]["train_accuracy"],
        "parameter_count": parameter_count,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
