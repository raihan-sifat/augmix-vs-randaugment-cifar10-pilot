"""Evaluate saved pilot checkpoints on clean CIFAR-10 and CIFAR-10-C."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from train import MEAN, STD, make_model

CANONICAL_CORRUPTIONS = {
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur",
}


def evaluate_batches(model, batches, labels, device, batch_size):
    correct, count = 0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            images = batches(start, end)
            images = images.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
            mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
            std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
            images = (images - mean) / std
            targets = torch.as_tensor(np.array(labels[start:end]), device=device, dtype=torch.long)
            correct += (model(images).argmax(1) == targets).sum().item()
            count += len(targets)
    return correct / count


def evaluate_clean(model, data_root, device, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = CIFAR10(root=str(data_root), train=False, download=False, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, count = 0, 0
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            correct += (model(images).argmax(1) == targets).sum().item()
            count += len(targets)
    return correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cifar-c-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this planned evaluation.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    cifar_c_root = args.cifar_c_root
    if cifar_c_root is not None and (cifar_c_root / "CIFAR-10-C").is_dir():
        cifar_c_root = cifar_c_root / "CIFAR-10-C"
    corruption_files = []
    if cifar_c_root is not None and cifar_c_root.is_dir():
        present = {p.stem: p for p in cifar_c_root.glob("*.npy") if p.name != "labels.npy"}
        missing = CANONICAL_CORRUPTIONS - set(present)
        if missing:
            raise FileNotFoundError(f"CIFAR-10-C archive is missing canonical corruption files: {sorted(missing)}")
        # Some CIFAR-10-C archive versions include four additional corruptions.
        # The primary aggregate uses the original 15-corruption benchmark set.
        corruption_files = [present[name] for name in sorted(CANONICAL_CORRUPTIONS)]
    labels_c = None
    if corruption_files:
        labels_path = cifar_c_root / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10-C labels file: {labels_path}")
        labels_c = np.load(labels_path, mmap_mode="r")

    checkpoints = sorted(args.checkpoint_root.glob("*seed*/model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found beneath {args.checkpoint_root}")

    for checkpoint in checkpoints:
        run_dir = checkpoint.parent
        model = make_model()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device)
        clean_accuracy = evaluate_clean(model, args.data_root, device, args.batch_size)
        rows = [{"dataset": "CIFAR-10", "corruption": "clean", "severity": 0,
                 "accuracy": clean_accuracy, "n": 10000}]
        print(f"{run_dir.name} CIFAR-10 clean accuracy={clean_accuracy:.4f}", flush=True)

        if corruption_files:
            corruption_rows = []
            for image_path in corruption_files:
                images = np.load(image_path, mmap_mode="r")
                if len(images) != len(labels_c) or len(images) % 5 != 0:
                    raise ValueError(f"Unexpected CIFAR-10-C array shape in {image_path}: {images.shape}")
                per_severity = len(images) // 5
                for severity in range(1, 6):
                    start = (severity - 1) * per_severity
                    end = severity * per_severity
                    accuracy = evaluate_batches(
                        model,
                        lambda a, b, arr=images, offset=start: torch.from_numpy(np.array(arr[offset + a:offset + b])),
                        labels_c[start:end], device, args.batch_size,
                    )
                    row = {"dataset": "CIFAR-10-C", "corruption": image_path.stem,
                           "severity": severity, "accuracy": accuracy, "n": per_severity}
                    corruption_rows.append(row)
                    print(f"{run_dir.name} {image_path.stem} severity={severity} accuracy={accuracy:.4f}", flush=True)
            all_corruptions_mean = float(np.mean([r["accuracy"] for r in corruption_rows]))
            by_severity = {
                f"severity_{level}_mean": float(np.mean([r["accuracy"] for r in corruption_rows if r["severity"] == level]))
                for level in range(1, 6)
            }
            rows.extend(corruption_rows)
            rows.append({"dataset": "CIFAR-10-C", "corruption": "mean_over_corruptions_and_severities",
                         "severity": "all", "accuracy": all_corruptions_mean,
                         "n": len(corruption_rows) * per_severity})
            summary = {"run": run_dir.name, "cifar10_clean_accuracy": clean_accuracy,
                       "cifar10c_mean_accuracy": all_corruptions_mean,
                       "cifar10c_corruption_types": len(corruption_files), **by_severity}
        else:
            summary = {"run": run_dir.name, "cifar10_clean_accuracy": clean_accuracy,
                       "cifar10c_evaluated": False}

        out_csv = args.output_root / f"{run_dir.name}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "corruption", "severity", "accuracy", "n"])
            writer.writeheader()
            writer.writerows(rows)
        (args.output_root / f"{run_dir.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
