"""Aggregate completed per-seed training and evaluation outputs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("results/final"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--output-root", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    groups = defaultdict(list)
    eval_rows = defaultdict(list)
    type_rows = defaultdict(list)
    severity_rows = defaultdict(list)
    for run_dir in sorted(args.run_root.iterdir()):
        summary_file = run_dir / "summary.json"
        eval_file = args.evaluation_root / f"{run_dir.name}.json"
        csv_file = args.evaluation_root / f"{run_dir.name}.csv"
        if not summary_file.exists() or not eval_file.exists():
            continue
        method = run_dir.name.rsplit("_seed", 1)[0]
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_file.read_text(encoding="utf-8"))
        groups[method].append({**summary, **evaluation})
        if csv_file.exists():
            with csv_file.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["dataset"] == "CIFAR-10-C" and row["corruption"] != "mean_over_corruptions_and_severities":
                        accuracy = float(row["accuracy"])
                        key = (method, row["corruption"], row["severity"])
                        eval_rows[key].append(accuracy)
                        type_rows[(method, row["corruption"])].append(accuracy)
                        severity_rows[(method, row["severity"])].append(accuracy)

    aggregate = []
    for method, runs in sorted(groups.items()):
        fields = [
            ("cifar10_clean_accuracy", "clean_accuracy"),
            ("cifar10c_mean_accuracy", "cifar10c_mean_accuracy"),
            ("training_seconds_including_epoch_data_loading", "training_seconds"),
            ("source_examples_per_second_overall", "source_examples_per_second"),
            ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_bytes"),
        ]
        seed_ids = [str(r["run"].rsplit("_seed", 1)[1]) for r in runs]
        row = {"method": method, "completed_runs": len(runs), "seeds": ",".join(seed_ids)}
        for source, target in fields:
            values = [float(r[source]) for r in runs if source in r and r[source] is not None]
            if values:
                row[target + "_mean"], row[target + "_sd"] = mean_sd(values)
            else:
                row[target + "_mean"], row[target + "_sd"] = "", ""
        aggregate.append(row)

    columns = ["method", "completed_runs", "seeds", "clean_accuracy_mean", "clean_accuracy_sd",
               "cifar10c_mean_accuracy_mean", "cifar10c_mean_accuracy_sd", "training_seconds_mean",
               "training_seconds_sd", "source_examples_per_second_mean", "source_examples_per_second_sd",
               "peak_cuda_memory_bytes_mean", "peak_cuda_memory_bytes_sd"]
    write_csv(args.output_root / "condition_summary.csv", aggregate, columns)

    corruption_summary = []
    for (method, corruption, severity), values in sorted(eval_rows.items()):
        avg, sd = mean_sd(values)
        corruption_summary.append({"method": method, "corruption": corruption, "severity": severity,
                                   "completed_checkpoint_evaluations": len(values),
                                   "accuracy_mean": avg, "accuracy_sd": sd})
    write_csv(args.output_root / "cifar10c_by_corruption.csv", corruption_summary,
              ["method", "corruption", "severity", "completed_checkpoint_evaluations", "accuracy_mean", "accuracy_sd"])
    by_type = []
    for (method, corruption), values in sorted(type_rows.items()):
        by_type.append({"method": method, "corruption": corruption,
                        "severity_averaged_accuracy_mean": statistics.mean(values),
                        "seed_severity_pairs": len(values)})
    write_csv(args.output_root / "cifar10c_by_type.csv", by_type,
              ["method", "corruption", "severity_averaged_accuracy_mean", "seed_severity_pairs"])
    by_severity = []
    for (method, severity), values in sorted(severity_rows.items()):
        by_severity.append({"method": method, "severity": severity,
                            "corruption_averaged_accuracy_mean": statistics.mean(values),
                            "seed_corruption_pairs": len(values)})
    write_csv(args.output_root / "cifar10c_by_severity.csv", by_severity,
              ["method", "severity", "corruption_averaged_accuracy_mean", "seed_corruption_pairs"])
    print(f"Wrote {args.output_root / 'condition_summary.csv'}")
    print(f"Wrote {args.output_root / 'cifar10c_by_corruption.csv'}")
    print(f"Wrote {args.output_root / 'cifar10c_by_type.csv'}")
    print(f"Wrote {args.output_root / 'cifar10c_by_severity.csv'}")


if __name__ == "__main__":
    main()
