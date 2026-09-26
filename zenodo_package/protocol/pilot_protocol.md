# Frozen Pilot Protocol

**Version:** 0.1  
**Date:** 2026-09-25  
**Purpose:** A small experimental-paper deliverable under a two-day deadline. This is a pilot/replication-style comparison, not a claim of publication-level novelty.

## Research question

Under a fixed short training schedule on CIFAR-10, how do standard crop-and-flip augmentation, RandAugment, and AugMix compare in clean accuracy, corruption robustness, and measured training time for MobileNetV3-Small?

## Fixed experimental design

- **Dataset:** torchvision CIFAR-10 training split and official CIFAR-10 test set. For corruption evaluation, use the original 15 CIFAR-10-C types—brightness, contrast, defocus blur, elastic transform, fog, frost, Gaussian noise, glass blur, impulse noise, JPEG compression, motion blur, pixelate, shot noise, snow, and zoom blur—at severities 1–5. Ignore extra corruption arrays in expanded archive versions. No CIFAR-10.1 in this pilot.
- **Model:** torchvision MobileNetV3-Small with first convolution stride changed from 2 to 1 and final classifier changed to 10 classes; trained from random initialization.
- **Input:** native 32x32 RGB images; crop padding 4; CIFAR-10 channel mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`.
- **A / baseline:** random crop with padding 4 and random horizontal flip.
- **B / RandAugment:** A plus torchvision `RandAugment(num_ops=2, magnitude=9)`.
- **C / AugMix:** same random crop/flip, then three views (clean and two independently generated torchvision `AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)` views); average cross-entropy over the three views plus Jensen-Shannon consistency loss weighted by 12.
- **Optimizer:** SGD, learning rate 0.05, momentum 0.9, weight decay 0.0005; cosine annealing over the fixed run length.
- **Seeds:** 17 and 23 for every final condition. Seed 99 is reserved for one-epoch runtime/memory pilots of each condition and excluded from accuracy comparisons.
- **Schedule:** 50 epochs, batch size 128, and two data-loader workers for all final runs. These settings were fixed after one-epoch feasibility runs and before any test-set evaluation. Keep training examples, optimizer, and final-epoch checkpoint rule identical.
- **Timing:** end-to-end wall time from the first training epoch through the last, including data loading and online augmentation, excluding initial dataset download, model setup, and evaluation. Report source CIFAR examples/second. For AugMix, each source image creates three model inputs; report this distinction.
- **Primary metrics:** CIFAR-10 test top-1 accuracy; CIFAR-10-C top-1 accuracy per corruption and severity and mean over corruption-severity pairs if the canonical complete benchmark is used; training time, throughput, peak allocated GPU memory, and parameter count.

## Data separation and reporting rules

- Do not use clean test or CIFAR-10-C results to select hyperparameters, stopping epoch, or preferred condition.
- Use the final-epoch checkpoint for every run; no best-test checkpoint selection.
- Report each completed seed. If both seeds complete, report mean and standard deviation. With one seed, label results single-seed exploratory; do not report inferential claims.
- If only a subset of CIFAR-10-C is evaluated, list the exact corruptions/severities and never call it the full benchmark.
- Do not silently replace failed runs. Record technical failures and any rerun.
- Report incomplete conditions plainly. Do not invent or infer missing values.

## Deviations to record

| Date/time | Planned setting | Actual change | Reason | Runs affected |
|---|---|---|---|---|
| 2026-09-25 | Runtime-only one-epoch pilot, seed 99 | CIFAR-10 downloaded; baseline 42.2 s/epoch, RandAugment 20.9 s/epoch, AugMix 89.9 s/epoch; peak allocated GPU memory 205 MB, 205 MB, and 555 MB respectively. The first baseline run included one-time GPU warm-up; these one-epoch timings are feasibility observations, not final comparative estimates. | Fixed 50 epochs, batch size 128, workers 2 for each final run; seeds 17 and 23. | Baseline seed 17 complete (710.7s); RandAugment seed 17 complete (729.2s); AugMix seed 17 next |
| 2026-09-25 | CIFAR-10-C acquisition | Official Zenodo archive downloaded; MD5 `56bf5dcef84df0e2308c6dcbcbbd8499` verified. | Archive checksum matches the Zenodo record. | Extract after training to avoid competing with timed runs. |
| 2026-09-25 | AugMix seed 17, epoch 1 | 217.4 s (230 source images/s), compared with 89.9 s in the separate one-epoch seed-99 feasibility run. | The interim timing is slower than the pilot estimate; no test results have been seen and no settings were changed. | Use complete-run wall time in the final comparison. |
| 2026-09-25 | AugMix seed 17, complete 50-epoch run | 8,257.7 s (137.6 min), 302.7 source examples/s, 557,116,928 peak allocated CUDA bytes. | The full-run time is far above the one-epoch feasibility estimate; online transformation and three-view forward/backward computation are included. The other methods remain measured under the same timing boundary. | Report the observed cost and the pilot/full-run discrepancy; do not compare against the one-epoch estimate as if it were final. |
| 2026-09-25 | RandAugment seed 23, complete 50-epoch run | 1,558.5 s, compared with 729.2 s for seed 17 under the same settings. | Per-epoch throughput became substantially lower during this run; no configuration changes were made. | Preserve each run's wall time and note local timing variability when interpreting the two-seed means. |
| 2026-09-25 | AugMix seed 23, complete 50-epoch run | 6,307.3 s (105.1 min); epoch 45 took 865.9 s while most late epochs took 80-110 s. | Large wall-clock variability occurred within and between matched runs; hardware/system state was not controlled tightly enough to explain it. | Preserve epoch log and report actual run-level times with a strong timing limitation. |
| 2026-09-25 | Held-out evaluation | All six final checkpoints evaluated on clean CIFAR-10 and 15 CIFAR-10-C corruptions at five severity levels. | The archive contains 19 arrays; the evaluator includes the canonical 15 and excludes the four additional arrays. | Per-run outputs are in `results/evaluation/`; aggregates are in `results/analysis/`. |

## Literature boundary

Fadhilah, Warsito, and Faridah (2026) already study mobile models including MobileNetV3-Small on CIFAR-10/CIFAR-10-C, compare a distortion augmentation with a standard recipe, and report training time. This pilot does not claim to establish a new gap or a new method. It instead measures a small head-to-head comparison of different standard augmentation policies under one controlled local setup. The related-work discussion is included in the manuscript.
