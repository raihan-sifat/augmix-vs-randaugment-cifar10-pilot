# Overview of My Research Paper

**Title:** *Training-Time Image Augmentation on MobileNetV3-Small: A Controlled CIFAR-10 Pilot*
**Author:** Sifat Raihan — Department of Computer Science and Technology, Hebei University of Science and Technology, Shijiazhuang, China
**Published:** Zenodo preprint, Version 1.0, 2026-09-25 — DOI `10.5281/zenodo.22961521` — CC BY 4.0
**Local copy:** `paper_draft.md` / `paper_draft.pdf` in this repository

> **Read this first.** This document explains the research in plain language, then gives the technical
> detail. It is designed so that (a) a professor can understand the work in 10 minutes, and
> (b) I can revise any section quickly before an interview.
>
> **The single most important thing to remember:** this is an **exploratory pilot**, not a
> breakthrough. Its honest headline is: *"stronger augmentation buys a lot of corruption
> robustness and costs a lot of training time, and we measured exactly how much on one
> laptop GPU."* Never claim novelty. Never call it peer-reviewed.

---

## Table of contents

1. [The 30-second version](#1-the-30-second-version)
2. [The problem, explained simply](#2-the-problem-explained-simply)
3. [Key concepts you must understand](#3-key-concepts-you-must-understand)
4. [The exact research question](#4-the-exact-research-question)
5. [Experimental design: what was fixed, what varied](#5-experimental-design-what-was-fixed-what-varied)
6. [How the work was done, step by step](#6-how-the-work-was-done-step-by-step)
7. [Results — every number](#7-results--every-number)
8. [What the results actually mean](#8-what-the-results-actually-mean)
9. [The timing problem, and what the raw logs really show](#9-the-timing-problem-and-what-the-raw-logs-really-show)
10. [Tools and software I used](#10-tools-and-software-i-used)
11. [Exactly how I used each tool](#11-exactly-how-i-used-each-tool)
12. [The code, file by file](#12-the-code-file-by-file)
13. [Reproducibility and data provenance](#13-reproducibility-and-data-provenance)
14. [Honest limitations](#14-honest-limitations)
15. [Things I would do next](#15-things-i-would-do-next)
16. [What this project proves I can do](#16-what-this-project-proves-i-can-do)
17. [Flashcard sheet — every key number](#17-flashcard-sheet--every-key-number)

---

## 1. The 30-second version

A neural network can recognise a blurry, noisy, snowy or foggy image much worse than a clean one.
This is a real problem for any model that has to work outside a laboratory. One popular remedy is
**data augmentation** — artificially damaging the training images (rotating, blurring, posterising,
adding noise) so the model learns to still work when real images are damaged.

But there is a catch: **stronger augmentation is not free.** It can slow training down a lot, and it
can cost you a little accuracy on clean images. Most people who use these methods quote only the
accuracy benefit and ignore the cost.

**What I did.** I took one small, efficient image classifier (MobileNetV3-Small), one small dataset
(CIFAR-10), and trained it three different ways — changing *only* the image augmentation and nothing
else:

| Condition | Augmentation used |
|---|---|
| **A — Crop-and-flip** (baseline) | random crop + random horizontal flip |
| **B — RandAugment** | crop-and-flip **+ 2 random operations at strength 9** |
| **C — AugMix** | crop-and-flip **+ 3 views per image with a consistency (Jensen-Shannon) loss** |

Everything else was held identical: same architecture (1,528,106 parameters, trained from scratch,
no pretrained weights), same 50,000 training images, same 50 epochs, same batch size 128, same
optimiser and learning-rate schedule, same seeds (17 and 23), same machine (one GTX 1650 laptop GPU).

I then measured three things for each condition:

1. **Clean accuracy** — normal test images.
2. **Corruption accuracy** — 15 kinds of corrupted images at 5 damage levels (750,000 images per
   model, the CIFAR-10-C benchmark).
3. **Cost** — wall-clock training time, images processed per second, and peak GPU memory.

**Headline result.**

| | Clean accuracy | Corrupted accuracy | Training time |
|---|---:|---:|---:|
| Crop-and-flip | **86.39 %** | 67.04 % | **13.2 min** |
| RandAugment | 86.18 % | 71.00 % (+3.96) | 19.1 min |
| AugMix | 86.02 % | **77.82 % (+10.78)** | 121.4 min |

So: **clean accuracy is essentially identical across all three (0.37 points apart — pure noise),
but corruption robustness differs enormously, and training cost differs enormously too.**

That is the whole story. AugMix bought +10.78 points of corruption robustness and cost roughly
6–9× the training time. Crop-and-flip is 11 points worse on corruptions and is nearly free.

---

## 2. The problem, explained simply

### 2.1 The intuition

Imagine you are training a child to recognise ten animals from photos. If every photo is clean and
well-lit, the child learns "shape" cues. Then you hand the child a photo in the dark, or a photocopy,
or a photo covered in snow. The child's accuracy collapses — not because the child is stupid, but
because **the child never saw anything like that during training**.

A neural network is exactly the same. Its first layer learns simple detectors (edges, colour
blobs). If training data only ever contains clean images, those detectors are tuned to clean signal.
Corrupted images push the input away from anything the network has seen, so the features degrade
downstream and accuracy falls.

**Data augmentation is the fix:** show the network damaged versions during training, so it learns
detectors that survive damage. This is sometimes called *training-time* augmentation, as opposed to
*test-time* augmentation, which only happens at inference.

### 2.2 Why this is not a solved problem

Three established methods exist, and they work in different ways:

- **RandAugment** (Cubuk et al., CVPR Workshops 2020): throw a small number of random image
  operations at each training image. Simple, cheap, and removes the need to *search* for a good
  augmentation policy, which older automatic methods (AutoAugment) had to do expensively.
- **AugMix** (Hendrycks et al., ICLR 2020): build several differently-augmented versions of the same
  image, mix them together, and additionally penalise the network for giving *different predictions
  on different versions of the same image*. That consistency pressure is the key idea.
- **Plain crop-and-flip:** the traditional recipe. Almost free, and already surprisingly strong.

But: **which one should you use, and what does it cost?** Most published comparisons report accuracy
and quietly omit the compute bill. On a small laptop GPU, a 6–9× training-time difference is the
difference between "I can run this experiment" and "I cannot."

### 2.3 The specific gap I addressed

There is very little clean, matched-setting, cost-reported comparison of these policies on
*lightweight* (mobile-oriented) classifiers. I filled that hole — at a very small scale, honestly
labelled as a pilot.

---

## 3. Key concepts you must understand

### 3.1 CIFAR-10

- 60,000 colour images, **32×32 pixels** each (tiny — about 3 KB per image).
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- Split: **50,000 train** / **10,000 test**. The test split is the standard held-out benchmark.
- I trained on all 50,000 training images and used the 10,000 test images only for final evaluation,
  once, after all training was finished.

### 3.2 CIFAR-10-C

Created by Hendrycks and Dietterich (ICLR 2019). It is CIFAR-10's test set with **15 corruption
types** applied at **5 severity levels**. So one model is tested on
`15 × 5 × 10,000 = 750,000` corrupted images.

The 15 canonical corruptions, grouped:

| Family | Corruptions |
|---|---|
| **Noise** | Gaussian noise, impulse noise, shot noise |
| **Blur** | defocus blur, glass blur, motion blur, zoom blur |
| **Weather** | fog, frost, snow |
| **Digital** | brightness, contrast, elastic transform, JPEG compression, pixelate |

Severity 1 = mild, severity 5 = extreme. Everything gets worse as severity rises — always, in my
results, for every method.

**Important caveat:** these are *synthetic* corruptions generated by image-processing code, not
real photographs of bad weather. Passing CIFAR-10-C does **not** mean a model is robust in the real
world. I state this explicitly in the paper.

**Detail worth knowing:** the archive I downloaded contains **19** corruption arrays, not 15. The
four extra are `gaussian_blur`, `saturate`, `spatter`, and `speckle_noise`. I deliberately used only
the canonical 15 so my numbers remain comparable to the published benchmark. The code hard-fails if
any canonical corruption file is missing, and it explicitly selects only the 15.

### 3.3 MobileNetV3-Small

- A **mobile-oriented** CNN: designed to be small and fast enough to run on phones and IoT devices.
  "Small" refers to the *deployed* model. Comes in Small and Large variants; I used **Small**.
- Built from three ideas: **depthwise separable convolutions** (split a normal convolution into two
  cheaper ones), **inverted residual blocks** (bottleneck blocks with expansion), and
  **hard-swish / squeeze-and-excitation** activation and gating.
- torchvision's version has **2,542,856** parameters for 1,000 ImageNet classes.

**My modification (this matters a lot for cost):** MobileNetV3 expects 224×224 images. Its very
first convolution uses stride 2, which halves the resolution immediately (224 → 112). For a 32×32
CIFAR image, that aggressive first downsampling destroys too much information — the first feature
map would be only 16×16. So I set the first convolution's stride to **1**, giving a 32×32 first
feature map, and I replaced the 1,000-class classifier with a **10-class** linear layer.

Result: **1,528,106 trainable parameters.** I verified this number directly in PyTorch.

**The hidden cost of that modification:** stride 1 instead of 2 means the early feature maps are
**4× larger in area** (32×32 instead of 16×16). Every convolution in the early and middle layers
therefore costs roughly 4× more. This is the main reason all three of my runs were slow on a
4 GB laptop GPU, and it matters especially for AugMix, which pushes 3× the images through the model
at once. Do not forget this when explaining the timing numbers.

### 3.4 The three augmentation conditions in detail

**Shared by all three (the baseline geometric transforms):**
- `RandomCrop(32, padding=4)` — pad the image with 4 pixels of black on each side, then take a
  random 32×32 window. This creates small translations and simulates slight camera shake.
- `RandomHorizontalFlip()` — mirror the image left-right. Objects are roughly symmetric, so this is
  a safe label-preserving transformation.

Then `ToTensor()` (uint8 → float in [0,1], HWC → CHW) and `Normalize(mean, std)` with the standard
CIFAR-10 channel statistics: mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)`.

**Condition A — Crop-and-flip only.** Nothing else. The reference point.

**Condition B — RandAugment.** `RandAugment(num_ops=2, magnitude=9)`.
- `num_ops=2`: apply **exactly 2** operations per image, sampled uniformly at random.
- The 14 candidate operations in the installed torchvision version: Identity, ShearX, ShearY,
  TranslateX, TranslateY, Rotate, Brightness, Color, Contrast, Sharpness, Posterize, Solarize,
  AutoContrast, Equalize.
- `magnitude=9`: strength index 9 on a scale of 0–30 (`num_magnitude_bins=31`). So it is a fairly
  *aggressive* setting — roughly 30 % of the way up the strength range. About half the operations
  (the signed ones) also get a random sign, so a shear or rotation can go either direction.
- Note the "Identity" op: with 14 ops, the chance both sampled ops are Identity is small but
  non-zero, so some images are barely augmented.
- **These are torchvision's default values.** I did not tune them.

**Condition C — AugMix (the full method, not just the transform).** This is the most important
technical detail in the whole project, and the most commonly misunderstood.

The AugMix *paper* has two parts. Many people implement only the first part. I implemented both:

**Part 1 — the transform.** `AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)`
again, torchvision's defaults, 13 operators available (all of the RandAugment ones plus Color and
Contrast, minus Identity/Cutout). For one call it does:
1. Build `mixture_width = 3` **augmentation chains**. Each chain applies a random number of random
   operations. `chain_depth=-1` means the depth is **sampled randomly from 1 to 3**.
2. Sample 3 weights from a **Dirichlet distribution** with `alpha = 1.0` (Dirichlet(1) is uniform on
   the simplex, so all weightings are equally likely a priori) and use them to combine the 3 chain
   outputs into a single blended image. The "mixing" is what makes the result look like several mild
   corruptions layered together rather than one brutal one.

**Part 2 — the multi-view consistency objective.** For **each source image** I produce **three
views**:
- View 1: the crop-and-flip image (I call this the "clean" view — it is *geometrically* augmented but
  not photometrically damaged; the paper's wording is "clean view")
- View 2: an independently generated AugMix mix
- View 3: a second, independently generated AugMix mix

All three go through the network **in one concatenated forward pass**. The loss is:

```
loss  =  mean( cross-entropy(view1), cross-entropy(view2), cross-entropy(view3) )   # ordinary CE
      +  12 × Jensen-Shannon-divergence( predictions of view1, view2, view3 )          # consistency
```

- The **Jensen-Shannon divergence (JSD)** measures how differently the three views are predicted.
  `JSD = [ KL(p1‖M) + KL(p2‖M) + KL(p3‖M) ] / 3` where `M = (p1+p2+p3)/3`. Bounded in `[0, ln 2]`,
  so it cannot blow up. Multiplied by 12, it is a strong but bounded regulariser.
- The weight **12** comes from the original AugMix paper's CIFAR/ImageNet recipes. I did not tune it.
- The whole point: the network is told *"these three pictures are the same object, so you must agree
  about what they are."* That is what transfers to unseen corruptions — the model learns features
  that are invariant to photometric damage, not just to flips and crops.

**If a professor asks "did you really implement AugMix, or just an augmentation transform?" — the
answer is the full method, and this is the strongest technical point in the project.**

### 3.5 Words you will be asked to define

| Term | Meaning |
|---|---|
| **Epoch** | One full pass over all 50,000 training images. I trained for 50 epochs. |
| **Top-1 accuracy** | Fraction of images where the highest-scoring class is correct. |
| **Batch size** | Images processed per optimiser step. 128 (AugMix effectively 384 model inputs). |
| **SGD** | Stochastic gradient descent — the optimiser that updates the weights. |
| **Momentum (0.9)** | Carries 90 % of the previous step's direction forward, smoothing and speeding up convergence. |
| **Weight decay (5e-4)** | L2 penalty on weights; a mild regulariser. |
| **Cosine annealing** | Learning rate starts at 0.05 and follows a half-cosine curve down to ~0 over the 50 epochs. Large steps early, tiny steps late — good for final accuracy. |
| **Backpropagation** | Computing how much each weight contributed to the error, then adjusting it. |
| **JSD** | Jensen-Shannon divergence — a bounded, symmetric measure of how far apart two or more probability distributions are. |
| **mCE** | The CIFAR-10-C paper's normalised "mean Corruption Error" metric. **I did not use it** — I report plain accuracy. A known limitation. |
| **Top-1 vs top-5** | I only measured top-1. |
| **FLOPs** | Floating-point operations — a hardware-independent compute measure. **I did not measure it.** |
| **CIFAR-10.1** | A separately *collected* CIFAR-10 test set. **I did not use it.** A planned follow-up. |
| **Preprint** | Publicly archived, **not peer-reviewed.** My Zenodo DOI is a preprint DOI, not a journal DOI. |

---

## 4. The exact research question

> **Under a fixed short training schedule on CIFAR-10, how do crop-and-flip, RandAugment, and AugMix
> compare in clean accuracy, CIFAR-10-C accuracy, and measured training time for MobileNetV3-Small?**

Supporting questions I asked:

1. Which method is best on clean images?
2. Which method is best across corruption types and severity levels?
3. How much extra end-to-end training time does each method need?
4. Do the conclusions hold across different random seeds?
5. Does improvement on synthetic corruptions generalise? *(planned, not done — needs CIFAR-10.1)*

My **pre-registered hypotheses** (written down before running, and stated as hypotheses rather than
expectations):

- Stronger augmentation should improve average corrupted-image accuracy.
- Stronger augmentation may reduce clean accuracy and increase training time.
- Gains will not be uniform across corruption types.
- Gains on synthetic corruptions may not transfer to naturally-collected data.

All three of the first four were supported; the last remains untested.

---

## 5. Experimental design: what was fixed, what varied

**The design principle: change exactly one thing.** Every condition used identical architecture,
identical data, identical optimiser, identical schedule, identical seeds. The *only* difference
between conditions is the augmentation function.

### Held constant (the control variables)

| Category | Setting |
|---|---|
| Architecture | torchvision MobileNetV3-Small, first conv stride 2→1, 10-class head, **from random init (no pretrained weights)** |
| Parameters | 1,528,106 trainable |
| Input | native 32×32 RGB, no resizing |
| Normalisation | mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)` |
| Training data | all 50,000 CIFAR-10 train images |
| Epochs | 50 |
| Batch size | 128 (2 for baseline/RandAugment, 3 views so 384 model inputs for AugMix) |
| Data-loader workers | 2 |
| Optimiser | SGD, lr 0.05, momentum 0.9, weight decay 0.0005 |
| LR schedule | `CosineAnnealingLR`, `T_max = 50` |
| Seeds | **17 and 23** (identical for every condition) |
| Checkpoint rule | **final epoch only** — never the best test epoch |
| Evaluation | deterministic, no test-time augmentation |
| Hardware | one NVIDIA GeForce GTX 1650, 4 GB VRAM (4,294,639,616 bytes) |
| Software | Windows 11, Python 3.14.5, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, NumPy 2.5.2, Pillow 12.3.0 |
| Timing boundary | `perf_counter` from start of epoch 1 to end of epoch 50, `cuda.synchronize()` before stopping; **includes** data loading and online augmentation; **excludes** dataset download, model construction, and evaluation |

### Varied (the independent variable)

| Condition | Augmentation | Augmentation hyperparameters |
|---|---|---|
| A `baseline` | crop-and-flip only | — |
| B `randaugment` | + RandAugment | `num_ops=2, magnitude=9` (torchvision defaults) |
| C `augmix` | + 3 views (1 crop-flip + 2 AugMix) + JSD | `severity=3, mixture_width=3, chain_depth=-1, alpha=1.0` (torchvision defaults), JSD weight 12 |

### Deliberate experimental-hygiene rules

These are the things that separate careful work from careless work. Memorise them:

1. **The test set was never used for anything except final evaluation.** Not for training, not for
   hyperparameter choice, not for early stopping, not for picking a winner. Augmentation
   hyperparameters are torchvision defaults / original-paper values, frozen in a written protocol
   dated **2026-09-25** *before* any test evaluation.
2. **No best-checkpoint selection.** Every result uses the **final-epoch** checkpoint. This is
   slightly worse for accuracy but removes the single most common source of test-set leakage.
3. **Settings were frozen after 1-epoch feasibility pilots (seed 99) and before evaluation.** The
   pilots were used *only* to estimate runtime and memory — never to compare accuracy.
4. **Same seeds across conditions** (17, 23), so differences are paired rather than confounded.
5. **All six runs completed.** No run was discarded, retried, or replaced.
6. **A written deviation log** (`pilot_protocol.md`) records every timing anomaly and every
   judgement call, dated, with the reason.
7. **No test-set-based "exploratory analysis"** — the analysis script only aggregates saved outputs.

---

## 6. How the work was done, step by step

### Step 1 — Literature screening (2026-09-25)

I searched primary sources for work combining `CIFAR-10-C`, augmentation, lightweight/mobile CNNs,
and runtime measurement, and built a source matrix with columns for research question, dataset,
model, methods, metrics, main result, limitations, and relevance. Key sources:

- Hendrycks & Dietterich 2019 (CIFAR-10-C), Hendrycks et al. 2020 (AugMix), Cubuk et al. 2020
  (RandAugment), Yin et al. 2019 (Fourier analysis of which corruptions are hard), Mintun et al. 2021
  (augmentation × corruption interaction), Modas et al. 2022 (PRIME), Kireev et al. 2022
  (adversarial training × corruption, with a wall-clock table), Trinh et al. 2024, and
  **Fadhilah, Warsito & Faridah 2026** (DistortionMix, JIKI, DOI 10.21609/jiki.v19i1.1558).

**Critical finding:** the 2026 DistortionMix paper already evaluates **MobileNetV3-Small** on
**CIFAR-10 and CIFAR-10-C**, compares **a baseline recipe against a distortion augmentation**, and
reports **50-epoch training times**. That combination is almost exactly my setup. I therefore made
**no novelty claim**, and repositioned the work as a *controlled, matched-setting, replication-style
pilot of different policies with honest cost reporting.* I logged this in `literature_review.md`,
`plan.md`, `pilot_protocol.md`, and the paper's Related Work and Limitations sections.

### Step 2 — Environment setup

Created a project-local virtual environment `.venv` (Python 3.14.5), installed
`torch 2.14.0+cu130` and `torchvision 0.29.0+cu130`, and verified CUDA + the GTX 1650 were actually
usable. The training script **refuses to run on CPU** (`raise RuntimeError`) so no silent CPU
fallback could contaminate the timing measurements.

### Step 3 — Writing the frozen protocol

Wrote `pilot_protocol.md`: the research question, every fixed setting, the data-separation rules, the
exclusion rules, and an empty deviation log. Dated and versioned 0.1. This is the document that makes
the work "pre-registered" in spirit.

### Step 4 — Feasibility pilots (seed 99, 1 epoch each)

Purpose: estimate runtime and peak memory only, so I could pick a 50-epoch schedule that fits a
two-day deadline. Results: baseline 42.2 s/epoch, RandAugment 20.9 s/epoch, AugMix 89.9 s/epoch;
peak allocated GPU memory 205 MB / 205 MB / 555 MB. **Excluded from all accuracy analysis.** Then I
froze 50 epochs, batch 128, 2 workers.

### Step 5 — The six final training runs

```powershell
.\.venv\Scripts\python.exe src\train.py --method baseline     --seed 17 --epochs 50 --batch-size 128 --workers 2 --download --output-root results\final
.\.venv\Scripts\python.exe --method randaugment --seed 17 ...   (and --method augmix --seed 17 ...)
# then repeat all three with --seed 23
```

Each run wrote `config.json` (every setting + full software/hardware manifest), `epochs.csv`
(per-epoch loss, accuracy, seconds, throughput, next-epoch LR), `model.pt`, and `summary.json`
(total seconds, throughput, peak CUDA bytes, final train accuracy, parameter count).

Total timed training: **18,438.9 s ≈ 5.12 hours** for the six runs (AugMix alone was 14,565 s ≈ 4.05 h).

### Step 6 — Acquiring CIFAR-10-C

Downloaded the official archive from Zenodo record `2535967` on 2026-09-25 and **verified its MD5
checksum** `56bf5dcef84df0e2308c6dcbcbbd8499` against the Zenodo record. Extraction was deliberately
delayed until after the timed training runs so that disk I/O could not compete with the timing
measurements — a small but genuinely careful decision.

### Step 7 — Held-out evaluation

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint-root results\final --data-root data --cifar-c-root data\CIFAR-10-C --output-root results\evaluation
```

For **each of the 6 checkpoints**: 10,000 clean test images + 750,000 corrupted images =
**760,000 forward passes per checkpoint, 4,560,000 total**. Evaluation is fully deterministic —
`torch.inference_mode()`, `model.eval()`, no random transforms, no test-time augmentation.

The evaluator hard-fails if any of the 15 canonical corruption files is missing, and explicitly
selects only those 15, ignoring the 4 extra arrays present in the archive.

### Step 8 — Aggregation

```powershell
.\.venv\Scripts\python.exe src\summarize.py --run-root results\final --evaluation-root results\evaluation --output-root results\analysis
```

Produced `condition_summary.csv` (per-condition mean ± SD), `cifar10c_by_corruption.csv` (each type ×
severity), `cifar10c_by_type.csv` (per type, averaged over 5 severities × 2 seeds),
`cifar10c_by_severity.csv` (per severity, averaged over 15 corruptions × 2 seeds). All paper tables
were generated from these files; **no number in the paper was typed by hand or edited afterwards.**

### Step 9 — Writing and releasing

Wrote `paper_draft.md` (Markdown source, 151 lines), rendered `paper_draft.pdf`, built
`zenodo_reproducibility.zip` (34.5 MB: scripts, protocol, configs, logs, checkpoints, evaluation
outputs, analysis tables — **excluding the datasets**), wrote the metadata, and published on Zenodo
as Version 1.0, resource type **Preprint**, CC BY 4.0, DOI `10.5281/zenodo.22961521`
(concept DOI `10.5281/zenodo.22961520`).

---

## 7. Results — every number

### 7.1 Run completion

| Condition | Seeds completed | Epochs each | Failed runs |
|---|---:|---:|---:|
| Crop-and-flip | 2 | 50 | 0 |
| RandAugment | 2 | 50 | 0 |
| AugMix | 2 | 50 | 0 |

All six planned runs finished. No substitutions, no silent retries.

### 7.2 Clean and corrupted accuracy

| Condition | Clean, seed 17 | Clean, seed 23 | **Clean, mean ± SD** | CIFAR-10-C, seed 17 | CIFAR-10-C, seed 23 | **CIFAR-10-C, mean ± SD** |
|---|---:|---:|---:|---:|---:|---:|
| Crop-and-flip | 86.21 % | 86.57 % | **86.39 ± 0.25 %** | 68.19 % | 65.89 % | 67.04 ± 1.63 % |
| RandAugment | 86.76 % | 85.60 % | 86.18 ± 0.82 % | 70.95 % | 71.04 % | 71.00 ± 0.06 % |
| AugMix | 86.03 % | 86.01 % | 86.02 ± 0.01 % | 78.04 % | 77.61 % | **77.82 ± 0.31 %** |

Differences relative to the crop-and-flip baseline:
- Clean: RandAugment **−0.21 pp**, AugMix **−0.37 pp** — both far inside seed noise.
- CIFAR-10-C: RandAugment **+3.96 pp**, AugMix **+10.78 pp**.

### 7.3 Training cost

| Condition | Time seed 17 (s) | Time seed 23 (s) | **Time, mean ± SD (s)** | Mean minutes | Source img/s (mean) | **Peak GPU memory** |
|---|---:|---:|---:|---:|---:|---:|
| Crop-and-flip | 710.7 | 875.6 | **793.1 ± 116.6** | 13.2 | 3,186.5 | 205,514,240 B (0.206 GB) |
| RandAugment | 729.2 | 1,558.5 | **1,143.8 ± 586.4** | 19.1 | 2,516.4 | 205,514,240 B (0.206 GB) |
| AugMix | 8,257.7 | 6,307.3 | **7,282.5 ± 1,379.1** | 121.4 | 349.6 | 557,116,928 B (0.557 GB) |

Derived ratios from the reported means: AugMix / baseline = **9.2×** the time, **9.1×** lower
throughput, **2.71×** the peak memory. RandAugment / baseline = **1.44×** the time.
AugMix's *model-input* throughput is ~1,049 inputs/s (3 inputs per source image), still ~3.0× below
the baseline's 3,186/s.

⚠️ **Peak GPU memory is `torch.cuda.max_memory_allocated`, not total GPU usage.** It counts only
allocated CUDA tensors. It excludes the CUDA context, cuDNN workspaces, allocator fragmentation, and
memory that is reserved-but-unused. Never present it as "GPU utilisation."

### 7.4 Accuracy per corruption (averaged over 5 severities × 2 seeds)

| Corruption | Family | Crop-and-flip | RandAugment | AugMix | AugMix − baseline |
|---|---|---:|---:|---:|---:|
| Brightness | digital | 84.50 | 85.22 | 84.56 | **+0.06** |
| Contrast | digital | 59.17 | 65.67 | 71.12 | +11.95 |
| Defocus blur | blur | 74.72 | 78.69 | 83.99 | +9.27 |
| Elastic transform | digital | 76.72 | 78.73 | 81.09 | +4.37 |
| Fog | weather | 74.53 | 77.73 | 78.72 | +4.19 |
| Frost | weather | 68.83 | 72.44 | 76.56 | +7.73 |
| Gaussian noise | noise | 47.75 | 51.50 | 70.81 | **+23.06** |
| Glass blur | blur | 53.66 | 62.15 | 70.61 | +16.94 |
| Impulse noise | noise | 51.83 | 63.36 | 71.87 | **+20.04** |
| JPEG compression | digital | 78.63 | 79.92 | 81.62 | +2.99 |
| Motion blur | blur | 65.89 | 70.53 | 80.14 | +14.24 |
| Pixelate | digital | 71.74 | **70.12** | 81.19 | +9.45 |
| Shot noise | noise | 56.82 | 59.99 | 75.22 | +18.40 |
| Snow | weather | 72.53 | 74.22 | 77.94 | +5.41 |
| Zoom blur | blur | 68.29 | 74.69 | 81.91 | +13.62 |

**Two findings worth highlighting:**

1. **Gains are strongly family-dependent.** The noise family benefits most from AugMix
   (+18 to +23 pp), blur second (+9 to +17 pp), and digital/weather least (+0.06 to +5.4 pp).
   Noise robustness is where the baseline is weakest (47.75 % on Gaussian noise) and where the
   consistency objective has the most room to help.
2. **Gains are not uniform, and one method actually loses.** RandAugment is **worse** than the
   baseline on pixelate (70.12 vs 71.74, −1.62 pp). And AugMix on brightness (84.56 vs 84.50) is a
   statistical tie. This is a concrete, honest counterexample to "more augmentation is always
   better", and it matches Yin et al. (2019), who found corruption-specific augmentation effects.

**Hardest and easiest corruptions are the same for all three methods:** noise hardest
(Gaussian noise lowest), brightness easiest. This ordering matches the published literature.

### 7.5 Accuracy per severity (averaged over 15 corruptions × 2 seeds)

| Severity | Crop-and-flip | RandAugment | AugMix |
|---:|---:|---:|---:|
| 1 | 79.77 | 81.54 | 83.29 |
| 2 | 74.22 | 77.44 | 81.29 |
| 3 | 68.17 | 72.70 | 79.10 |
| 4 | 60.89 | 66.18 | 75.77 |
| 5 | 52.17 | 57.12 | 69.67 |
| **Drop from sev 1 → 5** | **−27.60** | **−24.42** | **−13.62** |

Monotonic degradation for all three methods. **AugMix degrades most gracefully** — less than half
the baseline's total loss across the severity range. The advantage *grows* with severity
(AugMix − baseline: +3.52 at severity 1, +17.5 at severity 5), which is exactly the behaviour you
want from a robustness method.

### 7.6 Final-epoch training accuracy (training metric only — read the warning)

| Condition | seed 17 | seed 23 |
|---|---:|---:|
| Crop-and-flip | 88.77 % | 88.77 % |
| RandAugment | 83.77 % | 82.64 % |
| AugMix | 87.34 % | 87.63 % |

⚠️ **These three numbers are not directly comparable, and if a professor catches you comparing them
you will lose credibility.** Two separate reasons:

1. For AugMix, the logged `train_loss` **includes the 12 × JSD consistency term**, so its loss scale
   is completely different from a plain cross-entropy loss. Never compare AugMix's loss to the others.
2. Training accuracy is computed on *different input distributions*: baseline and AugMix are
   measured on the crop-and-flip view, but RandAugment's is measured on **RandAugment-damaged**
   images. RandAugment's lower "training accuracy" is a property of its measurement, not proof that
   it learned less well — and indeed its clean test accuracy is statistically identical.

### 7.7 One-epoch feasibility pilots (seed 99 — excluded from all accuracy analysis)

| Condition | Total time | Peak allocated memory | Final train accuracy |
|---|---:|---:|---:|
| Crop-and-flip | 42.19 s | 205,514,240 B | 36.42 % |
| RandAugment | 20.86 s | 205,514,240 B | 30.85 % |
| AugMix | 89.86 s | 555,347,456 B | 35.93 % |

⚠️ **Trap:** in these pilots RandAugment (20.9 s) appeared *faster* than the baseline (42.2 s). This
is a **warm-up artefact** — the very first run on the GPU paid one-time cuDNN/kernel initialisation.
In the full runs RandAugment was consistently slower. If a professor asks why the pilot ranking is
reversed, this is the answer, and it shows I understood my own pilot rather than cherry-picking it.

---

## 8. What the results actually mean

### 8.1 The trade-off, stated precisely

There is a **clean-accuracy / corruption-accuracy / compute** trade-off, and it is the whole finding:

- **Clean accuracy is a non-issue.** 0.37 pp separates the best and worst condition, which is
  smaller than the seed-to-seed spread *within* a single condition (SD up to 0.82 pp). The clean
  ranking is not real. Note that RandAugment produced both the single highest clean accuracy
  (86.76 %, seed 17) and the single lowest (85.60 %, seed 23) of all six runs — a textbook
  demonstration that the ordering is noise.
- **Corruption accuracy is a real and large effect.** AugMix's +10.78 pp is roughly 6.6× the
  baseline's own seed-to-seed SD (1.63 pp) and 35× AugMix's SD (0.31 pp). With only two seeds I
  cannot run a significance test, but a gap that large is very unlikely to be a seed artefact.
- **Compute cost is the price, and it is not small.** Roughly 6–9× the wall time and 2.7× the peak
  memory for AugMix.

### 8.2 Why AugMix wins so much on corruptions

Two mechanisms, and I can only claim the first from the literature and the second is my reading:

1. **Consistency regularisation (the paper's own explanation).** By forcing the model to agree
   across photometrically different views of the same object, AugMix learns damage-invariant
   features. This is why its advantage *grows* with severity, and why it concentrates on noise and
   blur — the transforms AugMix's chain actually simulates (solarize, posterize, contrast, equalize,
   sharp blur-like ops, autocontrast) look like extreme versions of those corruptions.
2. **More optimisation per image (an honest confound).** AugMix computes three cross-entropy terms
   and one consistency term per source image, versus one for the others. Some of its advantage may
   come from that extra gradient signal rather than from the augmentation distribution itself. **My
   design cannot separate these two.** This is the single most important missing experiment — see
   §15.

### 8.3 Why the results should not be over-read

- Two seeds. No hypothesis tests. Everything is descriptive.
- One architecture, one 32×32 dataset, 50 epochs (probably not converged), one laptop GPU.
- Hyperparameters are library defaults, not each method's best. So this is a comparison of
  *default-configured* policies, not of * optimally tuned* ones. A professor may reasonably say the
  comparison under-serves RandAugment and AugMix.
- AugMix got 3× the model compute. It is not a compute-matched comparison.

---

## 9. The timing problem, and what the raw logs really show

This is the most technically interesting part of the project, and the part where I can show real
analytical ability. I want to be completely transparent about it.

### 9.1 What I reported, and why it is not trustworthy

The paper reports mean training times with large standard deviations: RandAugment 1,143.8 ± 586.4 s
and AugMix 7,282.5 ± 1,379.1 s. RandAugment seed 23 took 2.14× as long as seed 17 *under identical
settings*. An 865.9-second epoch appeared in AugMix seed 23 while typical epochs took 80–110 s. So I
concluded, correctly, that the timing comparison is "a local observation, not a stable estimate of
augmentation overhead."

### 9.2 What the per-epoch logs actually reveal

Going back to `epochs.csv` (which records `epoch_seconds` for all 50 epochs of every run) shows the
noise is **not** randomly distributed — it is **episodic**, in long contiguous blocks:

| Run | Median epoch | Epochs ≫ median | Pattern |
|---|---:|---|---|
| baseline seed 17 | 13.8 s | 1 (epoch 1: 39.7 s) | clean; one-time GPU warm-up |
| baseline seed 23 | 13.5 s | 11 (epochs 1–11: ~32 s) | slow block at the *start* of the run |
| randaugment seed 17 | 14.5 s | 0 | completely clean |
| randaugment seed 23 | 22.7 s | 24 (epochs 27–50: ~47 s) | sustained slowdown from epoch 27 onward |
| augmix seed 17 | 93.7 s | 16 (epochs 1–8 and 41–50: ~320 s) | slow blocks at both ends |
| augmix seed 23 | 86.4 s | 5 (epochs 1–4: ~330 s; epoch 45: 865.9 s) | slow start + one extreme outlier |

Two important observations:

**(a) Baseline and RandAugment have virtually identical medians** (13.8 / 13.5 s and 14.5 s), even
though their reported totals differ by 23 % and 114 %. The totals are dominated by interference
episodes, not by the method.

**(b) AugMix's epoch times are bimodal** — a fast mode near 80–95 s and a slow mode near 310–330 s.
Both seeds show both modes.

### 9.3 A better estimate of the true augmentation overhead

If I take the **median epoch time** (robust to the interference) instead of the total, the picture
changes a lot:

| Condition | Median epoch (s) | Estimated clean 50-epoch time | vs baseline |
|---|---:|---:|---:|
| Crop-and-flip | ~13.7 | ~11.4 min | 1.00× |
| RandAugment | ~14.5 | ~12.1 min | **~1.06×** |
| AugMix | ~90 | ~75 min | **~6.6×** |

So the honest reading is:

- **RandAugment is essentially free** — roughly 5–10 % overhead, not 44 %.
- **AugMix's real overhead is roughly 6–7×**, not 9.2×.
- The reported means are **inflated by machine interference**, not by the augmentation methods.

⚠️ **I must present this as a re-analysis of data I already had, not as a new experiment**, and I must
be explicit that even medians are contaminated by the same unknown interference. It is a *hypothesis
that a controlled re-run would test*, not an established result. The paper's own conservative
wording — that timing is a local observation — remains the correct published claim.

### 9.4 Probable causes (hypotheses, not measured)

1. **Background load on a shared laptop** — another process competing for CPU/disk/GPU. Consistent
   with the long contiguous blocks rather than per-epoch jitter.
2. **Thermal or power-plan throttling** — a GTX 1650 in a laptop will throttle under sustained load.
   Consistent with the *late*-run slow blocks (randaugment seed 23 epochs 27–50, augmix seed 17
   epochs 41–50).
3. **Data-loader starvation.** With only `num_workers=2` and heavy PIL-based augmentation, the CPU
   input pipeline can become the bottleneck. AugMix is the worst case: each source image triggers
   2 AugMix calls, each building 3 chains of 1–3 ops, with internal PIL↔tensor round-trips.
   **This is my leading hypothesis for the ~2× of AugMix's cost that the 3× batching alone cannot
   explain** (3× batching predicts ≈3×; I observe ≈6.6×).
4. **Windows scheduler / Defender scanning** the data directory.

**What I could not do:** the paper explicitly states I did not separate image-transform time from
model-compute time, and did not measure energy. So I cannot *prove* the decomposition — I can only
reason about it. I state that as a limitation rather than as a finding.

### 9.5 ⚠️ A known imprecision in my own manuscript

The paper's Discussion says: *"A single epoch in AugMix seed 23 took 865.9 seconds while most late
epochs took 80–110 seconds."* That statement is accurate **for seed 23 only**. For **seed 17**, the
late epochs (41–50) actually ran at **~310–325 s each**, not 80–110 s. So "most late epochs" is
true for one seed and false for the other.

The conclusion of the paper is unaffected — variability is the point, and the claim is still
"timings are unstable" — but if a professor reads the logs, I must be able to explain this precisely
and correct it rather than defend it. This is exactly the kind of thing I should raise *before* they
find it.

---

## 10. Tools and software I used

| Tool | Version / detail | What it did |
|---|---|---|
| **PyTorch** | 2.14.0+cu130 | Tensors, GPU execution, autograd, `nn.Module`, optimisers, schedulers, CUDA memory stats |
| **torchvision** | 0.29.0+cu130 | `models.mobilenet_v3_small`, `transforms.RandAugment`, `transforms.AugMix`, `datasets.CIFAR10`, `torch.utils.data.DataLoader` |
| **NumPy** | 2.5.2 | Loading CIFAR-10-C `.npy` arrays, computing means |
| **Pillow** | 12.3.0 | Underlying image type for torchvision's PIL transforms (the augmentation CPU cost) |
| **CUDA** | cu130 runtime | GPU execution on the GTX 1650 |
| **Python** | 3.14.5 | Everything; `argparse`, `csv`, `json`, `time.perf_counter`, `statistics`, `pathlib` |
| **Windows 11** | 10.0.26200-SP0 | Host OS |
| **PowerShell** | Windows PowerShell 5.1 | Running the scripts |
| **Zenodo** | — | Public preprint + reproducibility archive, DOI minting, CC BY 4.0 |
| **Local `.venv`** | — | Isolated environment so the project did not pollute the system Python |
| **Markdown → HTML → PDF** | — | Rendering `paper_draft.md` to `paper_draft.pdf` |

**What I deliberately did *not* use:** no pretrained weights, no Albumentations, no TensorBoard or
W&B, no Optuna, no mixed precision, no PyTorch Lightning, no `torch.compile`, no Hugging Face
datasets, no Jupyter. Every dependency is either in the standard PyTorch stack or the Python
standard library — which is exactly why the reproducibility archive is only 34.5 MB and anyone can
re-run it.

---

## 11. Exactly how I used each tool

### 11.1 PyTorch

- `nn.Module` composition for the network; I surgically edited two submodules of a pretrained
  *architecture* (not weights): `model.features[0][0].stride = (1,1)` and
  `model.classifier[-1] = nn.Linear(in_features, 10)`.
- `torch.manual_seed` / `torch.cuda.manual_seed_all` for reproducibility;
  `torch.backends.cudnn.deterministic = True`, `benchmark = False` — the correct pairing, since
  `benchmark=True` would pick algorithms by timing, destroying run-to-run reproducibility.
- `torch.cat(views, dim=0)` to push AugMix's three 128-image views through **one** 384-image forward
  pass, then `logits.split(batch_n, dim=0)` to split them back. This is an efficiency detail worth
  mentioning: it avoids three separate kernel-launch sequences and better utilises the GPU.
- `F.cross_entropy`, `F.log_softmax`, `F.kl_div` for the loss and the JSD.
- `torch.inference_mode()` (not just `no_grad`) for evaluation — slightly faster and stricter.
- `non_blocking=True` with `pin_memory=True` on the DataLoader for overlapping host→device copies.
- `optimizer.zero_grad(set_to_none=True)` rather than `zero_grad()` — avoids a wasted memset.
- `torch.cuda.max_memory_allocated` / `reset_peak_memory_stats` for the memory metric.
- `torch.cuda.synchronize()` before stopping the timers, so asynchronous GPU work is not excluded
  from the measurement. **Getting this wrong is the single most common timing bug in DL benchmarking.**

### 11.2 torchvision

- `models.mobilenet_v3_small(weights=None)` — architecture with **random** initialisation. Passing
  `weights=None` explicitly is what guarantees no pretrained weights leak in.
- `transforms.RandAugment(num_ops=2, magnitude=9)` and `transforms.AugMix(severity=3, mixture_width=3,
  chain_depth=-1, alpha=1.0)` — the published reference implementations, at their default
  hyperparameters. Using the library implementation rather than my own means my "method" is exactly
  the published method, which is a strong internal-validity argument.
- `transforms.RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize()`.
- `datasets.CIFAR10` with `download=True` for the first run and `download=False` for evaluation (so
  evaluation can never silently re-download anything).
- `torch.Generator().manual_seed(seed)` passed to the DataLoader so the shuffling order is
  reproducible and independent of global RNG state.
- `worker_init_fn` seeding `numpy` and `random` per worker — without this, every worker would inherit
  the same default RNG state and produce correlated augmentations across workers, a subtle and real
  bug.
- `persistent_workers=True` so worker processes are not re-created every epoch.

### 11.3 NumPy

- `np.load(path, mmap_mode="r")` to memory-map the CIFAR-10-C arrays. Each corruption file is
  50,000 × 32 × 32 × 3 uint8 ≈ 150 MB; memory-mapping means I never load 750 MB into RAM at once, and
  I slice one severity block at a time.
- `np.mean` for the unweighted CIFAR-10-C aggregate.
- `np.random.seed` in the seeding function.

### 11.4 Pillow

Used implicitly, and it is worth understanding because it explains the timing. torchvision's
transforms take PIL images; `AugMix` internally converts PIL→tensor→PIL for each operation so it can
use tensor ops like `posterize` and `equalize`. On a 32×32 image these conversions are individually
cheap, but AugMix performs **2 AugMix calls × 3 chains × 1–3 ops** per source image, i.e. **6–18
operations plus round-trips**, in Python, on the CPU, across only 2 worker processes. That is a
plausible bottleneck — see §9.4.

### 11.5 Zenodo

Published `paper_draft.pdf` and `zenodo_reproducibility.zip` as Version 1.0, resource type Preprint,
CC BY 4.0, with a reserved DOI. Datasets deliberately **excluded** (CIFAR-10 and CIFAR-10-C terms do
not permit redistribution); I published download instructions and the MD5 checksum instead. This is
the correct way to handle dataset licensing, and I can explain why.

---

## 12. The code, file by file

### `src/train.py` (235 lines) — one file, three conditions

The design idea: **one script, parameterised by `--method`**, so the three conditions are guaranteed
to share every line of code except the augmentation branch. This is the strongest structural defence
of the internal validity of the comparison — there is no possibility that I accidentally gave one
condition a different optimiser, because there is only one code path.

- `seed_everything(seed)` — seeds Python, NumPy, Torch, CUDA; sets cuDNN deterministic.
- `seed_worker(worker_id)` — per-worker seeding.
- `make_model()` — builds MobileNetV3-Small, applies the two CIFAR adaptations, returns it.
- `class AugmentedCIFAR(Dataset)` — the heart. `__getitem__` returns:
  - baseline: one normalised tensor
  - randaugment: one normalised tensor
  - augmix: a **3-tuple** of normalised tensors
- `jensen_shannon(logits_a, logits_b, logits_c)` — the consistency loss, implemented as
  `mean over the 3 views of KL(p_i ‖ M)`, with a `1e-7` clamp on `log M` for numerical safety.
- `system_manifest(device)` — auto-records Python, platform, Torch, torchvision, NumPy versions, GPU
  name, and total GPU memory into `config.json`. **Automation of provenance** — I did not have to
  remember to write it down.
- `main()` — argument parsing, CUDA enforcement, DataLoader construction, training loop with
  per-epoch CSV logging (appended and flushed every epoch, so a crash still leaves usable logs),
  final checkpoint save, summary JSON.
- Only two places branch on method: the dataset's `__getitem__` and the loss/view handling in the
  training loop. Everything else is shared.

### `src/evaluate.py` (145 lines) — held-out evaluation

- Imports `MEAN`, `STD`, and `make_model` **from `train.py`**, guaranteeing the evaluator uses
  byte-identical preprocessing and architecture. This prevents a whole class of silent bugs where a
  normalisation constant differs between training and evaluation.
- `CANONICAL_CORRUPTIONS` — the explicit set of 15. Raises `FileNotFoundError` listing any missing
  file, and selects only those 15, explicitly documenting that some archive versions include four
  extra corruptions.
- `evaluate_batches` — slices the memory-mapped array per severity, converts `uint8 → /255.0 →
  normalize` (numerically equivalent to `ToTensor()` + `Normalize()`, but written explicitly so the
  corruption path is auditable without torchvision's help), and computes top-1.
- `evaluate_clean` — uses `datasets.CIFAR10(train=False)` with the same transform.
- Verifies array shape: length must equal the labels length **and** be divisible by 5, which
  structurally guarantees five equal severity blocks.
- Requires CUDA and refuses CPU.
- Writes one CSV (all 77 rows: 1 clean + 75 corruption-severity + 1 aggregate) and one JSON per
  checkpoint.

### `src/summarize.py` (109 lines) — aggregation

- Joins `summary.json` and the evaluation JSON per run, groups by method.
- `mean_sd` uses `statistics.stdev` (sample SD, n−1) and returns `""` for a single value, so
  single-seed runs can never silently print a fake SD.
- Guards: skips any run directory missing `summary.json` or an evaluation file, so an incomplete
  matrix produces a partial report rather than a crash or a fabricated number.
- Derives the method name by `run_dir.name.rsplit("_seed", 1)[0]`, so naming is consistent.
- Writes the four analysis CSVs. **Every table in the paper comes from here.**

### Supporting files

| File | Purpose |
|---|---|
| `pilot_protocol.md` | Frozen protocol + dated deviation log (7 entries) |
| `plan.md` | Full 24-week research plan, risk table, literature method, completion criteria |
| `literature_review.md` | Source matrix, novelty assessment, design implications, search log |
| `progress.md` | Phase tracker, research-decision log, per-run experiment log |
| `memory.md` | Context summary for resuming the project |
| `README.md` | Reproduction instructions |
| `zenodo_metadata.md`, `zenodo_checklist.md` | Release metadata and pre-publication checklist |
| `zenodo_package/` | The exact released contents |
| `results/final/<run>/` | `config.json`, `epochs.csv`, `model.pt`, `summary.json` per run |
| `results/evaluation/` | Per-run CSV + JSON (clean and all 75 corruption-severity cells) |
| `results/analysis/` | `condition_summary.csv`, `cifar10c_by_corruption.csv`, `cifar10c_by_type.csv`, `cifar10c_by_severity.csv` |
| `data/` | CIFAR-10 and CIFAR-10-C, local only, never redistributed |

---

## 13. Reproducibility and data provenance

**Data acquisition, with checksums:**

| Dataset | Source | Verification | Date |
|---|---|---|---|
| CIFAR-10 | downloaded automatically by `torchvision.datasets.CIFAR10` | torchvision's own checksum verification | 2026-09-25 |
| CIFAR-10-C | Zenodo record `https://zenodo.org/records/2535967` | **MD5 `56bf5dcef84df0e2308c6dcbcbbd8499` verified against the Zenodo record** | 2026-09-25 |

**Reproducibility measures actually implemented:**

1. Seeded Python, NumPy, Torch, CUDA; cuDNN deterministic with benchmark off.
2. A dedicated `torch.Generator` for the DataLoader's shuffling.
3. Per-worker RNG seeding to avoid correlated augmentations.
4. Full environment manifest auto-written into every `config.json`.
5. Per-epoch logs flushed immediately, so a crash loses nothing.
6. Every reported number traceable to a saved file; aggregation is scripted, never hand-edited.
7. A dated deviation log recording every anomaly and judgement call.
8. The complete reproducibility archive published alongside the paper.

**Honest caveats about reproducibility:**

- Bitwise-identical results are guaranteed only on the same GPU model, driver, and library versions.
  cuDNN may select different algorithms on different hardware.
- `torch.use_deterministic_algorithms(True)` was **not** enabled, and the absolute wall times are
  **not** reproducible at all — that is the central empirical finding.
- I did **not** formally record a repeat-evaluation check on the same checkpoint, even though it
  should be deterministic (no random transforms, `inference_mode`, `model.eval()`).
- Datasets are not redistributed; a user must download them and verify the checksum.

---

## 14. Honest limitations

I keep these in a single list because being caught hiding one is far worse than volunteering it.

1. **Two seeds.** With n = 2 the sample SD is barely meaningful and no hypothesis test is possible.
   All statistics are descriptive. This is the biggest limitation.
2. **One architecture.** MobileNetV3-Small only. No evidence the ranking transfers to ResNorms,
   WideResNets, or vision transformers.
3. **One dataset, low resolution, synthetic corruptions.** 32×32 CIFAR-10 and code-generated
   corruptions say nothing about real photographs or real distribution shift.
4. **50 epochs is probably not converged.** Absolute accuracies are lower than published CIFAR-10
   numbers. Rankings at 50 epochs need not hold at convergence.
5. **Timing is unreliable and hardware-specific.** §9 in full. It is one GTX 1650, one software
   stack, 2 workers, one implementation, and it varied by 2× between identical runs.
6. **No compute-matched control.** AugMix received 3× the model forward/backward work. The
   comparison is *policy-matched*, not *compute-matched*. Part of AugMix's advantage may be extra
   optimisation rather than better augmentation.
7. **Hyperparameters are library defaults, not tuned.** This is a fair, leakage-free design, but it
   means the comparison is of *default-configured* methods, not best-tuned ones. A tuned RandAugment
   might close some of the gap.
8. **No validation split in the pilot.** All 50,000 training images were used for training. The
   original plan proposed a 45,000/5,000 split; the pilot omitted it to fit the deadline. Because I
   used the final-epoch checkpoint and pre-set hyperparameters, there is no test leakage — but there
   is also no early stopping or model selection of any kind.
9. **The stride-1 stem makes this model unusually expensive for its size**, which inflates all
   absolute times and interacts with the timing comparison. A stride-2 stem (accepting the 16×16
   first feature map) would be much faster and is the natural follow-up.
10. **Timing variance exceeds some of the effects being measured.** For RandAugment, the ±586 s SD is
    larger than the +351 s mean difference from the baseline. That comparison is not interpretable.
11. **Accuracy only.** No calibration, ECE, NLL, or uncertainty — despite uncertainty being a stated
    motivation for AugMix. No mCE, no FLOPs, no energy, no inference latency.
12. **No inference latency measured**, and correctly so: with an identical architecture and identical
    inference preprocessing, training-time augmentation *cannot* change inference latency. Reporting
    it as an augmentation benefit would be a category error.
13. **No transfer check.** CIFAR-10.1 (a separately collected test set) was planned but not run. This
    is the most important missing generalisation test, especially given Mintun et al. (2021).
14. **Not peer-reviewed.** The Zenodo DOI is a preprint DOI. A Zenodo DOI identifies an archived
    report; it is not a quality signal.
15. **The literature scan is not exhaustive.** I read and matrixed 9 primary sources but did not
    complete a systematic review.
16. **Known imprecision in my own Discussion** about "most late epochs" (§9.5).
17. **Human PDF visual proof still open.** Local renderers and Zenodo's viewer were unavailable, so
    the author had not confirmed the PDF's layout in the Zenodo preview.

---

## 15. Things I would do next

Ranked by how much they would strengthen the work.

1. **Add a compute-matched control.** Train the baseline for 3× the epochs (or at 3× the batch
   throughput) at the same wall-clock budget. This is the only experiment that separates *"AugMix's
   augmentation is better"* from *"AugMix got 3× the gradient signal."* Without it, the central
   claim stays confounded. **This is the most important follow-up.**
2. **Fix the timing methodology.** Report the **median** epoch time rather than the total; discard
   the first epoch as warm-up; run each condition 3×; fix the Windows power plan to "maximum
   performance"; close background applications; record CPU/GPU utilisation and temperature
   automatically per epoch; and pre-declare the outlier rule.
3. **Profile the input pipeline separately.** Time the DataLoader with the model replaced by a no-op
   to get pure augmentation cost, then subtract. This directly tests my §9.4 hypothesis and is the
   evidence the current paper admits it lacks.
4. **Increase `num_workers`** from 2 to 4–8, and try `prefetch_factor`. If the pipeline is the
   bottleneck, this is a large, cheap win.
5. **Five or more seeds**, and a paired statistical test. Because seeds are shared across
   conditions, a **paired** analysis on per-seed differences is the right test and is much more
   powerful than the unpaired equivalent.
6. **Report mCE** alongside raw accuracy, since that is the benchmark's standard normalised metric.
7. **Tune each method properly** on a 45,000/5,000 train/validation split, then report both
   "default-configured" and "best-tuned" comparisons. That answers the obvious critique head-on.
8. **Add a second architecture** — a stride-2-stem MobileNetV3-Small and a small ResNet — to test
   whether the ranking is architecture-independent.
9. **Add CIFAR-10.1 v6** to test whether gains transfer to naturally collected data. Mintun et al.
   (2021) suggest they may not.
10. **Measure FLOPs, peak reserved memory, and energy** (e.g. via `torch.cuda` counters or
    `nvidia-smi` power draw) to give a hardware-independent cost metric alongside wall time.
11. **Complete the literature review** and, if a genuine gap emerges, reframe toward it. Do not
    submit to a venue until the gap statement is defensible.
12. **Report calibration metrics** (ECE, NLL) to test AugMix's uncertainty claim, which is one of its
    stated motivations and which I did not test at all.

---

## 16. What this project proves I can do

For a CV or master's application, these are the concrete, defensible claims — and every one is
backed by a file in this repository.

- **Frame a research question precisely and scope it honestly.** Including recognising when a
  contribution is *not* novel and repositioning accordingly, rather than overselling.
- **Read primary literature critically** and build a structured source matrix; detect a close
  overlapping 2026 paper and respond to it by narrowing claims instead of ignoring it.
- **Write a pre-registered, frozen experimental protocol** with data-separation rules, exclusion
  rules, and a dated deviation log.
- **Implement deep learning experiments correctly in PyTorch**, including multi-view consistency
  training, a custom JSD objective, deterministic seeding, and memory-mapped large-array evaluation.
- **Adapt a pretrained *architecture* to a new resolution** correctly, and reason about the compute
  consequences of that choice.
- **Run a controlled, matched-setting experiment** where one variable changes and nothing else does,
  enforced structurally in code.
- **Prevent data leakage by design**: no test-set tuning, no best-checkpoint selection, frozen
  hyperparameters, held-out evaluation performed once at the end.
- **Measure compute honestly**, including the subtle parts — CUDA synchronisation, the timing
  boundary, what memory metrics do and do not include, and reporting throughput per *source* image
  when a method expands images.
- **Analyse raw logs critically and find structure in noise** — the episodic timing-bimodality
  finding in §9 is genuine original analysis of my own data, and it contradicts my own paper's
  headline timing table in a useful way.
- **Report negative and inconvenient findings**: RandAugment losing to the baseline on pixelate,
  AugMix tying on brightness, my own Discussion sentence being imprecise.
- **Produce machine-readable, traceable results** where no published number was hand-typed.
- **Manage data licensing and provenance** — checksums, no redistribution, download instructions.
- **Publish a citable artefact** with a DOI, clear licensing (CC BY 4.0), and a reproducibility
  package.
- **Write in a register that does not overstate** — abstract, discussion, and limitations all
  explicitly refuse novelty and generalisation claims.

---

## 17. Flashcard sheet — every key number

### Accuracy (mean over seeds 17 & 23)

| Condition | Clean | CIFAR-10-C (mean of 75 cells) | Δ clean | Δ CIFAR-10-C |
|---|---:|---:|---:|---:|
| Crop-and-flip | **86.39 ± 0.25 %** | 67.04 ± 1.63 % | — | — |
| RandAugment | 86.18 ± 0.82 % | 71.00 ± 0.06 % | −0.21 pp | **+3.96 pp** |
| AugMix | 86.02 ± 0.01 % | **77.82 ± 0.31 %** | −0.37 pp | **+10.78 pp** |

### Cost

| Condition | Time mean ± SD | Minutes | Per-seed seconds | Source img/s | Peak CUDA allocated |
|---|---:|---:|---|---:|---:|
| Crop-and-flip | 793.1 ± 116.6 s | 13.2 | 710.7 / 875.6 | 3,186.5 | 205,514,240 B |
| RandAugment | 1,143.8 ± 586.4 s | 19.1 | 729.2 / 1,558.5 | 2,516.4 | 205,514,240 B |
| AugMix | 7,282.5 ± 1,379.1 s | 121.4 | 8,257.7 / 6,307.3 | 349.6 | 557,116,928 B |

### Median-epoch re-analysis (robust to machine interference)

| Condition | Median epoch | Implied 50-epoch | Overhead vs baseline |
|---|---:|---:|---:|
| Crop-and-flip | ~13.7 s | ~11.4 min | 1.00× |
| RandAugment | ~14.5 s | ~12.1 min | **~1.06×** |
| AugMix | ~90 s | ~75 min | **~6.6×** |

### Per-severity (mean over 15 corruptions × 2 seeds)

| Severity | 1 | 2 | 3 | 4 | 5 | Total drop |
|---|---:|---:|---:|---:|---:|---:|
| Crop-and-flip | 79.77 | 74.22 | 68.17 | 60.89 | 52.17 | −27.60 |
| RandAugment | 81.54 | 77.44 | 72.70 | 66.18 | 57.12 | −24.42 |
| AugMix | 83.29 | 81.29 | 79.10 | 75.77 | 69.67 | **−13.62** |

### Setup constants

| Item | Value |
|---|---|
| Architecture | torchvision MobileNetV3-Small, first conv stride 2→1, 10-class head, random init |
| Parameters | **1,528,106** |
| Train / test | 50,000 / 10,000 CIFAR-10 (native 32×32) |
| Corruption eval | 15 corruptions × 5 severities = **750,000** images per checkpoint; +10,000 clean = **760,000** |
| Total eval forwards | 6 × 760,000 = **4,560,000** |
| Epochs / batch / workers | 50 / 128 / 2 |
| Optimiser | SGD, lr 0.05, momentum 0.9, weight decay 5e-4, CosineAnnealingLR(T_max=50) |
| Seeds | 17, 23 (final); 99 (1-epoch feasibility pilot, excluded) |
| Augmentation params | RandAugment(num_ops=2, magnitude=9); AugMix(severity=3, mixture_width=3, chain_depth=−1, alpha=1.0), JSD weight 12 |
| Normalisation | mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616) |
| GPU | NVIDIA GeForce GTX 1650, 4,294,639,616 bytes (4 GB) |
| Software | Windows 11 · Python 3.14.5 · torch 2.14.0+cu130 · torchvision 0.29.0+cu130 · NumPy 2.5.2 · Pillow 12.3.0 |
| Total timed training | **18,438.9 s ≈ 5.12 h** (AugMix alone 14,565.0 s ≈ 4.05 h) |
| CIFAR-10-C MD5 | `56bf5dcef84df0e2308c6dcbcbbd8499` |
| Publication | Zenodo preprint v1.0, 2026-09-25, DOI `10.5281/zenodo.22961521`, CC BY 4.0 |
| Total | 3 conditions × 2 seeds × 50 epochs = **6 runs, 300 epochs, 0 failures** |

---

*Companion document: `qa.md` — a question-and-answer bank of the questions a professor is likely to
ask about this work, with prepared answers and the traps to avoid.*
