# Zenodo Record Metadata (Published)

**Status:** Published on 2026-09-25 as Zenodo record 22961521 (Version 1.0). Verified via the Zenodo API: status `published`, both files present with checksums matching the local copies. The author has reported supervisor/institution approval for public release.

## Basic information
- **Record URL:** https://zenodo.org/records/22961521
- **DOI:** 10.5281/zenodo.22961521
- **All-versions concept DOI:** 10.5281/zenodo.22961520
- **Title:** Training-Time Image Augmentation on MobileNetV3-Small: A Controlled CIFAR-10 Pilot
- **Resource type:** Publication — Preprint
- **Language:** English (`eng`)
- **Publication date:** 2026-09-25
- **Version:** 1.0

## Creator

- **Name (as published):** `RAIHAN, SIFAT` (family name Raihan, given names Sifat)
- **Role:** Researcher
- **Affiliation:** Department of Computer Science and Technology, Hebei University of Science and Technology, Shijiazhuang 050018, Hebei, China
- **ORCID:** Not provided; left blank. A name identifier can be added later without creating a new version.

## Description

Training-time image augmentation may improve classification under image corruption, but stronger policies can add computational cost and may change clean accuracy. This study compares random crop and horizontal flip, RandAugment, and AugMix using a CIFAR-adapted MobileNetV3-Small trained from scratch on CIFAR-10. Each condition was trained for 50 epochs with two seeds. We measured top-1 accuracy on the clean CIFAR-10 test set and the original 15 CIFAR-10-C corruption types at five severities, along with training wall time, source-image throughput, and peak allocated GPU memory on an NVIDIA GeForce GTX 1650 with 4 GB VRAM. Mean clean accuracies were 86.39% for crop-and-flip, 86.18% for RandAugment, and 86.02% for AugMix; mean CIFAR-10-C accuracies were 67.04%, 71.00%, and 77.82%, respectively. Mean training times were 13.2, 19.1, and 121.4 minutes. Run times varied substantially between seeds, particularly for RandAugment and AugMix. With two seeds, one architecture, one dataset, and one local system, these results are exploratory and do not support broad generalization or a novelty claim. This is an exploratory report and has not undergone journal or conference peer review.

## Keywords

CIFAR-10-C; image augmentation; corruption robustness; MobileNetV3-Small; RandAugment; AugMix; training cost

## Related work/data identifiers

- CIFAR-10-C dataset record: `https://doi.org/10.5281/zenodo.2535967` (DOI `10.5281/zenodo.2535967`).
- The deposit form exposed only *Alternate identifiers* and no *Related identifiers* field, so the dataset DOI was **not** recorded in the published metadata. It is cited inside `paper_draft.pdf` and named in the description and keywords. Do not move it into Alternate identifiers, which would misstate it as an identifier for this record.

## Published files

1. `paper_draft.pdf` — 175,183 bytes, MD5 `d3f990c9466e4a6de08d6abc941e79ee`
2. `zenodo_reproducibility.zip` — 34,477,815 bytes, MD5 `a5afbac043bce4100f144693f8433d1d`, 45 content files: scripts, frozen protocol, configurations, logs, checkpoints, evaluation outputs, and aggregate tables. Excludes datasets, planning notes, and manuscript source variants.

Both published checksums match the local files exactly. Files can only be added, removed, or modified by the author within 45 days of publication (i.e., until 2026-11-09); metadata stays editable indefinitely.

## Rights / license

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), as published.
- **Copyright:** Copyright (c) 2026 Sifat Raihan
- **Access:** Open (files public).
- **Release rights:** The author reports that supervisor/institution approval for public release has been obtained.
- **Scope:** This license applies to the paper and reproducibility archive uploaded with this record. The datasets are excluded and remain subject to their source terms.
