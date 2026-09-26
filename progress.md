# Research Progress Tracker

**Project:** Robustness–efficiency trade-offs of image augmentation for lightweight image classifiers  
**Last updated:** 2026-09-26  
**Overall status:** Pilot, held-out evaluation, and the Zenodo preprint release are complete (record 22961521, DOI 10.5281/zenodo.22961521); the mandatory AI disclosure required for an SSRN/CompSciRN submission has been added to the manuscript and PDF; only the human PDF visual proof remains open
**Main plan:** [plan.md](plan.md)  
**Frozen pilot protocol:** [pilot_protocol.md](pilot_protocol.md)
**Zenodo pre-publication checklist:** [zenodo_checklist.md](zenodo_checklist.md)

## Current position

- [x] Selected a research direction: image augmentation and corruption robustness for lightweight classifiers.
- [x] Created the full research plan and proposed 24-week schedule in `plan.md`.
- [x] Began a first-pass scan of primary papers on corruption benchmarks and augmentation trade-offs.
- [x] Created `literature_review.md` with an initial source matrix and review questions.
- [x] Refined the provisional question so training cost is the direct compute outcome.
- [x] Confirmed an experimental paper with original results is due within two days.
- [x] Confirmed NVIDIA GTX 1650 with 4 GB VRAM; project-local PyTorch CUDA environment installed and GPU use verified.
- [x] Narrowed the urgent deliverable to a three-condition MobileNetV3-Small pilot; literature scan found close overlap, so no novelty claim will be made.
- [x] Drafted a reproducible training script and frozen pilot protocol.
- [x] CIFAR-10 downloaded; one-epoch feasibility runs completed for all three conditions; 50 epochs and batch size 128 frozen before test evaluation.
- [x] Official CIFAR-10-C archive downloaded, checksum verified, and extracted after timed training.
- [x] Complete six matched 50-epoch runs and evaluate CIFAR-10 and all 15 canonical CIFAR-10-C corruptions at five severities.
- [x] Populate the experimental paper draft from saved outputs; report timing variability and study limits.
- [x] Add the author's name and institutional affiliation to the manuscript.
- [x] Cross-check manuscript references and reported numbers; prepare the PDF, reproducibility archive, and Zenodo metadata draft.
- [x] Publish the preprint and reproducibility archive on Zenodo with a reserved DOI and CC BY 4.0.
- [ ] Complete a supervisor and venue-specific review before submission.

## Milestones

| Phase | Status | Completion date | Notes / evidence |
|---|---|---|---|
| A. Define project scope | Complete for urgent pilot | 2026-09-25 | Two-day deadline; experimental paper with original results; GPU confirmed. |
| B. Literature review and gap | Initial scan complete; no verified novelty | 2026-09-25 | Source matrix includes close 2026 MobileNetV3-Small/CIFAR-C/augmentation/runtime overlap. |
| C. Experimental protocol | Complete for pilot | 2026-09-25 | Frozen settings in `pilot_protocol.md`; MobileNetV3-Small, three augmentation conditions, 50 epochs, seeds 17 and 23. |
| D. Data/software pipeline | Complete for pilot | 2026-09-25 | `.venv` has torch 2.14.0+cu130 and torchvision 0.29.0+cu130; GPU verified. CIFAR-10 and CIFAR-10-C archive are available; archive MD5 verified. |
| E. Pilot and settings freeze | Complete | 2026-09-25 | All three one-epoch feasibility runs complete; final schedule and settings frozen. |
| F. Core experiments | Complete | 2026-09-25 | All six planned 50-epoch runs completed; all checkpoints and logs saved. |
| G. Evaluation and analysis | Complete | 2026-09-25 | All six checkpoints evaluated on clean CIFAR-10 and all 75 corruption-severity pairs; analysis CSVs generated. |
| H. Interpretation | Complete for pilot | 2026-09-25 | AugMix had the highest CIFAR-10-C mean; crop-and-flip had the highest clean mean. Training-time variation is large and limits cost conclusions. |
| I. Manuscript writing | Complete for first draft | 2026-09-25 | `paper_draft.md` has methods, all seed-level results, corruption/severity tables, discussion, limitations, and the author's affiliation. |
| J. Review and revision | Pending | — | Check references and format with the supervisor and selected venue; no publication venue has been selected. |
| K. Venue selection and submission | Not started | — | Decide with supervisor after the contribution and paper are clear. |
| L. Review response and archive | Not started | — | Pending submission. |

## Next actions

1. Visually inspect the regenerated `paper_draft.pdf` (179,987 bytes, 2026-09-26) and paste the AI disclosure into the SSRN submission form. Local PDF regeneration is now possible via headless Edge, and text extraction confirms the disclosure is present, but only the author can confirm the visual layout.
2. Optionally add an ORCID to the creator and mention the CIFAR-10-C dataset DOI in the record description; both are metadata-only edits, not a new version.
3. If pursuing a journal or conference later, expand the literature review and apply its template/email requirements.

## Research decisions

| Date | Decision | Reason / follow-up |
|---|---|---|
| 2026-09-25 | Focus on image augmentation for robust, lightweight image classification. | Chosen to align primarily with the user's stated interests in ML/AI and image processing. |
| 2026-09-25 | Proposed core comparison: basic augmentation, RandAugment, and AugMix. | Starting design in `plan.md`; confirm against literature, compute, and supervisor feedback before freezing. |
| 2026-09-25 | Proposed evaluation: CIFAR-10 clean test, CIFAR-10-C, and CIFAR-10.1 v6. | Enables clean, corruption, and separately collected test-set evaluation; do not train or tune on these test sets. |
| 2026-09-25 | Urgent pilot uses MobileNetV3-Small and baseline/RandAugment/AugMix, with seeds 17 and 23. | Two-day deadline; keep scope executable and label it exploratory. |
| 2026-09-25 | Make no novelty claim for the broad augmentation/robustness/runtime question. | A 2026 study already evaluates MobileNetV3-Small on CIFAR-10/CIFAR-10-C with augmentation and training-time measurements. |
| 2026-09-25 | Six runs, clean/CIFAR-10-C evaluation, analysis tables, and the first full paper draft are complete. | Windows 11, Python 3.14.5, torch 2.14.0+cu130, torchvision 0.29.0+cu130; GTX 1650, 4 GB VRAM. |

## Initial literature screening (not a complete review)

- The general area is established: prior primary work evaluates augmentation methods on corruption benchmarks and reports that gains can vary by corruption type.
- AugMix is an established method with prior CIFAR-10-C evaluation, so merely comparing it to a basic baseline is not by itself a novelty claim.
- A NeurIPS 2021 study examines interactions between augmentations and test corruptions; NeurIPS 2024 work studies a different robustness approach. These are starting points for reading, not proof that the proposed lightweight-model/compute question is already answered or remains open.
- A close 2026 paper was found in this combination; the broad novelty claim is unsupported. Keep the short-deadline work framed as a controlled pilot and continue a fuller literature review after the deadline.
- Sources: [NeurIPS 2021 augmentation-corruption study](https://proceedings.neurips.cc/paper/2021/hash/1d49780520898fe37f0cd6b41c5311bf-Abstract.html); [NeurIPS 2024 multiplicative weight perturbations paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3e9412a9c1d93810ef3ef7825115016b-Abstract-Conference.html).
- A more complete source matrix and focused reading tasks are in [literature_review.md](literature_review.md).
- Measurement clarification: training-time augmentation directly affects training/input-pipeline cost. Inference latency is not expected to vary from augmentation alone if architecture and inference preprocessing are unchanged.

## Experiment log

Add one entry per run or meaningful technical change. Keep full logs and raw outputs in the project repository rather than only here.

| Date | Run / change | Configuration or commit | Outcome / artifact | Follow-up |
|---|---|---|---|---|
| 2026-09-25 | One-epoch feasibility pilots: all three methods, seed 99. | `src/train.py`; `results/runtime_pilot/`. | Baseline 42.2s, RandAugment 20.9s, AugMix 89.9s; peak memory 205/205/555 MB. | Excluded from accuracy analysis; final schedule frozen at 50 epochs, batch 128, workers 2. |
| 2026-09-25 | Final baseline seed 17, 50 epochs. | `results/final/baseline_seed17/`. | 710.7s; checkpoint saved; final training accuracy 0.8877 (training metric only). | Held-out evaluation later: clean 86.21%, CIFAR-10-C mean 68.19%. |
| 2026-09-25 | Final RandAugment seed 17, 50 epochs. | `results/final/randaugment_seed17/`. | 729.2s; checkpoint saved; final training accuracy 0.8377 (training metric only). | Held-out evaluation later: clean 86.76%, CIFAR-10-C mean 70.95%. |
| 2026-09-25 | Final AugMix seed 17, 50 epochs. | `results/final/augmix_seed17/`. | 8,257.7s (137.6 min); 302.7 source examples/s; peak allocated memory 557,116,928 bytes; final training accuracy 0.8734 (training metric only). | Held-out evaluation later: clean 86.03%, CIFAR-10-C mean 78.04%. |
| 2026-09-25 | Final baseline seed 23, 50 epochs. | `results/final/baseline_seed23/`. | 875.6s (14.59 min); 2,855.2 source examples/s; peak allocated memory 205,514,240 bytes; final training accuracy 0.8877 (training metric only). | Held-out evaluation later: clean 86.57%, CIFAR-10-C mean 65.89%. |
| 2026-09-25 | Final RandAugment seed 23, 50 epochs. | `results/final/randaugment_seed23/`. | 1,558.5s (25.97 min); 1,604.1 source examples/s; peak allocated memory 205,514,240 bytes; final training accuracy 0.8264 (training metric only). | Held-out evaluation later: clean 85.60%, CIFAR-10-C mean 71.04%. |
| 2026-09-25 | Final AugMix seed 23, 50 epochs. | `results/final/augmix_seed23/`. | 6,307.3s (105.12 min); 396.4 source examples/s; peak allocated memory 557,116,928 bytes; final training accuracy 0.8763 (training metric only). Epoch 45 took 865.9s, while most other late epochs took 80-110s. | Held-out evaluation later: clean 86.01%, CIFAR-10-C mean 77.61%. |
| 2026-09-25 | Held-out evaluation and aggregation. | `src/evaluate.py`; `src/summarize.py`; `results/evaluation/`; `results/analysis/`. | Six checkpoints evaluated on clean CIFAR-10 and all 15 canonical CIFAR-10-C types at severities 1-5. Clean/CIFAR-C mean (%): crop-and-flip 86.39/67.04; RandAugment 86.18/71.00; AugMix 86.02/77.82. | Descriptive two-seed results; CIFAR-10-C is synthetic and timings showed substantial local variability. |
| 2026-09-25 | First full experimental paper draft and author block completed. | `paper_draft.md`. | Includes abstract, related work, protocol, per-seed accuracy/training times, severity/corruption tables, discussion, limitations, and institutional affiliation. | Verify sources and obtain supervisor review; no venue/novelty claim established. |
| 2026-09-25 | Zenodo pre-publication preparation. | `zenodo_checklist.md`, `zenodo_metadata.md`, `paper_draft.pdf`, `zenodo_reproducibility.zip`. | Local reference/result check completed; PDF and initial supplemental archive prepared. Dataset files are excluded. | PDF visual proof and upload remain; no Zenodo record has been created or published. |
| 2026-09-25 | Zenodo release decisions recorded. | `zenodo_metadata.md`, `zenodo_checklist.md`, `zenodo_package/README.md`, `memory.md`. | Author selected CC BY 4.0 and reported supervisor/institution approval; package README synchronized and archive rebuilt. New SHA-256: `4bc2541fc67e80a050aef745115698f74c5c83c4da95d794646b51d4a7db59df`. | Visually proof the PDF and upload the final record after the author's final instruction. |
| 2026-09-25 | Zenodo record published. | https://zenodo.org/records/22961521 | Version 1.0 published as a Preprint with DOI 10.5281/zenodo.22961521 (concept DOI 10.5281/zenodo.22961520), CC BY 4.0, copyright (c) 2026 Sifat Raihan, English, seven keywords, open access. Two files published: `paper_draft.pdf` (175,183 bytes) and `zenodo_reproducibility.zip` (34,477,815 bytes); both MD5 checksums match the local copies. | Human PDF visual proof still open; the form exposed no related-identifier field, so the CIFAR-10-C DOI is cited in the paper only. |
| 2026-09-26 | Mandatory AI disclosure added for the SSRN/CompSciRN submission; HTML and PDF regenerated. | `paper_draft.md`, `paper_draft.html`, `paper_draft.pdf`. | New `## AI Disclosure` section inserted between the keywords and Section 1. It names the four uses of Claude (drafting/editing, script writing/review, literature screening, value cross-checking) and states that no data, results, tables, figures, or citations were generated or altered, and that all reported values come from the released scripts and saved outputs. PDF rebuilt with headless Edge using `--no-pdf-header-footer`: 179,987 bytes, up from 175,183. Independent PDF text extraction confirms the section is present and intact. | The local `paper_draft.pdf` and `zenodo_package/paper_draft.md` now diverge from Zenodo v1, which published the 175,183-byte PDF without the disclosure. Decision (2026-09-26): Zenodo v1 is left unchanged and the disclosure-bearing PDF goes to SSRN/CompSciRN only. The same disclosure text must also be pasted into the SSRN submission form. |
| 2026-09-26 | Decided to leave the published Zenodo v1 record unchanged. | `progress.md`. | Zenodo imposes no AI-disclosure requirement, so v1 remains valid as published. The disclosure-bearing PDF (179,987 bytes) is reserved for the SSRN/CompSciRN submission. | The two records intentionally carry different PDF versions; do not "fix" this as drift. No Zenodo v2, no new DOI, and the v1 archive checksums recorded above stay authoritative. |

## Writing and submission checklist
- [ ] Research gap is supported by current literature.
- [x] Methods and results can be reproduced from documented settings and saved scripts.
- [x] Test sets were kept out of training and hyperparameter selection.
- [x] Results include both planned seeds and corruption types/severities.
- [x] Compute measurements include hardware and timing procedure.
- [x] Claims match the evidence and limitations are stated.
- [ ] Supervisor reviewed the final paper and venue choice.
- [ ] Dataset terms, attribution, university rules, and venue instructions were checked.
