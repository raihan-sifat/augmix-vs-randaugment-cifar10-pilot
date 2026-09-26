# Project Memory for Resuming Later

This file summarizes the project context so a future assistant session can continue without restarting the planning work. The user can edit it as decisions change.

## User and goal

- The user is completing a bachelor's degree in Computer Science and Technology and expects to finish next year.
- They plan to apply for a master's degree, with interests in machine learning, AI, image processing, data engineering, and software engineering.
- They selected a bachelor-level research direction around image augmentation and robustness of lightweight image classifiers.
- The goal is to complete a research paper and explore an appropriate publication route; publication is not guaranteed.

## Agreed working topic

**How well do image augmentation methods improve the robustness of lightweight image classifiers, and what is the cost in accuracy and computing time?**

Proposed central question:

> Pilot question: Under a fixed short training schedule on CIFAR-10, how do crop-and-flip, RandAugment, and AugMix compare in clean accuracy, CIFAR-10-C accuracy, and measured training time for MobileNetV3-Small?

## Current plan in `plan.md`

- Core experiment compares basic augmentation (random crop/flip), RandAugment, and AugMix.
- One lightweight classifier is the core scope; MobileNetV3-Small adapted for 32x32 is one candidate. A CIFAR-adapted ResNet-18 is an optional second-model extension only if time/resources allow.
- Train using a fixed stratified split from CIFAR-10 training data; use the standard CIFAR-10 test split, CIFAR-10-C, and CIFAR-10.1 v6 for final evaluation only.
- Do not train on or tune hyperparameters against the final test sets.
- Proposed measurements: clean accuracy; CIFAR-10-C accuracy by corruption and severity; CIFAR-10.1 accuracy; training wall-clock time; inference latency/throughput; parameter count; repeat across seeds where feasible.
- The user's two-day deliverable is complete as an exploratory draft with original results. It is not publication-ready: the literature scan is not exhaustive, the study uses one model/dataset and two seeds, timing was unstable, and venue review remains. The full 24-week plan remains for later extension.
- No novelty claim: a 2026 study already evaluates MobileNetV3-Small on CIFAR-10/CIFAR-10-C with an augmentation and training-time measurements.

## Important cautions

- The initial method comparison may already be well covered. Search recent primary literature and revise the contribution if needed before running the full study.
- CIFAR benchmarks are useful controlled tests but do not establish performance on every real-world image domain.
- Report results honestly, including mixed or negative findings; don't tune on test results or hide failed runs.
- Confirm dataset terms, university requirements, compute resources, supervisor expectations, and venue details when known.

## Files already created

- `plan.md`: full end-to-end plan, research design, timeline, writing structure, risk table, and completion criteria.
- `progress.md`: phase tracker, next actions, research decisions, experiment log, and submission checklist.
- `memory.md`: this resume summary.
- `literature_review.md`: primary-source matrix including the close 2026 DistortionMix comparison.
- `pilot_protocol.md`: frozen short-deadline experimental design and deviation log.
- `src/train.py`, `src/evaluate.py`, `src/summarize.py`: training, evaluation, and aggregation scripts.
- `paper_draft.md`: first complete experimental paper draft with results populated and the author's name and institutional affiliation added; verify sources and request supervisor review.
- `paper_draft.pdf`: reader copy. Two versions now exist and this is intentional: the 175,183-byte copy inside Zenodo v1, and the current local 179,987-byte copy that carries the AI disclosure for SSRN. The editable Markdown and browser-readable HTML remain available locally.
- `zenodo_reproducibility.zip`: published lean archive with scripts, frozen protocol, configurations, logs, checkpoints, evaluation outputs, and aggregate tables; excludes CIFAR datasets, planning notes, and manuscript source variants.
- `zenodo_metadata.md`: as-published record metadata.
- `zenodo_checklist.md`: release tasks, completion statuses, and published record details.
- `results/final/`: six 50-epoch model checkpoints, configs, epoch logs, and training summaries.
- `results/evaluation/`: six clean/CIFAR-10-C per-run CSV and JSON outputs.
- `results/analysis/`: condition summary, by-corruption, by-type, and by-severity CSVs.
- `README.md`: local reproduction instructions.

## Current status as of 2026-09-25

- User confirmed the experimental-paper deliverable is due within two days; GPU is on their computer (NVIDIA GTX 1650, 4 GB VRAM).
- Project-local `.venv` is installed: Python 3.14.5, torch 2.14.0+cu130, torchvision 0.29.0+cu130. CUDA and the GTX 1650 were verified.
- A primary-literature scan found direct overlap: Fadhilah et al. (2026), DOI 10.21609/jiki.v19i1.1558, tests MobileNetV3-Small with CIFAR-10/CIFAR-10-C, augmentation, and 50-epoch training-time measurements. Treat the current work as a controlled, replication-style pilot, not a novel contribution.
- Pilot protocol: MobileNetV3-Small adapted for 32x32; crop-and-flip vs RandAugment vs AugMix (two AugMix views plus clean view and JSD); 50 epochs, batch 128, workers 2, seeds 17 and 23. All six runs and all held-out evaluations are complete. CIFAR-10-C used the canonical original 15 corruptions at severities 1-5. The checksum-verified archive MD5 is `56bf5dcef84df0e2308c6dcbcbbd8499`.
- Environment: Windows 11, Python 3.14.5, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, NumPy 2.5.2, NVIDIA GeForce GTX 1650 with 4 GB VRAM.
- Two-seed means (clean accuracy / CIFAR-10-C mean / time): crop-and-flip 86.39% / 67.04% / 793.1 s; RandAugment 86.18% / 71.00% / 1,143.8 s; AugMix 86.02% / 77.82% / 7,282.5 s. Peak allocated GPU memory was 205,514,240 bytes for crop-and-flip and RandAugment, and 557,116,928 bytes for AugMix. These descriptive two-seed values do not support inferential claims.
- Per-run times in seconds: baseline seed17 710.7; baseline seed23 875.6; RandAugment seed17 729.2; RandAugment seed23 1,558.5; AugMix seed17 8,257.7; AugMix seed23 6,307.3. Timing varied widely. AugMix seed23 had an 865.9-second epoch 45; RandAugment seed23 took 2.14x as long as seed17. Treat timings as local measurements, not stable method-only costs.
- The paper draft has all experimental results, corruption/severity tables, limitations, and author/affiliation metadata. References [1]–[5] and reported aggregate values were checked against primary-source records and saved outputs; the literature search is not exhaustive. The close 2026 DistortionMix study overlaps directly, so make no novelty claim.
- Zenodo release is complete: https://zenodo.org/records/22961521, Version 1.0, published 2026-09-25, DOI 10.5281/zenodo.22961521 (concept DOI 10.5281/zenodo.22961520). Resource type Preprint, publisher Zenodo, language English, CC BY 4.0, copyright (c) 2026 Sifat Raihan, creator `RAIHAN, SIFAT` (Researcher) with the Hebei University of Science and Technology affiliation, seven keywords, and two open files (`paper_draft.pdf`, `zenodo_reproducibility.zip`) whose published MD5 checksums match the local copies. The author reported supervisor/institution approval. The deposit form had no related-identifier field, so the CIFAR-10-C DOI appears only in the paper. A Zenodo DOI identifies an archived preprint/report, not peer review. Only the human PDF visual proof is still open.
- Do not tune against test results or describe the exploratory draft as accepted or publication-ready.
- `progress.md` and `literature_review.md` were updated with the deadline, GPU, close prior work, and active status. `plan.md` contains the full long-term process plus the urgent protocol.

## SSRN / CompSciRN submission (in progress as of 2026-09-26)

The author wants to add this preprint to SSRN. Policy findings from the official Elsevier/SSRN sources, current as of 2026-09-26:

- **Favourable change:** SSRN has stopped posting final peer-reviewed articles in computer science and engineering and now focuses on "unpublished, high-quality preprints, and early-stage research" in those fields. This paper qualifies because it is a preprint, not a peer-reviewed article.
- SSRN runs a **Computer Science Research Network (CompSciRN)**, so an ML/CV paper is no longer a scope mismatch. Useful URLs: `https://www.elsevier.support/ssrn/answer/get-started`, `https://www.ssrn.com/index.cfm/en/compscirn/`, `https://www.ssrn.com/index.cfm/EN/compscirn/compscirn-research-paper-series/`.
- **AI disclosure is mandatory** and must appear both in the submission's abstract/disclosure field and on the PDF itself. SSRN states that undisclosed AI use leads to rejections and possible account closure. This applied here because Claude was used throughout the project.
- **SSRN has no appeals process** — it will not reconsider a rejected submission. Get the metadata right the first time.
- Form is six steps: Upload PDF, Confirm key details, Author information, Classify, Research integrity, Review and submit.
- Content Format = **Paper**; Content Type = **Preprint**. (Abstract-only submissions, books, calls for papers, and conference proceedings are rejected.)
- Date paper was written = **2026-09-25** (required field).
- Enter the existing Zenodo DOI `10.5281/zenodo.22961521` under Manuscript Identification. Because Content Type is Preprint, SSRN mints its own additional Preprint DOI (`10.2139/ssrn.XXXXXXX`), so the author will hold two DOIs; that is expected and fine.
- Keywords must be entered **one at a time with Enter after each**; pasting a comma-separated string does not work.
- Author self-classification is now required (minimum 1, maximum 7); SSRN removed the auto-classification option. Target the CompSciRN Artificial Intelligence and/or Applied Computing eJournals. JEL codes are optional.
- Research integrity: funding, conflict-of-interest, and (for human-subject work) ethics statements. Suggested wording — funding: "This research received no external funding."; conflict of interest: "The author declares no conflict of interest."; ethics: not required, since the study uses public image datasets and no human participants.
- SSRN takes **no copyright**; the author grants a non-exclusive distribution right and can withdraw at any time.
- Processing is usually quick but can take **up to 10 business days**.
- Because the manuscript is already public on Zenodo, the submission should state the Zenodo version and DOI so the two records are not confused. Dual-posting a preprint is acceptable, but the prior-publication wording should be checked against current SSRN terms at submission time.

### AI disclosure blocker — resolved 2026-09-26

A `## AI Disclosure` section was added to `paper_draft.md` at line 16, between the keywords and Section 1, and the HTML and PDF were regenerated. It names the four actual uses of Claude (drafting and editing the manuscript and planning/protocol/literature documents; writing and reviewing the released scripts; screening and summarising related work; cross-checking reported values against saved result files) and states that no experimental data, results, tables, figures, or citations were generated or altered, that every reported value came from the released scripts and saved outputs, and that the author takes full responsibility. **The author must confirm this wording is still accurate before publishing, since it becomes public.**

Text for the SSRN form field (single paragraph):

> **AI Disclosure.** AI tools (Anthropic's Claude) were used to assist with drafting and editing this manuscript and its accompanying documents, writing and reviewing the released training, evaluation, and aggregation scripts, screening and summarising related work, and cross-checking reported values against saved result files. AI tools were not used to generate, fabricate, alter, or interpolate any experimental data, results, tables, figures, or citations. Every quantitative value was produced by the released scripts from the saved run outputs and was independently verified by the author, who designed the protocol, ran all six training runs and all held-out evaluations, and takes full responsibility for the work.

If the form has no dedicated AI field, append this after the abstract text in the Step 2 metadata box. Keep the abstract itself verbatim so the form and the PDF match, since SSRN extracts the abstract from the PDF.

### Rendering and verification notes

- Local PDF rendering **is** possible, contrary to the earlier note: `& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="<out.pdf>" "file:///D:/Apply/research_paper/paper_draft.html"`. The benign `fallback_task_provider.cc` stderr warning can be ignored. Regenerate the HTML first with `src/render_paper.py`.
- The venv has no PDF library (no pypdf/pdfminer/fitz), and the assistant cannot view PDFs. Text was verified instead with a dependency-free script at `C:\Users\RUGEN\AppData\Local\Temp\opencode\verify_pdf_text.py`, which decodes the PDF's embedded `ToUnicode` CMaps and reconstructs the text. Run it as `verify_pdf_text.py <pdf> "<phrase>"` to check any phrase after future edits. Note that it concatenates words split across line breaks, so multi-word phrases can appear missing when they are actually present.

### Decision: Zenodo v1 left unchanged

The author decided **not** to publish a Zenodo v2. Zenodo imposes no AI-disclosure requirement, so v1 remains valid as published, and leaving it alone avoids a new version DOI plus re-verifying the published archive checksums. Consequence: `zenodo_package/paper_draft.md` (16,773 bytes) and the published 175,183-byte `paper_draft.pdf` deliberately differ from the current local files. **This is intentional, not drift — do not "fix" it.** The v1 archive checksums recorded in `progress.md` remain authoritative.

### Still outstanding for SSRN

1. The author must visually inspect the regenerated `paper_draft.pdf`; text extraction confirms content and ordering but not pagination.
2. The disclosure paragraph must be pasted into the SSRN form.
3. arXiv `cs.CV` / `cs.LG` was recommended as a cheap complement (free, no screening, real ML readership, roughly ten minutes to post) and offered to the author, but **has not been started**. The author has not yet decided.

## Resume instructions

The six-run pilot, the first paper draft, and the Zenodo preprint release are complete (record 22961521). The mandatory AI disclosure for an SSRN/CompSciRN submission has been added to `paper_draft.md`, `paper_draft.html`, and `paper_draft.pdf`, and verified by text extraction. On return, read `progress.md`, this file, and `paper_draft.md`; then check `zenodo_checklist.md` and `zenodo_metadata.md` if Zenodo matters. Immediate next step is the SSRN submission itself: the author visually checks the 179,987-byte PDF, pastes the disclosure text from this file into the form, and submits following the six-step sequence above. Do not create a Zenodo v2 — that was explicitly declined on 2026-09-26; Zenodo files can also only be modified within 45 days of publication. Optionally add an ORCID and the CIFAR-10-C dataset DOI to the Zenodo description as metadata-only edits. Keep the two-seed, one-model, synthetic-benchmark, literature-overlap, and timing-variability limits explicit, and do not describe the work as peer reviewed — neither a Zenodo DOI nor an SSRN DOI represents peer review. Add the author's contact email if a later selected journal requires it.

