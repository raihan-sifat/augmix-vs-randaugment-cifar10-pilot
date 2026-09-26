# Zenodo Pre-publication Checklist

**Record goal:** Public preprint / exploratory research report with a DOI. A Zenodo record is an archived research output, not evidence of journal or conference peer review.

**Current release status:** PUBLISHED. Zenodo record 22961521, Version 1.0, published 2026-09-25, DOI 10.5281/zenodo.22961521. Two files published and checksum-verified. Only the human PDF visual proof remains open.

## Tasks

- [x] **Finalize the manuscript text and references.** Check factual statements against primary sources, validate all numbers against saved results, and remove project-local-only references. The full literature search still needs supervisor review.
- [x] **Create a reader-ready PDF.** Preserve the author block, abstract, tables, references, and exploratory-report status. The source Markdown and browser-readable HTML are retained alongside it.
- [ ] **Visually proof the PDF.** Open it and check the title/author block, table widths, page breaks, and references. Automated rendering is unavailable locally and Zenodo's IIIF endpoint blocked this network, so the author must confirm the layout in the Zenodo record preview.
- [x] **Prepare reproducibility files.** Include the training/evaluation scripts, protocol, configurations, logs, aggregate and per-run results. Do not bundle the CIFAR datasets; provide their source and checksum instead.
- [x] **Prepare Zenodo metadata.** Saved in `zenodo_metadata.md` and published on the record.
- [x] **Choose a license and confirm release rights.** CC BY 4.0 applied to the record; the author reports that supervisor/institution approval has been obtained.
- [x] **Obtain supervisor review.** The author reports approval to release this as a preprint after checking the framing, related work, and result interpretation.
- [x] **Upload, preview, and publish.** Done. The published record was verified through the Zenodo API (status `published`, both files present, checksums match, CC BY 4.0, copyright statement, seven keywords, English, Preprint type).

## Work log

| Task | Status | Notes |
|---|---|---|
| Manuscript, references, and result consistency | Complete for local preflight | References [1]–[5] checked against publisher/primary-source records; reported means and corruption/severity tables match saved analysis CSVs. Literature scan is not exhaustive. |
| PDF | Published; visual proof still pending | `paper_draft.pdf` is a valid 3-page PDF and is published (175,183 bytes, MD5 `d3f990c9466e4a6de08d6abc941e79ee`). Author should still confirm the rendered layout in the record preview. |
| Reproducibility files | Published | `zenodo_reproducibility.zip` contains 45 content files: scripts, frozen protocol, six checkpoints/configs/logs, six evaluation CSV/JSON pairs, and aggregate CSVs. It excludes datasets, planning notes, and manuscript source variants. Published size 34,477,815 bytes, MD5 `a5afbac043bce4100f144693f8433d1d`, local SHA-256 `4bc2541fc67e80a050aef745115698f74c5c83c4da95d794646b51d4a7db59df`. |
| Zenodo metadata | Published | Creator name split corrected to `RAIHAN, SIFAT`; ORCID left blank; no related-identifier field was available, so the CIFAR-10-C DOI was not added to the record. |
| Supervisor / release approval | Complete per author | The author reports that supervisor/institution approval for public release has been obtained. |
| Zenodo upload and publication | Complete | https://zenodo.org/records/22961521 — Version 1.0, published 2026-09-25, DOI 10.5281/zenodo.22961521, concept DOI 10.5281/zenodo.22961520. |

## Current artifacts

- Published record: https://zenodo.org/records/22961521
- Reader copy: `paper_draft.pdf`
- Editable manuscript (local only; not uploaded): `paper_draft.md`
- Browser-readable copy (local only; not uploaded): `paper_draft.html`
- Reproducibility archive: `zenodo_reproducibility.zip`
- Metadata record: `zenodo_metadata.md`
- Reproducibility archive SHA-256: `4bc2541fc67e80a050aef745115698f74c5c83c4da95d794646b51d4a7db59df`

## Published upload set

- [x] `paper_draft.pdf` — reader-facing manuscript PDF, published.
- [x] `zenodo_reproducibility.zip` — lean archive, 45 content files, no dataset or planning files, published.
- [x] Zenodo preview verified — files, metadata, CC BY 4.0, copyright, DOI, keywords, and English language confirmed before publishing.
- [ ] PDF visual proof — confirm the title/author block, tables, page breaks, and references in the record preview.
- [ ] Optional follow-up: add an ORCID to the creator, and consider mentioning the CIFAR-10-C dataset DOI in the description text.

## Published Zenodo metadata

- **Record:** https://zenodo.org/records/22961521 (Version 1.0)
- **Title:** Training-Time Image Augmentation on MobileNetV3-Small: A Controlled CIFAR-10 Pilot
- **Creator:** RAIHAN, SIFAT (Researcher)
- **Affiliation:** Department of Computer Science and Technology, Hebei University of Science and Technology, Shijiazhuang 050018, Hebei, China
- **Resource type:** Publication — Preprint
- **Language:** English
- **Publication date / Issued:** 2026-09-25
- **Keywords:** CIFAR-10-C; image augmentation; corruption robustness; MobileNetV3-Small; RandAugment; AugMix; training cost
- **License / copyright:** CC BY 4.0; Copyright (c) 2026 Sifat Raihan
- **Description:** Final paper abstract, stating that the work is an exploratory two-seed pilot and has not undergone peer review.
- **Citation:** RAIHAN, S. (2026). Training-Time Image Augmentation on MobileNetV3-Small: A Controlled CIFAR-10 Pilot (Version 1.0). Zenodo. https://doi.org/10.5281/zenodo.22961521
