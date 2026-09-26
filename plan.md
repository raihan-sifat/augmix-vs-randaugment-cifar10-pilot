# Research Paper Plan

## Working title

**How Well Do Image Augmentation Methods Improve the Robustness of Lightweight Image Classifiers, and What Is the Cost in Accuracy and Computing Time?**

Possible shorter title: **Robustness–Efficiency Trade-offs of Image Augmentation for Lightweight Classifiers**.

## 1. Project summary

This project will compare a small set of image augmentation strategies for lightweight image classification. Models will be trained on CIFAR-10 and evaluated on the standard clean test set and CIFAR-10-C (common image corruptions); CIFAR-10.1 is an optional separately collected test set. The study will measure predictive performance and the direct compute cost of training-time augmentation, especially end-to-end training time and input-pipeline throughput.

**Current deadline constraint:** the user needs a complete experimental paper with original results within two days. That is enough time only for a small, transparently reported pilot and paper draft; it is not enough to establish a strong research gap or produce a publication-ready study. The short-deadline protocol below supersedes the 24-week schedule for this submission. The longer A-to-Z plan remains a path for extending and validating the work after the deadline.

The project is intended to build research experience for a master's degree in machine learning, AI, or image processing. A clean, reproducible experiment pipeline also demonstrates data engineering and software engineering skills.

### Main research question (pilot version)

> Under a fixed short training schedule on CIFAR-10, how do standard crop-and-flip augmentation, RandAugment, and AugMix compare in clean accuracy, CIFAR-10-C accuracy, and measured training time for MobileNetV3-Small?

### Supporting questions

1. Which strategy performs best on clean CIFAR-10 images?
2. Which strategy performs best across corruption types and severity levels in CIFAR-10-C?
3. How much extra end-to-end training time and data/augmentation processing time does each strategy require?
4. Do the trade-offs differ across multiple random seeds or between a compact model and, if resources allow, a reference CNN?
5. As an optional generalization check, do gains also appear on CIFAR-10.1?

### Initial hypotheses (exploratory; do not treat as expected outcomes)

- Stronger augmentation may improve average performance on corrupted images compared with basic augmentation.
- Stronger augmentation may reduce clean-image accuracy or increase training time; the size of these trade-offs may vary by method and corruption type.
- Improvement on synthetic corruptions may not transfer equally to a separately collected test set.

These are hypotheses to test, not expected results to force. Report findings even if they do not support them.

### Two-day deliverable and feasibility boundary

- Produce an honest experimental-paper draft with original pilot measurements, code, configurations, logs, and plots. Do not call it publication-ready or claim a novel method.
- Use one compact architecture (MobileNetV3-Small adapted for 32x32 inputs), three augmentation conditions, two fixed seeds if runtime permits, a fixed epoch count, and CIFAR-10/CIFAR-10-C evaluation. Omit CIFAR-10.1 and a second architecture.
- Start with a short runtime/memory pilot. If AugMix or the full corruption benchmark does not fit the available time, report a reduced, predeclared subset or one-seed exploratory study and clearly label it; never imply it is the full benchmark.
- Make no novelty claim: prior work already studies mobile classifiers with corruption robustness and augmentation runtime. Frame this as a small controlled replication/extension, with the distinct scope and limitations stated.
- Do not invent, estimate, or smooth results. All reported values must come from saved outputs. If runs are incomplete at the deadline, provide the partial empirical report and explicitly describe what remains.

## 2. Scope and boundaries

### Core experiment

- **Dataset for training:** CIFAR-10 training split.
- **Primary model:** One lightweight classifier, selected with the supervisor. Candidate: MobileNetV3-Small adapted for 32×32 inputs. Keep the architecture and all non-augmentation settings fixed across comparisons.
- **Augmentation conditions:**
  1. Basic augmentation baseline: random crop with padding and random horizontal flip.
  2. RandAugment added to the same basic baseline.
  3. AugMix added to the same basic baseline.
- **Evaluation sets:** untouched CIFAR-10 test split, CIFAR-10-C, and CIFAR-10.1 version v6.
- **Replicates:** At least three training seeds per condition if feasible. With one model, this is nine core training runs.
- **Optional extension:** Repeat the comparison using a second lightweight model, such as a CIFAR-adapted ResNet-18. Do this only after the core study is complete and the schedule and compute budget allow it.

Use published, documented implementations or carefully documented equivalents for RandAugment and AugMix. Freeze the augmentation settings before final training. Keep the same training examples, split, optimizer, number of epochs, scheduler, batch size, and stopping rule across conditions unless a justified change is necessary. Record any differences.

### Frozen short-deadline pilot protocol

1. **Hardware:** user's NVIDIA GeForce GTX 1650 with 4 GB VRAM. Record software and system details in the run manifest.
2. **Data:** CIFAR-10 training split, CIFAR-10 test split, and CIFAR-10-C if the official archive can be obtained and evaluated within the deadline. Keep both test sets out of training and configuration decisions.
3. **Model:** torchvision MobileNetV3-Small adapted for CIFAR resolution. Freeze and document exact input stem/classifier edits and parameter count before final runs.
4. **Conditions:** (A) random crop with padding plus horizontal flip; (B) the same baseline plus torchvision RandAugment with fixed documented parameters; (C) the same baseline plus AugMix with a documented implementation and loss. Do not describe a single AugMix transform as the full AugMix method if consistency views/JSD are omitted.
5. **Runs:** use seeds 17 and 23 for all three conditions if feasible. Frozen schedule after the runtime pilot: 50 epochs, batch size 128, two data-loader workers, SGD with learning rate 0.05/momentum 0.9/weight decay 0.0005, and cosine annealing. Keep training examples identical. The runtime pilot is for feasibility only, not choosing a winner.
6. **Outcomes:** clean CIFAR-10 test accuracy; CIFAR-10-C accuracy by corruption/severity plus an overall mean if the full benchmark is evaluated; end-to-end training time and examples/second. Log peak CUDA memory when available.
7. Report individual run values and mean/standard deviation across completed seeds. With fewer than two completed seeds, label results single-seed exploratory results and omit inferential language.
8. Save configurations, per-epoch CSV logs, checkpoints, run manifest, and evaluation outputs. Record software versions, hardware, data source/checksum, timestamps, failed runs, and deviations.
9. If the full run matrix or benchmark does not fit the deadline, disclose the completed subset and the remaining work precisely.

### Out of scope for the first version

- Proposing a new neural-network architecture.
- Training on CIFAR-10-C or CIFAR-10.1.
- Tuning methods against the final test sets.
- Claiming the results represent all image datasets or real-world deployment conditions.
- Adding many models, datasets, or augmentation policies before the core experiment is finished.

### Novelty check

Robustness benchmarks and augmentation methods already have substantial prior literature. The project must not claim that comparing these methods is automatically novel. During the literature review, identify what is not already answered by recent studies—for example, a carefully controlled comparison on lightweight models that reports both corruption-wise results and measured compute cost. If no defensible gap remains, revise the question with the supervisor before running the full experiment.

## 3. Background reading and starting references

Start with these primary sources, then follow their references and search for newer work from the last three to five years:

1. Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ICLR 2019): [paper](https://arxiv.org/abs/1903.12261), [authors' benchmark repository](https://github.com/hendrycks/robustness).
2. Hendrycks et al., *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty* (ICLR 2020): [paper](https://openreview.net/forum?id=S1gmrxHFvB).
3. Recht et al., *Do CIFAR-10 Classifiers Generalize to CIFAR-10?*: [paper](https://arxiv.org/abs/1806.00451), [CIFAR-10.1 dataset and guidance](https://github.com/modestyachts/CIFAR-10.1).
4. The original RandAugment paper and official or author-maintained implementation, identified and verified during the literature search.
5. Recent primary studies on corruption robustness, lightweight vision models, augmentation, and evaluation methodology. Prefer papers with released code or detailed experimental protocols.

### Literature review method

1. Search Google Scholar, Semantic Scholar, IEEE Xplore, ACM Digital Library, arXiv, and the proceedings of ICLR, ICML, NeurIPS, CVPR, ICCV, and ECCV.
2. Use search phrases such as `CIFAR-10-C lightweight model augmentation`, `corruption robustness image augmentation compute`, `RandAugment AugMix CIFAR-10-C`, and `robustness efficient image classification`.
3. Record each relevant paper in a literature matrix with: citation, research question, dataset, models, methods, metrics, main result, limitations, code availability, and how it relates to this project.
4. Read review papers to map the field, but use the primary papers to verify claims and experimental details.
5. Compare the planned experiment with the closest studies. Write a short gap statement supported by citations. Update it if the evidence does not support the initial gap.

## 4. Experimental protocol

### 4.1 Data and splits

- Download CIFAR-10, CIFAR-10-C, and CIFAR-10.1 from their official or author-maintained sources. Record URLs, download date, release/version, checksums where available, and dataset terms.
- Split the 50,000 CIFAR-10 training images into a fixed, stratified training and validation split (initial proposal: 45,000/5,000). Save the indices and use the same split for every run.
- Use CIFAR-10's official test split only for final evaluation.
- Use CIFAR-10-C and CIFAR-10.1 v6 only for final evaluation. Do not train on these data or select hyperparameters based on their results.
- Ensure normalization, label mapping, and input dimensions are consistent across conditions. Record any resizing or model-specific input adaptation.
- Check for corrupt files, duplicate records, missing labels, and unexpected tensor shapes before training. Summarize these checks in the repository and paper.

### 4.2 Training conditions

- Implement the basic augmentation baseline first and verify the training/evaluation pipeline.
- Add RandAugment and AugMix as separate conditions. Keep all other settings fixed.
- Select model, optimizer, learning rate, scheduler, batch size, epoch count, normalization, and augmentation parameters using the validation split and literature. Freeze these choices before the final runs.
- Use the same fixed set of random seeds for all conditions. Save the configuration and seed for every run.
- Do not select the best test-set checkpoint. Select checkpoints using the validation rule fixed in advance.
- Record hardware, operating system, library versions, training duration, peak memory if available, and any failed or repeated runs.

### 4.3 Evaluation measures

Report results separately for each dataset and each training seed.

**Predictive performance**

- Top-1 accuracy on the clean CIFAR-10 test set.
- Accuracy on CIFAR-10-C, broken down by corruption type and severity, plus clearly defined aggregate scores.
- Top-1 accuracy on CIFAR-10.1 v6.
- Difference in clean accuracy and corruption accuracy relative to the basic augmentation baseline, in percentage points.

**Computational cost**

- End-to-end training wall-clock time per run, average time per epoch, and training throughput.
- If feasible, separately profile data loading/augmentation time to estimate how much of the training cost comes from the input pipeline.
- Record hardware, worker settings, batch size, and timing procedure; compare methods on the same machine and under the same training schedule.
- Parameter count. FLOPs may be reported if the calculation method is available and consistent.
- Relative time overhead compared with the baseline.
- Inference latency is optional and describes the deployed model. Since the augmentation methods are applied during training, do not interpret inference latency as a direct cost of the training augmentation when the model and inference pipeline are otherwise identical.

Do not combine accuracy and compute into one score unless a score and its rationale are defined before examining results. Present the accuracy-versus-cost trade-off directly, using tables and plots.

### 4.4 Repeatability and statistical reporting

- Report the mean and standard deviation across training seeds for the main measures.
- Show per-seed values in supplementary tables or released result files.
- For CIFAR-10-C, show performance by corruption family/type and severity so a single average does not conceal failures.
- If using confidence intervals or significance tests, define the procedure before analysis and account for the fact that the same images are evaluated by multiple trained models.
- Treat the second model as a robustness check on the conclusion, not as a reason to hide disagreement between models.

## 5. Data handling, ethics, and research integrity

- This planned project uses public image datasets and no human participants. Confirm with the supervisor and institution whether any formal ethics or data-use review is required before starting.
- Follow each dataset's terms and attribution requirements. Do not redistribute dataset files unless the terms permit it; provide download instructions and checksums instead.
- Keep raw datasets separate from derived data and generated outputs.
- Never alter labels, discard inconvenient results, or change the protocol after seeing test results without documenting the change and its reason.
- Keep a dated experiment log, version-controlled code, configuration files, and raw result files.
- Disclose any AI tools used for writing, coding, or analysis according to university and target-venue policies. Verify all generated code, statements, and citations yourself.
- Cite the original methods and datasets, not only tutorials or blog posts.

## 6. Reproducible project structure

Suggested folder layout:

```text
research-paper/
  plan.md
  README.md
  requirements.txt              # or environment.yml / pyproject.toml
  configs/
    baseline.yaml
    randaugment.yaml
    augmix.yaml
  src/
    data.py
    models.py
    augmentations.py
    train.py
    evaluate.py
    metrics.py
  scripts/
    run_experiments.*
    summarize_results.*
  results/
    raw/
    processed/
    figures/
    tables/
  paper/
    outline.md
    manuscript.docx            # or LaTeX source
  references.bib
```

Do not commit large datasets, private credentials, or machine-specific absolute paths. The README should explain setup, dataset acquisition, one training run, full experiment execution, and result reproduction.

## 7. End-to-end work plan (A to Z)

### Phase A — Define the project (Weeks 1–2)

1. Confirm the project scope and expected paper format with a supervisor or lecturer.
2. Check available hardware, programming environment, deadline, and weekly hours.
3. Finalize the primary research question and decide the smallest complete experiment.
4. Create a project log, reference manager library, and version-controlled code repository.
5. Deliverable: one-page scope note and confirmed schedule.

### Phase B — Review research and establish the gap (Weeks 2–5)

1. Read the benchmark, augmentation-method, and dataset papers listed above.
2. Search for close recent comparisons; complete the literature matrix.
3. Identify what is already known about clean accuracy, corruption robustness, lightweight models, and training/inference cost.
4. Draft a short related-work section and a citation-supported gap statement.
5. If the planned comparison is already fully answered, adjust the focus (for example, a specific lightweight architecture or a more careful compute-matched evaluation) before continuing.
6. Deliverable: literature matrix, draft gap statement, and revised research question.

### Phase C — Write and approve the protocol (Weeks 5–6)

1. Specify datasets and versions, split indices, model, augmentation conditions, hyperparameters, seeds, metrics, timing procedure, and exclusion rules.
2. Define the validation-only tuning process and test-set lock.
3. Ask the supervisor to review feasibility and whether the contribution is appropriate for a bachelor’s paper.
4. Save the dated protocol in the repository before final experiments.
5. Deliverable: approved experimental protocol.

### Phase D — Build the data and software pipeline (Weeks 6–8)

1. Create the environment and record dependency versions.
2. Implement dataset loading, split generation, data checks, preprocessing, model, and metrics.
3. Add deterministic configuration and random-seed handling as far as the framework/hardware allows.
4. Add structured logging for configuration, epoch metrics, checkpoints, run time, and errors.
5. Verify that evaluation code produces the same result when run twice on a saved checkpoint.
6. Deliverable: documented pipeline and a reproducible baseline run.

### Phase E — Pilot and freeze settings (Weeks 8–9)

1. Run a small pilot to estimate runtime and memory requirements.
2. Check training curves, validation behavior, data transformations, and metric calculations.
3. Choose final hyperparameters using only training/validation data and written sources.
4. Fix the complete experiment matrix and estimate the remaining compute time.
5. Deliverable: frozen configurations and final run schedule.

### Phase F — Run the core experiments (Weeks 9–13)

1. Train the basic, RandAugment, and AugMix conditions with each planned seed.
2. Save checkpoints, logs, configuration files, and measured training time.
3. Monitor failures and rerun only for documented technical errors, not because the results are unfavorable.
4. Keep test-set evaluation separate until the complete training protocol is finished.
5. Deliverable: complete trained checkpoints and raw training logs.

### Phase G — Evaluate and analyze (Weeks 13–16)

1. Evaluate the locked checkpoints on clean CIFAR-10, CIFAR-10-C, and CIFAR-10.1.
2. Measure inference time using the prewritten timing procedure.
3. Generate tables and plots for clean accuracy, corruption type/severity, CIFAR-10.1 accuracy, and compute cost.
4. Summarize across seeds and inspect failures or unexpectedly large method differences.
5. Run only the preplanned additional checks. Clearly label any exploratory analysis.
6. Deliverable: final result tables, figures, analysis notes, and machine-readable result files.

### Phase H — Interpret results and check claims (Weeks 16–17)

1. Answer each research question using the reported results.
2. Compare findings with the closest prior studies and explain agreements or differences.
3. Discuss limitations: small low-resolution dataset, chosen model(s), benchmark corruptions, compute measurement hardware, and transfer to other domains.
4. Check that every conclusion is supported by a result and that no causal claim exceeds the experimental design.
5. Deliverable: concise findings summary and limitations list.

### Phase I — Write the paper (Weeks 17–20)

Use the target venue's or university's required format. A standard structure is:

1. **Title and abstract:** problem, method, dataset, headline result, and contribution.
2. **Introduction:** motivation, research gap, question, and contributions.
3. **Related work:** robustness benchmarks, augmentation methods, lightweight models, and cost reporting.
4. **Method:** datasets, split policy, model, augmentation conditions, training settings, metrics, compute timing, and reproducibility details.
5. **Results:** main results, trade-offs, corruption breakdowns, and seed variation.
6. **Discussion:** meaning of the results and comparison to prior work.
7. **Limitations and conclusion:** what the study cannot establish and the next research step.
8. **References and appendices:** complete citations, extra tables, configurations, and implementation notes.

Write Methods and Results from the saved protocol and outputs. Do not invent missing measurements. Deliverable: complete first manuscript draft.

### Phase J — Review and revise (Weeks 20–22)

1. Ask the supervisor and at least one technically informed reader to review the manuscript.
2. Check correctness of equations, dataset descriptions, model details, and citations.
3. Verify each table and figure against the raw result files.
4. Improve clarity, remove unsupported claims, and ensure all limitations are explicit.
5. Run an originality and citation check using approved university tools.
6. Deliverable: submission-ready draft and reproducibility package.

### Phase K — Select venue and submit (Weeks 22–24)

1. Choose a suitable student journal, undergraduate research venue, or conference with the supervisor.
2. Verify its scope, peer-review process, author instructions, fees, copyright terms, and submission deadlines using its official website.
3. Format the manuscript, abstract, references, figures, and supplementary files exactly as required.
4. Submit only after all authors approve the final version and the work complies with university rules.
5. Save the submitted files and confirmation.
6. Deliverable: completed submission, or a final university paper if no appropriate submission window is available.

### Phase L — Respond to review and archive (after submission)

1. Read reviewer/editor feedback with the supervisor.
2. Prepare a point-by-point response and revise the manuscript; explain any requested change that cannot be made.
3. Re-run affected experiments if necessary and document changes.
4. Archive the final code, configurations, results, paper version, and data-access instructions in an appropriate repository, following dataset terms.
5. Record the outcome and lessons for master's applications or future research.

## 8. Proposed 24-week schedule

| Weeks | Main work | Milestone |
|---|---|---|
| 1–2 | Scope, supervisor, resources, workspace | Topic and schedule confirmed |
| 2–5 | Literature search and comparison matrix | Defensible gap and refined question |
| 5–6 | Experimental protocol and review | Protocol frozen |
| 6–8 | Data/software pipeline and baseline | Reproducible baseline |
| 8–9 | Pilot, runtime estimate, settings freeze | Final experiment matrix |
| 9–13 | Core training runs | All planned checkpoints and logs |
| 13–16 | Final evaluation, compute measurements, analysis | Results and figures complete |
| 16–17 | Interpretation and limitations | Findings summary complete |
| 17–20 | Manuscript writing | Full draft |
| 20–22 | Supervisor/peer review and revisions | Submission-ready paper |
| 22–24 | Venue selection and submission preparation | Submission or final university paper |

If the degree deadline is sooner, prioritize the core single-model experiment, a careful literature review, and a complete paper. Treat the second model and extra datasets as optional extensions.

## 9. Risks and responses

| Risk | Response |
|---|---|
| Literature shows the exact comparison is already well covered | Narrow or revise the question before full training; do not claim novelty without evidence. |
| Compute is limited | Keep one model, three methods, and fewer seeds only with explicit justification; use a pilot to forecast runtime. |
| Augmentation code differs from published settings | Record implementation source, parameters, and deviations; run a small correctness check. |
| Test results are used for repeated tuning | Lock test sets until final evaluation; use the validation split for all tuning. |
| One average hides a weakness on a corruption type | Report per-type and per-severity results as well as aggregate metrics. |
| Timing measurements are noisy or hardware-specific | Fix device, batch size, warm-up, and repetition count; report the hardware and timing procedure. |
| The paper finds small or mixed gains | Report them honestly; a well-controlled negative or nuanced result is still a valid research outcome. |
| Publication deadline or review cycle exceeds graduation schedule | Complete the university paper on time and submit externally only to an appropriate venue. |

## 10. Definition of completion

The project is complete when it has:

- A literature-supported and appropriately scoped research question.
- A documented, leakage-aware experimental protocol.
- Completed planned runs with saved configurations and logs.
- Final evaluations on all planned test sets and measured compute costs.
- Results reported across seeds and with corruption-level detail.
- A complete paper with supported claims, limitations, and accurate citations.
- A README or equivalent instructions that allow another student to reproduce the pipeline and understand how to obtain the datasets.
- Supervisor-approved final paper and a venue decision or completed submission, subject to university requirements.

## 11. Immediate checklist

- [ ] Find a supervisor/lecturer in machine learning or computer vision.
- [ ] Confirm available GPU/CPU hardware and project deadline.
- [ ] Read the three core papers in Section 3.
- [ ] Search recent papers to test whether the proposed gap is real.
- [ ] Confirm dataset terms and whether the institution requires any review.
- [ ] Approve the final model and three augmentation conditions.
- [ ] Freeze and date the experimental protocol before final runs.
