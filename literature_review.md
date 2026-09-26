# Literature Review Work Log

**Status:** Initial primary-source screening; this is not yet a complete literature review or a verified novelty claim.  
**Last updated:** 2026-09-25  
**Project plan:** [plan.md](plan.md)  
**Progress tracker:** [progress.md](progress.md)

## Candidate refinement

The broad question about augmentation and robustness is already an established research area. A more focused candidate is:

> On compact image classifiers trained on CIFAR-10, how do selected training-time augmentation methods affect clean accuracy, corruption robustness, and end-to-end training cost under a controlled training schedule?

Possible comparative extension: test whether the trade-off differs between one compact/mobile-oriented model and a reference CNN. This depends on available compute and the full literature review.

This framing is provisional. It is not yet established that the compact-model and measured-cost angle is novel. Search directly for papers that combine lightweight architectures, corruption benchmarks, and runtime or throughput measurements before freezing the question.

## Initial source matrix

| Source | What it studies / relevant takeaway | How it informs this project | Follow-up to verify |
|---|---|---|---|
| Hendrycks & Dietterich, 2019, [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261) | Establishes standardized evaluation of image classifiers on common corruptions and perturbations, including CIFAR-10-C. | Gives the basis for the corruption evaluation and reporting by corruption/severity. | Confirm the exact CIFAR-10-C release and metric conventions used in the final protocol. |
| Yin et al., NeurIPS 2019, [A Fourier Perspective on Model Robustness in Computer Vision](https://papers.nips.cc/paper_files/paper/2019/hash/b05b57f6add810d3b7490866d74c0053-Abstract.html) | Finds corruption robustness gains can vary by corruption characteristics; one augmentation can help some corruption families and hurt others. | Supports reporting per-corruption results rather than only one average. | Read experimental tables and identify overlap with candidate augmentation methods. |
| Hendrycks et al., ICLR 2020, [AugMix](https://openreview.net/forum?id=S1gmrxHFvB) | Evaluates AugMix on four CIFAR model families (AllConvNet, DenseNet, WideResNet, ResNeXt) and reports corruption robustness gains. The method combines multiple augmentation views with a Jensen-Shannon consistency loss. | AugMix is an established comparison method; its extra views/loss can add model computation as well as image-transform cost. A compact-model runtime study must measure both honestly. | Compare the exact CIFAR recipe and runtime reporting to our intended implementation; determine whether AllConvNet in that paper is sufficiently comparable to mobile-oriented backbones. |
| Cubuk et al., CVPR Workshops 2020, [RandAugment](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html) | Reduces the search space and cost of automated augmentation policy selection. | Relevant to choosing a practical augmentation baseline and separating policy-search cost from online augmentation cost. | Check how the paper defines computational cost and whether its measurements transfer to compact CIFAR models. |
| Mintun et al., NeurIPS 2021, [On Interaction Between Augmentations and Corruptions](https://proceedings.neurips.cc/paper/2021/hash/1d49780520898fe37f0cd6b41c5311bf-Abstract.html) | Studies the relationship between training augmentations and test-time corruptions; reports that robustness may degrade on corruptions dissimilar to those represented by the benchmark. | Motivates careful scope and avoiding claims that CIFAR-10-C performance guarantees broad real-world robustness. | Review datasets and model sizes, and note whether augmentation runtime is measured. |
| Modas et al., ECCV 2022, [PRIME](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850615.pdf) | Compares corruption-robust augmentation methods, includes CIFAR-10-C results with ResNet-18, studies accuracy/robustness trade-offs, and reports some relative augmentation runtime measurements. | Confirms that both robustness and compute cost have prior work; the candidate gap must be narrower than “compare accuracy and time.” | Determine exactly which timing is measured, which models/datasets are timed, and whether compact/mobile backbones are covered. |
| Kireev et al., UAI 2022, [On the Effectiveness of Adversarial Training Against Common Corruptions](https://proceedings.mlr.press/v180/kireev22a.html) | Uses PreAct ResNet-18 on CIFAR-10 and CIFAR-10-C; reports corruption accuracy and a wall-clock table for several robust-training recipes on one V100 GPU. The time table does not directly give timings for every augmentation baseline. | Confirms that controlled CIFAR corruption evaluation and wall-clock reporting already exist; compare the proposed scope with this work, while distinguishing augmentation overhead from adversarial-training cost. | Read the timing table and training details; record which methods were timed, hardware, epochs, and what it leaves unmeasured. |
| Fadhilah, Warsito & Faridah, JIKI 2026, [Enhanced Robustness in Image Classification through DistortionMix](https://doi.org/10.21609/jiki.v19i1.1558) | Evaluates clean CIFAR-10 and corrupted CIFAR-10-C across many architectures, including MobileNetV3-Small, with and without a simple noise/contrast augmentation. Reports accuracy and 50-epoch training-time comparisons; reports a MobileNetV3-Small clean/corrupt accuracy increase. | This is a close and very recent overlap: mobile classifier, CIFAR corruption benchmark, augmentation, and training time are already combined in one study. Our experiment must be positioned only as a small head-to-head pilot of different policies under matched settings, not as an unaddressed general question or new method. | Read its detailed implementation and timing tables. Note that the paper describes 19 corruption types, whereas the canonical CIFAR-10-C benchmark has 15; verify the exact artifact/version before comparing numbers. Its pretrained fine-tuning protocol also differs from our planned from-scratch pilot. |
| Trinh et al., NeurIPS 2024, [Improving Robustness to Corruptions with Multiplicative Weight Perturbations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3e9412a9c1d93810ef3ef7825115016b-Abstract-Conference.html) | Studies corruption robustness with a weight-perturbation training method and compares against the broader robustness literature. | A recent lead for related trade-offs and alternative methods; not the same as a direct augmentation-only benchmark. | Read its baselines and references for newer related studies on CIFAR-10-C. |

## Key design implications so far

1. **Training time is the direct compute outcome.** These are training-time augmentations. With a fixed network and inference preprocessing, augmentation choice should not materially change model inference latency. If latency is included, report it as a model deployment characteristic, not as an augmentation benefit.
2. **Separate different notions of cost.** Track end-to-end training wall time/throughput, and, if feasible, augmentation/data-loader time. Distinguish this from the cost of searching for an augmentation policy, precomputing augmented data, or model inference.
3. **Show per-corruption results.** Aggregate robustness scores can hide a method that improves noise robustness while weakening performance on blur, weather, or digital corruptions.
4. **Treat CIFAR-10-C as a controlled benchmark.** Conclusions should be limited to the tested datasets, models, augmentations, and hardware.
5. **Do not claim novelty.** PRIME already compares robustness and some runtime costs, including ResNet-18 corruption results. Fadhilah et al. (2026) further cover a MobileNetV3-Small, CIFAR-C, an augmentation comparison, and training time. The planned short-deadline experiment is best described as a limited, controlled head-to-head replication-style pilot of other augmentation policies.

## Search log and next review tasks

Searches started on 2026-09-25 using primary proceedings and author papers for combinations of `CIFAR-10-C`, augmentation, lightweight/mobile CNNs, and training efficiency. A close 2026 paper makes a broad novelty claim untenable; this scan is still not exhaustive.

Next:

- [ ] Read the full AugMix, RandAugment, Mintun et al., PRIME, and Kireev et al. papers; extract architecture, training settings, seed count, evaluation protocol, and compute measurements.
- [x] Search specifically for mobile classifiers evaluated on CIFAR-10-C with augmentation and training-time measurements; a close 2026 MobileNetV3-Small study is recorded above.
- [ ] Search for papers reporting per-epoch time, end-to-end training time, throughput, memory, or energy for augmentation pipelines.
- [ ] Follow references and citations from PRIME and Mintun et al. to find the closest prior comparisons.
- [ ] Decide whether CIFAR-10.1 adds a useful second evaluation or makes the scope unnecessarily broad.
- [ ] Write a one-paragraph gap statement with at least three direct comparisons to the closest prior studies.
- [ ] Only then freeze the research question, architecture, methods, and run matrix.

## Short-deadline study note

The user has a two-day deadline for a complete experimental-paper draft with original results. The initial literature search found direct overlap with the broad subject, so the immediate deliverable is an exploratory, matched-setting comparison of basic crop/flip, RandAugment, and AugMix on MobileNetV3-Small. It cannot establish a strong novelty claim or a publishable contribution. The 2026 DistortionMix paper reports fine-tuning and describes 19 corruptions; do not compare its numeric values directly with a different training setup or assume that its CIFAR-C artifact matches the canonical 15-corruption release.

## Literature extraction template

For each source, record:

```text
Full citation and stable URL:
Publication type/year:
Research question:
Dataset(s) and split:
Model(s) and parameter count:
Augmentation methods and settings:
Training schedule and seeds:
Clean/robustness metrics:
Compute metric (and hardware):
Main result relevant to our question:
Limitations stated by authors:
What it answers for our project:
What remains unanswered (verified from full paper):
```
