# Training-Time Image Augmentation on MobileNetV3-Small: A Controlled CIFAR-10 Pilot

**Sifat Raihan**  
Department of Computer Science and Technology  
Hebei University of Science and Technology  
Shijiazhuang 050018, Hebei, China  

**Status:** Exploratory experimental report; results are populated from the saved runs.  

## Abstract

Training-time image augmentation may improve classification under image corruption, but stronger policies can add computational cost and may change clean accuracy. This study compares random crop and horizontal flip, RandAugment, and AugMix using a CIFAR-adapted MobileNetV3-Small trained from scratch on CIFAR-10. Each condition was trained for 50 epochs with two seeds. We measured top-1 accuracy on the clean CIFAR-10 test set and the original 15 CIFAR-10-C corruption types at five severities, along with training wall time, source-image throughput, and peak allocated GPU memory on an NVIDIA GeForce GTX 1650 with 4 GB VRAM. Mean clean accuracies were 86.39% for crop-and-flip, 86.18% for RandAugment, and 86.02% for AugMix; mean CIFAR-10-C accuracies were 67.04%, 71.00%, and 77.82%, respectively. Mean training times were 13.2, 19.1, and 121.4 minutes. Run times varied substantially between seeds, particularly for RandAugment and AugMix. With two seeds, one architecture, one dataset, and one local system, these results are exploratory and do not support broad generalization or a novelty claim.

**Keywords:** CIFAR-10-C; image augmentation; corruption robustness; MobileNetV3-Small; RandAugment; AugMix; training cost

## 1. Introduction

Image classifiers can lose accuracy when test images contain blur, noise, weather effects, or digital artifacts. The CIFAR-10-C benchmark provides a controlled way to evaluate common corruptions across severity levels [1]. Training-time augmentation is one approach to improving generalization, but different policies expose a model to different transformed examples and may impose different training costs.

RandAugment applies a fixed number of randomly selected transformations at sampled strengths, avoiding the separate policy-search phase used by some automated methods [2]. AugMix composes and mixes augmentation chains and uses a consistency objective to encourage similar predictions across augmented views [3]. Both techniques have previously been studied on corruption benchmarks. More recently, Fadhilah et al. evaluated an additional distortion-based augmentation with multiple architectures, including MobileNetV3-Small, and reported clean/corrupted accuracy and training time [4]. Therefore, the broad question of augmentation, corruption robustness, and training cost is already represented in prior work.

This paper reports a small local comparison of three established augmentation conditions under matched training settings on one compact classifier. It asks: **Under a fixed short training schedule on CIFAR-10, how do crop-and-flip, RandAugment, and AugMix compare in clean accuracy, CIFAR-10-C accuracy, and measured training time for MobileNetV3-Small?** The contribution is a reproducible pilot measurement under the stated local setup, not a new augmentation method or a claim that the topic is novel.

## 2. Related Work

Hendrycks and Dietterich introduced CIFAR-10-C and related corruption benchmarks for measuring robustness beyond clean test accuracy [1]. Their benchmark motivates reporting performance by corruption and severity, since a single aggregate can hide differences between corruption types.

RandAugment reduces the search space of automated augmentation by choosing transformations and strengths from a compact policy [2]. AugMix combines stochastic augmentation chains and a Jensen-Shannon consistency loss; its original evaluation includes common corruption benchmarks [3]. These methods are established baselines rather than new proposals in the present study.

Fadhilah et al. (2026) compare their contrast/noise-based DistortionMix method with a standard training recipe on CIFAR-10 and CIFAR-10-C across several model families, including MobileNetV3-Small, and report training-time measurements [4]. Their architecture coverage and outcome types overlap directly with the present topic. Our pilot differs in the augmentation policies compared and uses one controlled native-resolution setup with matched settings; the small run budget and distinct implementation are limitations, not grounds for a broad novelty claim.

## 3. Methods

### 3.1 Data

The training data are the 50,000-image CIFAR-10 training split. The 10,000-image official CIFAR-10 test split is used only for final clean evaluation, once per final checkpoint after all runs are complete. For corruption evaluation, we use the original 15 CIFAR-10-C corruption types at severities 1–5: brightness, contrast, defocus blur, elastic transform, fog, frost, Gaussian noise, glass blur, impulse noise, JPEG compression, motion blur, pixelate, shot noise, snow, and zoom blur [1]. This gives 750,000 corrupted test images per checkpoint. The archive was downloaded from the official [Zenodo record](https://zenodo.org/records/2535967) on 2026-09-25; its MD5 is `56bf5dcef84df0e2308c6dcbcbbd8499`. Any extra corruption arrays in an expanded archive version are excluded from the primary aggregate. No test images are used for training, hyperparameter selection, stopping, or selecting a preferred condition.

### 3.2 Model

The classifier is torchvision MobileNetV3-Small [5] initialized with random weights. To accommodate 32 by 32 CIFAR images, we change the stride of the first convolution from 2 to 1 and replace the final classifier with a 10-class linear layer. All conditions use the same architecture, with 1,528,106 trainable parameters. Input pixels are normalized with CIFAR-10 channel means (0.4914, 0.4822, 0.4465) and standard deviations (0.2470, 0.2435, 0.2616).

### 3.3 Augmentation conditions

All methods use random crop with four-pixel padding and random horizontal flip.

1. **Crop-and-flip baseline:** the shared geometric transforms only.
2. **RandAugment:** the shared transforms followed by torchvision RandAugment with two operations and magnitude 9.
3. **AugMix:** the shared geometric transforms followed by three views of each source image: one clean view and two independently generated AugMix views. Each AugMix transform uses severity 3, mixture width 3, random chain depth, and alpha 1.0. We average cross-entropy over the three views and add a Jensen-Shannon consistency term weighted by 12.

### 3.4 Training protocol

Each condition is trained with seeds 17 and 23. We use 50 epochs and batch size 128, stochastic gradient descent with initial learning rate 0.05, momentum 0.9, weight decay 0.0005, and cosine annealing over the run. The final-epoch checkpoint is evaluated for every run; test accuracy does not determine checkpoint selection. Separate one-epoch feasibility runs using seed 99 are excluded from accuracy comparisons.

The training code records the configuration, software versions, epoch loss and accuracy, epoch duration, source-image throughput, checkpoint, total training time, peak allocated CUDA memory, and parameter count. Training time includes data loading and online augmentation from the beginning of the first epoch through the end of the final epoch; it excludes initial dataset download, model initialization, and evaluation. AugMix processes three model views per source image, which is noted when interpreting source-example throughput. One-epoch runs for each condition with seed 99 are feasibility measurements only and are excluded from accuracy comparisons. Experiments ran on Windows 11 with Python 3.14.5, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, and NumPy 2.5.2, using CUDA on an NVIDIA GeForce GTX 1650 with 4 GB VRAM.

### 3.5 Metrics and analysis

We report top-1 clean test accuracy and top-1 accuracy for every original CIFAR-10-C corruption at each of five severity levels. The aggregate CIFAR-10-C accuracy is the unweighted mean over all 75 corruption-severity pairs. We report training time in seconds/minutes, source CIFAR examples per second, GPU peak allocated memory, and parameter count. Per-seed values are shown with the sample mean and standard deviation across two seeds. These descriptive statistics do not support inferential claims.

## 4. Results

### 4.1 Run completion

| Condition | Completed seeds | Epochs per run | Failed/incomplete runs |
|---|---:|---:|---|
| Crop-and-flip | 2 | 50 | 0 |
| RandAugment | 2 | 50 | 0 |
| AugMix | 2 | 50 | 0 |

### 4.2 Accuracy and training cost

| Condition | Clean accuracy, seed 17 | Clean accuracy, seed 23 | Clean accuracy, mean ± SD | CIFAR-10-C accuracy, seed 17 | CIFAR-10-C accuracy, seed 23 | CIFAR-10-C accuracy, mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| Crop-and-flip | 86.21% | 86.57% | 86.39 ± 0.25% | 68.19% | 65.89% | 67.04 ± 1.63% |
| RandAugment | 86.76% | 85.60% | 86.18 ± 0.82% | 70.95% | 71.04% | 71.00 ± 0.06% |
| AugMix | 86.03% | 86.01% | 86.02 ± 0.01% | 78.04% | 77.61% | 77.82 ± 0.31% |

| Condition | Training time, seed 17 (s) | Training time, seed 23 (s) | Training time, mean ± SD (s) | Source examples/s, seed 17 | Source examples/s, seed 23 | Peak allocated GPU memory (GB) |
|---|---:|---:|---:|---:|---:|---:|
| Crop-and-flip | 711 | 876 | 793 ± 117 | 3,518 | 2,855 | 0.206 |
| RandAugment | 729 | 1,558 | 1,144 ± 586 | 3,429 | 1,604 | 0.206 |
| AugMix | 8,258 | 6,307 | 7,283 ± 1,379 | 303 | 396 | 0.557 |

GPU memory is peak allocated CUDA tensor memory in decimal GB (bytes divided by 10^9); it does not include the CUDA context or overall device use.

Crop-and-flip had the highest mean clean accuracy, although the three condition means differed by less than 0.4 percentage points. Relative to crop-and-flip, RandAugment increased mean CIFAR-10-C accuracy by 3.96 percentage points and AugMix by 10.78 points. The measured mean training times were 13.2, 19.1, and 121.4 minutes, respectively. Run-level times varied substantially: RandAugment seed 23 took 2.14 times as long as seed 17, and AugMix took 137.6 versus 105.1 minutes. These timing means are descriptive local measurements; the variation limits conclusions about the methods' relative compute cost. Per-seed outputs and aggregate tables are included in the accompanying reproducibility files.

### 4.3 Corruption-wise results

The following values average each corruption over five severities and both seeds.

| Corruption | Crop-and-flip (%) | RandAugment (%) | AugMix (%) |
|---|---:|---:|---:|
| Brightness | 84.50 | 85.22 | 84.56 |
| Contrast | 59.17 | 65.67 | 71.12 |
| Defocus blur | 74.72 | 78.69 | 83.99 |
| Elastic transform | 76.72 | 78.73 | 81.09 |
| Fog | 74.53 | 77.73 | 78.72 |
| Frost | 68.83 | 72.44 | 76.56 |
| Gaussian noise | 47.75 | 51.50 | 70.81 |
| Glass blur | 53.66 | 62.15 | 70.61 |
| Impulse noise | 51.83 | 63.36 | 71.87 |
| JPEG compression | 78.63 | 79.92 | 81.62 |
| Motion blur | 65.89 | 70.53 | 80.14 |
| Pixelate | 71.74 | 70.12 | 81.19 |
| Shot noise | 56.82 | 59.99 | 75.22 |
| Snow | 72.53 | 74.22 | 77.94 |
| Zoom blur | 68.29 | 74.69 | 81.91 |

Mean accuracy by corruption severity, averaging across the 15 corruptions and two seeds, was:

| Severity | Crop-and-flip (%) | RandAugment (%) | AugMix (%) |
|---:|---:|---:|---:|
| 1 | 79.77 | 81.54 | 83.29 |
| 2 | 74.22 | 77.44 | 81.29 |
| 3 | 68.17 | 72.70 | 79.10 |
| 4 | 60.89 | 66.18 | 75.77 |
| 5 | 52.17 | 57.12 | 69.67 |

## 5. Discussion

The results show a clean-accuracy and corruption-accuracy trade-off in this pilot. Crop-and-flip led the clean mean at 86.39%, while AugMix had the highest mean CIFAR-10-C accuracy at 77.82%, 10.78 percentage points above the 67.04% baseline. RandAugment reached 71.00% on CIFAR-10-C. The corruption-wise table shows that gains were not uniform: for example, RandAugment's pixelate average was below crop-and-flip, while AugMix improved the noise, blur, and contrast averages. These outcomes apply to this 50-epoch schedule, the CIFAR-adapted model, these implementations, and this GPU. AugMix uses three model views per source image and a consistency loss, so its measured time reflects both online transforms and additional model computation. Timing variability was large: RandAugment took 12.2 versus 26.0 minutes across seeds, and AugMix took 137.6 versus 105.1 minutes. A single epoch in AugMix seed 23 took 865.9 seconds while most late epochs took 80–110 seconds. The machine's system state was not controlled tightly enough to explain these differences; therefore the timing comparison is a local observation, not a stable estimate of augmentation overhead. CIFAR-10-C uses synthetic corruptions and does not establish robustness to arbitrary real-world distribution shifts.

This pilot complements the existing literature only as a locally measured comparison of three established policies under one matched configuration. The close overlap with Fadhilah et al. [4], the short training schedule, small seed count, and single architecture preclude a strong novelty or generalization claim.

## 6. Limitations

1. The study uses one compact architecture and one low-resolution dataset.
2. The fixed short schedule may not reflect converged performance.
3. Two seeds provide limited information about training variability; the reported standard deviations are descriptive and no hypothesis tests were performed.
4. Timing is specific to one GTX 1650 system, software stack, worker configuration, and implementation. It is not a hardware-independent compute-cost estimate.
5. CIFAR-10-C contains synthetic corruptions; performance there does not guarantee performance on naturally corrupted images or other domains.
6. Prior work already studies mobile classifiers, corruption robustness, augmentation, and training time. This pilot does not establish a new general research gap.
7. Wall-clock time varied substantially between seeds under identical settings, including a 865.9-second AugMix epoch; system load and power state were not controlled or measured throughout all runs.
8. The study does not measure energy use or separate image-transform time from model computation, so it cannot identify the source of each method's compute overhead.

## 7. Conclusion

This study compared crop-and-flip, RandAugment, and AugMix for a CIFAR-adapted MobileNetV3-Small over two seeds and 50 epochs. Crop-and-flip had the highest clean mean accuracy (86.39%); AugMix had the highest CIFAR-10-C mean (77.82%), followed by RandAugment (71.00%) and crop-and-flip (67.04%). The measured average training times were 13.2, 19.1, and 121.4 minutes, but substantial run-to-run variability makes these local measurements uncertain estimates of relative compute cost. The results answer the pilot question for this model, dataset, schedule, and hardware only. They do not establish a new method, broad robustness, or a publication-level research gap. A stronger follow-up should control system state, increase the seed count, use longer training, include more compact architectures, and compare the design directly with the closest prior work.

## References

[1] D. Hendrycks and T. Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,” ICLR, 2019. https://arxiv.org/abs/1903.12261

[2] E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, “RandAugment: Practical Automated Data Augmentation with a Reduced Search Space,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 702–703, 2020. https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html

[3] E. D. Cubuk, N. Mu, D. Hendrycks, B. Lakshminarayanan, B. Zoph, and J. Gilmer, “AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty,” in Proceedings of the International Conference on Learning Representations (ICLR), 2020. https://openreview.net/forum?id=S1gmrxHFvB

[4] H. Fadhilah, B. Warsito, and H. Faridah, “Enhanced Robustness in Image Classification through DistortionMix: A Hybrid Distortion-Based Augmentation Technique,” Jurnal Ilmu Komputer dan Informasi, vol. 19, no. 1, pp. 33–52, 2026. https://doi.org/10.21609/jiki.v19i1.1558

[5] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, “Searching for MobileNetV3,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324, 2019. https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html
