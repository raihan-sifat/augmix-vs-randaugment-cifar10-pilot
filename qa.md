# Interview Q&A Bank — Augmentation on MobileNetV3-Small (CIFAR-10 Pilot)

**Paper:** *Training-Time Image Augmentation on MobileNetV3-Small: A Controlled CIFAR-10 Pilot*
**Author:** Sifat Raihan · Hebei University of Science and Technology · DOI `10.5281/zenodo.22961521`
**Companion:** `overview.md` (full explanation) · **Total questions: 130**

---

## How to use this document

**The golden rule: never overstate.** This is a two-seed, one-model, one-dataset pilot. Every answer
below is calibrated to that. Volunteering a limitation *before* you are asked is worth more than any
result — it is the single strongest signal that you understand your own work.

**The three fatal mistakes to avoid:**
1. Claiming novelty. A 2026 paper (Fadhilah et al.) already did almost the same combination.
2. Quoting the timing table as if the numbers are reliable. They are not, and §H explains why.
3. Comparing the training-accuracy or training-loss numbers across the three methods. They are
   measured on different things. §F/§G explain exactly why.

**Difficulty legend:** 🟢 warm-up · 🟡 real probe · 🔴 hard / adversarial · 💣 trap

---

## Table of contents

- [A. The 30-second elevator pitch](#a-the-30-second-elevator-pitch)
- [B. Fundamentals: what, why, how](#b-fundamentals-what-why-how)
- [C. Datasets: CIFAR-10 and CIFAR-10-C](#c-datasets-cifar-10-and-cifar-10-c)
- [D. The model: MobileNetV3-Small](#d-the-model-mobilenetv3-small)
- [E. The three augmentation methods](#e-the-three-augmentation-methods)
- [F. Training setup and hyperparameters](#f-training-setup-and-hyperparameters)
- [G. Results and how to interpret them](#g-results-and-how-to-interpret-them)
- [H. Timing and performance (the hard one)](#h-timing-and-performance-the-hard-one)
- [I. Methodology, validity, and data leakage](#i-methodology-validity-and-data-leakage)
- [J. Statistics and seeds](#j-statistics-and-seeds)
- [K. Limitations and self-criticism](#k-limitations-and-self-criticism)
- [L. Literature, related work, and novelty](#l-literature-related-work-and-novelty)
- [M. Reproducibility, code, and tooling](#m-reproducibility-code-and-tooling)
- [N. Ethics, licensing, and AI use](#n-ethics-licensing-and-ai-use)
- [O. Future work and research vision](#o-future-work-and-research-vision)
- [P. Questions about you](#p-questions-about-you)
- [Q. Rapid-fire one-liners](#q-rapid-fire-one-liners)
- [R. The 12 hardest questions, ranked](#r-the-12-hardest-questions-ranked)

---

## A. The elevator pitch

### 1. 🟢 Explain your paper in two minutes.

> I studied whether stronger **training-time image augmentation** makes a small image classifier more
> robust to corrupted images, and what that costs in training time.
>
> I trained one model — MobileNetV3-Small, adapted for 32×32 CIFAR images, trained from scratch —
> exactly three times, changing **only** the augmentation: (A) random crop + horizontal flip, (B) plus
> RandAugment, (C) plus the full AugMix method, which uses three augmented views per image and a
> Jensen-Shannon consistency loss. Everything else was identical: same 50,000 training images, 50
> epochs, batch 128, SGD with cosine annealing, and the same two seeds.
>
> I measured clean test accuracy, accuracy on the CIFAR-10-C corruption benchmark (15 corruption
> types at 5 severities, 750,000 images per model), and training wall time, throughput, and peak GPU
> memory on one GTX 1650.
>
> **The result:** clean accuracy was essentially identical across all three — 86.39 %, 86.18 %, and
> 86.02 %, a spread of 0.37 points, which is smaller than the seed-to-seed noise. But corruption
> accuracy differed enormously: 67.04 % for the baseline, 71.00 % for RandAugment, and 77.82 % for
> AugMix. And the cost differed enormously too — about 13, 19, and 121 minutes.
>
> So the honest conclusion is a trade-off, not a winner: AugMix buys about 11 points of corruption
> robustness for roughly 6–9× the training time, and essentially nothing in clean accuracy. And the
> timing numbers themselves are so noisy that I re-analysed the per-epoch logs and found the true
> augmentation overhead is probably closer to 6.6× for AugMix and only about 1.06× for RandAugment —
> the reported means are inflated by interference on my machine, not by the methods.
>
> It is a small exploratory pilot and I make no novelty claim — a 2026 paper already covers
> essentially this combination.

### 2. 🟢 What is the one-sentence takeaway?

> Stronger augmentation buys large gains in corruption robustness at little or no cost in clean
> accuracy, but the compute cost varies enormously — and the methods that gain the most robustness
> are the most expensive to train.

### 3. 🟡 What would you say is your actual contribution, given that it is not novel?

> A **fully reproducible, matched-setting, cost-reported measurement** on a lightweight classifier,
> with every artifact released: code, frozen protocol, per-epoch logs, all six checkpoints, per-run
> evaluation CSVs, aggregate tables, dataset checksums, and a dated deviation log. The value is that
> a reader can re-run it and audit every number — not that I discovered something new. I also
> re-analysed my own timing logs and found the noise is *episodic* in contiguous epoch blocks, which
> lets me give a better estimate of true augmentation overhead than my own paper's headline table.
> That is a small but genuine analytical finding.

---

## B. Fundamentals: what, why, how

### 4. 🟡 What problem are you actually solving?

> Object classifiers degrade badly when the input image is degraded — blur, noise, bad weather, or
> digital artefacts. This matters operationally, because real camera and network inputs are never
> clean. The training-time remedy is **data augmentation**: deliberately degrade the training images
> so the network learns features that survive degradation. The open practical question is *which*
> augmentation, and *what it costs*. Most comparisons report only the accuracy benefit and omit the
> compute bill, so practitioners cannot make an informed choice.

### 5. 🟡 Explain the intuition of data augmentation to a non-ML person.

> If a child only ever sees photographs of cats in good light, they learn to rely on lighting and
> sharpness. Show that child a blurry, dark, snowy photo and they fail — not because they are stupid
> but because nothing in their experience prepared them. Training-time augmentation means showing the
> child blurry, dark, snowy photos *during* learning, so those situations stop being surprising.
> A neural network is exactly the same: its first layers learn simple detectors, and those detectors
> are tuned to the statistics of the training data. Corrupted input pushes the image away from
> anything the network has seen.

### 6. 🟢 What is "training-time" versus "test-time" augmentation, and why does it matter here?

> Training-time augmentation is applied to images while the model is learning; it changes training
> cost. Test-time augmentation is applied at inference to average or vote over multiple predictions;
> it changes inference cost but not training. My study is entirely about **training-time**
> augmentation, so the cost I measure is a **training** cost. A consequence: with an identical
> architecture and identical inference preprocessing, my three models would have essentially
> **identical inference latency**. Reporting inference latency as a benefit of these augmentations
> would be a category error, and I deliberately did not report it as one.

### 7. 🟡 Why is measuring training time a legitimate research contribution rather than just engineering?

> Because for these methods the compute overhead is not negligible — up to an order of magnitude on
> my setup — and because compute is a real constraint for the target audience. Lightweight classifiers
> exist precisely because people train them on limited hardware. If a practitioner has a fixed
> training budget, "which augmentation gives the best robustness per unit of compute" is a genuine
> decision problem. Also, cost measurement is methodologically delicate — it depends on data-pipeline
> design, worker count, hardware, and system state — and my contribution includes showing *how badly*
> such measurements can mislead when system state is uncontrolled.

### 8. 🟢 What is a "pilot study" and why did you frame it that way?

> A pilot is a small, carefully-controlled feasibility measurement that establishes whether a larger
> study is worth doing, and it reports its own limits explicitly. I had a two-day deadline for a
> complete experimental paper with original results, on a single laptop GPU. A pilot framing was the
> honest one: it sets the reader's expectations correctly and prevents over-interpretation. My
> `pilot_protocol.md` even says explicitly, in writing and before I ran anything, that this is
> "a pilot/replication-style comparison, not a claim of publication-level novelty."

### 9. 🟡 Why MobileNetV3-Small? Why not ResNet-18 or a bigger model?

> Three reasons. **First**, the research question was explicitly about **lightweight** classifiers, so
> the model had to be a real mobile-oriented architecture, not a shrunken ImageNet model.
> **Second**, it is a documented reference implementation in torchvision, so the architecture is
> verifiable and my adaptation is auditable. **Third**, the closest prior work (Fadhilah et al. 2026)
> also used MobileNetV3-Small, so at least the architecture is comparable even though the training
> protocols are not. I agree that adding a second architecture is the most important extension, and I
> say so in the paper's limitations.

### 10. 🟡 What are depthwise separable convolutions and why do they matter here?

> A standard convolution computes every output channel from every input channel. A depthwise separable
> convolution splits that into (1) a depthwise convolution, where each input channel is convolved
> independently, and (2) a pointwise 1×1 convolution that mixes channels. The cost drops from roughly
> `H·W·C_in·C_out·K²` to `H·W·C_in·K²` plus `H·W·C_in·C_out` — a large saving for the parameter and
> FLOP counts. That is what makes the network "lightweight" in deployment. It is relevant to my work
> because the parameter count (1.53 M) and the stride-1 adaptation determine my absolute timings, and
  because "efficient" refers to the *deployed* model, not to how fast it trains.

### 11. 🟡 You used an "efficient" architecture, yet training took 2 hours. Isn't that contradictory?

> Yes, and this is a genuinely important point that the paper makes explicit. "Efficient" describes
> the **deployed inference** cost. It says nothing about training cost, and training cost also depends
> on the input resolution. My adaptation sets the first convolution's stride to 1 so that the first
> feature map is 32×32 instead of 16×16 — I had to, or a 32×32 CIFAR image loses too much information
> immediately. That **4× larger** early feature map makes every early convolution about 4× more
> expensive. So my model is a deliberately expensive-to-train, cheap-to-deploy model, and I report
> timings accordingly rather than hiding it.

---

## C. Datasets: CIFAR-10 and CIFAR-10-C

### 12. 🟢 Describe CIFAR-10.

> 60,000 32×32 colour images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse,
> ship, truck), split 50,000 train / 10,000 test. Each image is about 3 KB. Small, fast, and one of
> the standard benchmarks for lightweight-model work, which is why I used it.

### 13. 🟡 What exactly is CIFAR-10-C, and how many images does it contain?

> Created by Hendrycks and Dietterich (ICLR 2019). It is the CIFAR-10 **test** set with 15 corruption
> types applied at 5 severity levels. So per model: 15 × 5 × 10,000 = **750,000 corrupted images**,
> plus 10,000 clean = 760,000 evaluation images per checkpoint, and 4,560,000 forward passes across
> my six checkpoints.

### 14. 💣 Your CIFAR-10-C is derived from the test set. Doesn't that mean your clean and corrupted metrics are not independent?

> Correct, and this is worth being precise about. CIFAR-10-C is generated by applying corruptions to
> the **same** 10,000 CIFAR-10 test images I use for clean accuracy. So the clean score and the
> corruption scores are **not independent measurements** — they share underlying images. This does not
> invalidate anything, because I never treat them as independent evidence and never combine them into a
> single score. But it does mean that if clean accuracy is unchanged, the corruption metric is
> essentially the only thing carrying signal — which is exactly what happened in my results. It also
> means a method cannot "trade" clean for corrupted accuracy by simply learning to recognise the
> specific test images; there is nothing to learn, because no test image is ever trained on.

### 15. 🔴 Your archive has 19 corruption arrays. Why did you use only 15? Did you cherry-pick?

> Not cherry-picking — the opposite. The canonical, published CIFAR-10-C benchmark has **15**
> corruption types. The archive I downloaded from the official Zenodo record contains 19 arrays,
> because later releases added four extra corruptions: `gaussian_blur`, `saturate`, `spatter`, and
> `speckle_noise`. I restricted to the 15 for three reasons. **First**, comparability: using the
> canonical set is the only way my numbers can be compared against the published literature, including
> the 2026 paper I cite. **Second**, the 19-corruption version is not a single well-defined benchmark,
> so an aggregate over 19 types is ambiguous. **Third**, transparency: the exclusion is hard-coded as
> an explicit named set in my code, with a comment, and the evaluator **raises an error** if any of
> the 15 canonical files is missing — so I could not have silently dropped a canonical corruption. I
> report the 15-corruption scope explicitly in the paper. Using all 19 would be defensible if
> pre-declared, but it is not the standard benchmark and it would break comparability.

### 16. 🟡 Group the 15 corruptions and say which were hardest.

> - **Noise (3):** Gaussian noise, impulse noise, shot noise
> - **Blur (4):** defocus blur, glass blur, motion blur, zoom blur
> - **Weather (3):** fog, frost, snow
> - **Digital (5):** brightness, contrast, elastic transform, JPEG compression, pixelate
>
> The ordering was **identical for all three of my methods**: noise hardest, brightness easiest. For
> the baseline, Gaussian noise was worst at 47.75 % and brightness best at 84.50 %. This matches the
> published finding that high-frequency perturbations such as noise are the hardest corruption family
> — consistent with Yin et al. (2019), who analysed corruptions in the Fourier domain.

### 17. 🔴 Why don't you report mCE, the benchmark's standard metric?

> mCE — mean Corruption Error — is the CIFAR-10-C paper's normalised metric, which divides each
> model's error by a reference model's error (AlexNet and ResNet-50 in the original paper) to make
> scores comparable across model scales. I report **plain top-1 accuracy** instead. The honest reasons:
> I did not run the reference models needed to compute the normalisation, and adding them was outside
> the pilot's compute budget. The trade-off is that my absolute numbers are not directly comparable to
> mCE-based tables in other papers. This is a real limitation and I list it. I would add it as a
> follow-up — it is a small amount of compute for a real gain in comparability.

### 18. 🟡 Why did you not use CIFAR-10.1?

> CIFAR-10.1 is a **separately collected** test set — new images, not the original CIFAR-10 test
> images — and it exists precisely to test whether a CIFAR-10 model generalises beyond the
> distribution it was validated on. It was in my original plan as a generalisation check, and I had to
> drop it to fit the two-day deadline. It is the **most important missing experiment** in the project,
> because Mintun et al. (2021) showed that augmentation-induced gains on synthetic corruptions often do
> **not** transfer to real, naturally collected data. Without CIFAR-10.1 I cannot rule out that my
> AugMix gain is partly benchmark-specific. I say this in the paper.

### 19. 🟡 Did you use a validation split?

> No — in the pilot I trained on all 50,000 training images. My original plan proposed a
> 45,000/5,000 stratified train/validation split, and I dropped it to fit the deadline; that is a
> recorded deviation. Two things make this defensible rather than sloppy. **First**, there is **no test
> leakage**: I never tuned anything on the test sets, I used the **final-epoch** checkpoint rather
> than a best-epoch checkpoint selected on any held-out data, and all augmentation hyperparameters
> were frozen in a dated protocol before any evaluation. **Second**, it makes the comparison stricter,
> since all three conditions see exactly the same 50,000 images. But the honest cost is that I had no
> early stopping and no model selection of any kind, and I could not have detected overfitting. I
> recommend restoring the split in any follow-up.

### 20. 🟡 How did you verify the data was correct?

> **Checksums.** I downloaded CIFAR-10-C from the official Zenodo record and verified its MD5,
> `56bf5dcef84df0e2308c6dcbcbbd8499`, against the value published on that record. **Structural checks
> in code:** the evaluator asserts that each corruption array's length equals the labels length *and*
> is divisible by 5, which structurally guarantees five equal severity blocks; it raises an error
> listing any missing canonical corruption file; and the 10-class output layer matches CIFAR-10's 10
> classes. **Not done, and I would do it next:** I did not audit for duplicate images within CIFAR-10
> or between the train split and CIFAR-10-C.

### 21. 🟡 Why extract CIFAR-10-C *after* training rather than before?

> A deliberate choice to protect the timing measurement. Extracting a multi-gigabyte archive is heavy
> disk I/O, and I was measuring wall-clock training time on a laptop. Extracting beforehand would have
> contaminated the timing with disk contention. I logged this decision in the deviation log. It is a
> small thing, but it is the kind of detail that separates a careful measurement from a casual one.

---

## D. The model: MobileNetV3-Small

### 22. 🟢 Why did you have to modify MobileNetV3-Small? What did you change?

> Two changes. **First**, the first convolution's stride from 2 to 1. MobileNetV3 is designed for
> 224×224 ImageNet inputs, where immediate downsampling is efficient. For a 32×32 image, stride 2
> would produce a 16×16 first feature map and throw away too much. **Second**, I replaced the
> 1,000-class classifier with a 10-class linear layer to match CIFAR-10. The result has **1,528,106**
> trainable parameters, which I verified directly in PyTorch.

### 23. 🔴 What is the compute consequence of the stride change, and does it bias your timing comparison?

> It makes the early feature maps 4× larger in area — 32×32 instead of 16×16 — so early and middle
> convolutions cost roughly 4× more. That inflates my **absolute** times substantially. It does **not**
> bias the **comparison** between conditions, because the architecture is byte-identical across all
> three, so the effect is a common multiplier. But it does interact with the comparison in one way
> worth naming: AugMix pushes 3× the images through the model, so it eats 3× the inflated per-image
> cost. So the stride-1 adaptation amplifies AugMix's disadvantage in absolute terms. A stride-2 stem
> would be dramatically cheaper to train and is a natural thing to test — but it throws away half the
> spatial resolution, so there is a genuine accuracy/compute trade-off there that I did not explore.

### 24. 🔴 Did you fine-tune pretrained weights or train from scratch? Why?

> **From scratch** — `weights=None` in torchvision, which guarantees no pretrained weights leak in.
> Three reasons. **First**, fairness of the comparison: all three conditions start from identical
> random initialisations, so any difference is attributable to augmentation. **Second**, ImageNet
> features are known to be fragile to noise and blur corruptions, so pretrained features could have
> *masked* the effect I was trying to measure. **Third**, comparability of protocol: the closest prior
> work fine-tunes pretrained weights, which means our absolute numbers are not comparable anyway —
> from-scratch is a cleaner, self-contained starting point. I acknowledge that this puts my absolute
> accuracies below published fine-tuned numbers, and that it is a different regime.

### 25. 🟡 Is 86 % clean accuracy good or bad for MobileNetV3-Small on CIFAR-10?

> Reasonable, not exceptional — and the honest framing matters. In the literature, well-tuned
> CIFAR-scale ResNet-family models reach roughly 92–95 % with 200 epochs and a full recipe, and
> from-scratch lightweight models typically land in the mid-80s to low-90s. I got 86 % in **50 epochs**
> with **library-default** augmentation hyperparameters and a deliberately cheap model adaptation. So
> 86 % is consistent with a short, untuned schedule rather than evidence of a weak model. The
> important point for my argument is that **all three conditions land within 0.37 points of each
> other**, so the clean-accuracy comparison is uninformative either way — the action is all in the
> corruption metric.

### 26. 🟡 Why Small and not Large?

> Small is the more aggressive efficiency claim — it is the architecture where compute cost actually
> matters, which is the point of the research question. Large would have doubled the per-image cost
> and made the timing comparison even noisier relative to the signal. Small also matches the prior
> work I cite. If I had more compute I would test both, since "does the augmentation ranking depend on
> model capacity?" is an open and interesting question.

### 27. 🟡 You report 1,528,106 parameters. How do I know that is right?

> I instantiated the model in PyTorch and counted `p.numel()` over all parameters, both for the
> unmodified torchvision model (2,542,856 for 1,000 classes) and for my modified version. The count is
> also written automatically into every `config.json` and `summary.json` by the training script, so it
> is machine-recorded rather than typed by me. I can show you the check.

### 28. 💣 Did you accidentally use pretrained weights when you meant to train from scratch?

> No. The call is `torchvision.models.mobilenet_v3_small(weights=None)`, which explicitly requests
> random initialisation — passing `weights=None` is the safeguard, because omitting the argument in
> older torchvision versions could trigger a download. The evaluator reloads checkpoints with
> `load_state_dict(..., weights_only=True)` into a freshly constructed random model, so evaluation
> cannot silently substitute pretrained weights. I also released the exact training command in the
> README, so the claim is checkable rather than something you have to take on trust.

---

## E. The three augmentation methods

### 29. 🟢 What is RandAugment?

> An automated augmentation method that removes the expensive *policy-search* phase that earlier
> methods like AutoAugment required. Instead of learning a policy, it samples a small fixed number of
> operations uniformly at random from a candidate set, at a chosen strength. The original insight was
> that a well-chosen fixed policy and a random policy perform comparably, while the random one costs
> nothing to find. That makes it attractive when compute is limited — which is exactly my situation.

### 30. 🟢 What is AugMix?

> It has **two** parts, and conflating them is the most common misunderstanding. **Part 1, the
> transform:** build several chains of random operations, mix their outputs with weights drawn from a
> Dirichlet distribution, producing one image with several mild corruptions layered together.
> **Part 2, the objective:** create multiple augmented views of the same source image, and add a
> Jensen-Shannon divergence penalty that forces the network to give **similar predictions across those
> views**. Part 2 is the actual robustness mechanism — it teaches the network that a photo and a
> degraded version of it are the same object.

### 31. 🔴 Did you implement the *full* AugMix method, or just the augmentation transform?

> **The full method**, and this is the point I would most want a professor to verify.
>
> The transform alone is just "make a messy image." The method is the transform **plus** a
> multi-view consistency objective. Specifically, for each source image I generate **three views**:
> one crop-and-flip view, and **two independently generated** AugMix views. All three go through the
> network in a single concatenated forward pass. The loss is the mean cross-entropy across the three
> views, **plus 12 × the Jensen-Shannon divergence** between the three prediction distributions.
>
> The consistency term is what forces damage-invariance, and it is why my AugMix result is qualitatively
> different from RandAugment's rather than just "a bit more augmentation." I explicitly noted in my
> own project plan that describing a single AugMix transform as "the full method" would be wrong, so I
> checked this deliberately.

### 32. 🔴 Write out the AugMix loss and explain every term.

> ```
> loss = ( CE(view₁, y) + CE(view₂, y) + CE(view₃, y) ) / 3
>      + 12 · JSD(p₁, p₂, p₃)
>
> where JSD = [ KL(p₁‖M) + KL(p₂‖M) + KL(p₃‖M) ] / 3,   M = (p₁+p₂+p₃)/3
> ```
>
> - `CE` is standard cross-entropy — the ordinary supervised classification loss.
> - `p₁, p₂, p₃` are the softmax output distributions of the three views.
> - `M` is their pointwise mean, the "consensus" prediction.
> - `KL(pᵢ‖M)` measures how far view *i* deviates from the consensus.
> - Averaging the three KL terms gives the Jensen-Shannon divergence.
> - **Why JSD and not KL?** JSD is symmetric and, crucially, **bounded in [0, ln 2]**. KL divergence is
>   unbounded, so a badly disagreeing view could dominate the loss and destabilise training. Boundedness
>   is exactly what makes a regulariser safe to weight heavily.
> - **The weight 12** comes from the original AugMix paper's CIFAR/ImageNet recipes. I did not tune it.

### 33. 🔴 How did you implement the JSD numerically?

> Three steps in code: apply `log_softmax` to each set of logits to get log-probabilities, exponentiate
> to get probabilities, average the three probability tensors to get `M`, then compute
> `F.kl_div(log(M), p, reduction="batchmean")` for each view and average the three results. Two
> details worth noting. `reduction="batchmean"` sums over classes and divides by batch size, which gives
> the correct per-example KL that then averages over the batch. And I clamp `log(M)` with
> `clamp_min(1e-7)` before taking the log, so a near-zero probability cannot produce `log(0)` and a
> NaN. That clamp is a real numerical-stability concern with softmax outputs, and it is a cheap guard.

### 34. 🔴 Where exactly is the JSD in your code, and is the loss really the average of three cross-entropies?

> In `src/train.py`, the training loop has an `if args.method == "augmix":` branch. There, the three
> views are moved to the GPU, concatenated along the batch dimension into one 384-image tensor, passed
> through the model in a single call, then split back apart with `logits.split(batch_n, dim=0)`. The
> loss is the **mean** of the three `F.cross_entropy` terms, plus `12.0 * jensen_shannon(...)`.
>
> One implementation note worth mentioning: I concatenate rather than calling the model three times.
> That avoids three separate kernel-launch sequences and gives the GPU a larger batch to work on,
> which matters on a small GPU. The reported timing includes this, so the comparison stays honest.

### 35. 🟡 What are the exact augmentation hyperparameters, and did you tune them?

> **I did not tune anything.** All values are either torchvision defaults or taken from the original
> papers:
> - `RandAugment(num_ops=2, magnitude=9)` — both are torchvision's defaults. `num_ops=2` means exactly
>   two operations per image, sampled from 14 candidates. `magnitude=9` on a 0–30 scale
>   (`num_magnitude_bins=31`) is a fairly aggressive setting, and roughly half the operators get a
>   random sign, so shears and rotations go either way.
> - `AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)` — all torchvision defaults.
>   `mixture_width=3` is three chains; `chain_depth=-1` means depth is **randomly sampled from 1 to 3**;
>   `alpha=1.0` means Dirichlet(1), which is uniform on the simplex, so all weightings are equally
>   likely a priori. JSD weight 12 is from the AugMix paper.
>
> **Why this is a feature and a limitation.** Using library defaults and published values means the
> comparison is not contaminated by tuning on the test set — a real methodological strength. But it
> also means this compares *default-configured* policies, not *optimally tuned* ones. A professor could
> fairly say the comparison under-serves RandAugment and AugMix. I say this in the paper's limitations
> and it is a priority follow-up.

### 36. 🟡 Why use the torchvision implementations instead of writing your own?

> Because it makes my "method" the **published** method rather than my reimplementation of it. If a
> professor wants to check whether I implemented AugMix correctly, they can read the reference
> implementation rather than auditing my code and wondering if I got it wrong. This is a strong
> internal-validity argument: the comparison is between the standard implementations at documented
> settings, under one controlled harness. The trade-off is that I am partly measuring torchvision's
> implementation efficiency, which I return to in the timing section.

### 37. 🟡 What is the difference between RandAugment and AugMix conceptually?

> **Randomness of composition.** RandAugment applies a small number of operations **in sequence** to
> one image, all at one magnitude. AugMix applies several operations **in parallel chains**, mixes
> their outputs, and then enforces **agreement across multiple views** of the same image with an
> explicit loss term.
>
> That is the crux: RandAugment is a purely **data-side** intervention — it changes what the model
> sees. AugMix is **data-side plus objective-side** — it also changes what the model is *penalised for*.
> That is why I expected AugMix to win more on corruptions, and why I was careful to also note that
> its extra loss term is a potential confound (§G, Q50).

### 38. 🟡 Is `RandomCrop(32, padding=4)` a real augmentation, or does it just add black borders?

> It adds 4 pixels of padding on each side, producing a 40×40 image, then takes a random 32×32 window.
> So the model sees the object at slightly different positions and scales within the frame. It is a
> standard, cheap translation/scale augmentation. It is a *weak* augmentation compared to RandAugment
> or AugMix, which is exactly why it makes a useful baseline: it isolates how much of the robustness
> gain comes from the sophisticated methods rather than from "any augmentation at all."

### 39. 🟡 All three conditions include crop-and-flip. Why?

> So the comparison is **additive and controlled**. Each stronger method is the baseline *plus* one
> addition, so any difference is attributable to that addition. If I had compared, say, plain
> ResNet-style augmentation against AugMix with nothing in common, I would not know whether the
> difference came from AugMix or from having crop-and-flip at all. This is a controlled-baseline
> design, and I think it is the single most important design decision in the study.

### 40. 🟡 Why is Normalize applied *after* the augmentation and not before?

> Because the augmentation operators are defined on image-like inputs. `RandomCrop` and
> `RandomHorizontalFlip` are geometric and would work either way, but photometric operations inside
> RandAugment and AugMix — brightness, contrast, posterize, equalize, solarize — are defined on
> 0–255 image values. Normalising first would mean those operators act on zero-mean, unit-variance
> values where "posterize" and "solarize" have no sensible meaning. So the order is
> `PIL uint8 → geometric → photometric → ToTensor (÷255, HWC→CHW) → Normalize`. I use the standard
> CIFAR-10 channel statistics: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616).

### 41. 💣 AugMix's "clean view" is not actually clean. Is that misleading?

> It is worth being precise, and I accept the criticism of the *wording*. The "clean view" is the
> crop-and-flip image, so it is geometrically augmented but not photometrically damaged. "Clean" in my
> usage means "not AugMix-processed", which is the same usage as the original AugMix paper, where the
> clean view anchors the consistency objective. It is not a pristine, unaugmented image. I would not
> describe it as "the original image" in an interview.

### 42. 🟡 Did you use test-time augmentation when evaluating?

> No. Evaluation is fully deterministic: `model.eval()`, `torch.inference_mode()`, no random
> transforms, no multi-crop, no flip averaging. Every reported number comes from a single deterministic
> forward pass per image. Test-time augmentation would have conflated the training-time augmentation
> with an inference-time technique, which is not what I set out to measure.

---

## F. Training setup and hyperparameters

### 43. 🟢 List your exact training configuration.

> - Optimiser: **SGD**, initial learning rate **0.05**, momentum **0.9**, weight decay **0.0005**
> - Schedule: **cosine annealing** over the full run, `CosineAnnealingLR(T_max=50)`
> - Epochs: **50** · Batch size: **128** · Data-loader workers: **2**
> - Seeds: **17 and 23** for every condition
> - Augmentation: as described, JSD weight 12 for AugMix
> - Checkpoint used: **final epoch only**
> - Identical for all three conditions — only the augmentation differs

### 44. 🟡 Why SGD and not AdamW, which is more common now?

> SGD with momentum and weight decay is the **standard recipe for the CIFAR-10 and CIFAR-10-C
> literature**, including the AugMix paper and the closest prior work. Using AdamW would have made my
> numbers incomparable to the published results I am trying to sit alongside. For a 50-epoch
> from-scratch run, SGD with cosine annealing is also a well-understood, well-behaved choice. I would
> not say SGD is better — only that it is the right choice for comparability.

### 45. 🟡 Why learning rate 0.05 with batch size 128? Isn't that high?

> It is on the higher side, but it is a standard CIFAR-scale setting: with batch 128, SGD at 0.05 with
> cosine annealing is a very common recipe, and the cosine decay to near zero by epoch 50 lets it
> settle. The risk of a high initial LR is under-training on a 50-epoch budget, which is a real
> limitation of my schedule — and I would not be surprised if a lower LR with more epochs did better.
> Again, the important point for my argument is that the value is **identical across all three
> conditions**, so it cannot explain the differences between them.

### 46. 🟡 What do momentum and weight decay actually do?

> **Momentum 0.9** accumulates 90 % of the previous step's direction into the next one. This averages
> out noisy per-batch gradients, smoothing the trajectory and letting a larger learning rate be stable.
> **Weight decay 0.0005** adds an L2 penalty on the weights, which discourages large weights and acts
> as a mild regulariser. Both are standard, and both are held fixed across conditions.

### 47. 🟡 Why cosine annealing? What would a constant LR have done?

> Cosine annealing decays the learning rate along a half-cosine curve from 0.05 to approximately zero
> over the 50 epochs. The intuition: take large steps early to make fast progress, then progressively
> smaller steps to settle into a minimum. A constant LR would keep bouncing around the minimum and
> typically land 1–2 points worse. I chose cosine because it is the standard CIFAR recipe and because
> it is defined by `T_max = number of epochs`, so it needs no tuning — one fewer thing that could
> differ between conditions.

### 48. 🔴 Was 50 epochs chosen by looking at test results? How do I know?

> No. The sequence was: I ran **one-epoch feasibility pilots on seed 99** purely to estimate runtime
> and peak memory, then **froze** 50 epochs / batch 128 / 2 workers in a written, dated protocol
> (`pilot_protocol.md`, version 0.1, 2026-09-25), and only then ran the six final runs and evaluated.
> The pilots are stored separately in `results/runtime_pilot/` and are **excluded from every accuracy
> analysis** — they are not merged into the aggregate tables. The evidence is structural: the protocol
> document, with its date, precedes the results, and the pilot outputs live in a different directory
> that the summariser never reads for accuracy.

### 49. 🔴 How do I know you didn't tune hyperparameters on the test set?

> Four independent safeguards.
> **First**, every augmentation hyperparameter is a **torchvision default or a value from the original
> paper** — I ran no search at all, so there was nothing to tune. **Second**, the settings were written
> into a dated, frozen protocol before evaluation. **Third**, I used the **final-epoch** checkpoint
> rather than selecting the best epoch on test data, which is the most common form of test-set leakage
> in augmentation studies. **Fourth**, I ran **no validation split and no early stopping**, so there was
> no mechanism by which test data could influence training.
>
> I also recorded a deviation log with seven dated entries, including every timing anomaly, so the
> history of decisions is auditable.

### 50. 🔴 You used the same two seeds for all conditions. Is that a problem or a benefit?

> It is a deliberate benefit, and it is the reason my design is stronger than a naive one. Because
> seeds are **paired** across conditions, I can compare conditions on the *same* random initialisation
> and the *same* data ordering. That removes initialisation and data-order variance as a confound. A
> design with different seeds per condition would confound "method difference" with "seed difference".
>
> The limitation is statistical: with n = 2, a paired test still has 1 degree of freedom and almost no
> power. So pairing gives me a cleaner design but does not rescue me from the small sample size.

### 51. 🟡 Why only two seeds? That seems very few.

> Compute. The six 50-epoch runs already took 5.12 hours of training on my single GTX 1650 — AugMix
> alone was 4.05 hours — and I had a two-day deadline that also had to cover the literature review,
> 4.56 million evaluation forward passes, the manuscript, and the release. AugMix at 50 epochs is
> roughly 2 hours per seed, so going from 2 seeds to 5 would have cost about 6 extra hours of training
> alone. I chose to guarantee a complete, honest, fully released study over a larger-n study that would
> not have finished. Two seeds is genuinely too few, I say so explicitly, and I report mean and SD as
> **descriptive only** with no significance claims.

### 52. 🟡 What was seed 99 for?

> A **one-epoch feasibility pilot** for each of the three conditions, run before the final schedule was
> frozen, used only to estimate runtime and peak memory. It is explicitly excluded from all accuracy
> comparisons — it lives in `results/runtime_pilot/` and the final analysis only reads
> `results/final/`. Its final-epoch training accuracies (36.4 %, 30.9 %, 35.9 %) are meaningless as
> accuracy results because they are after a single epoch.

### 53. 🔴 Your one-epoch pilot showed RandAugment was FASTER than the baseline. But in the full runs it was slower. Why? Did you drop the inconvenient pilot?

> No, and this is worth explaining carefully because it is a good illustration of warm-up effects. In
> the pilots, the baseline took 42.2 s and RandAugment 20.9 s. That is a **warm-up artefact**: the
> baseline was the very first thing ever run on that GPU in that process, so it paid one-time cuDNN
> kernel initialisation, algorithm selection, and memory-pool growth. RandAugment ran second and
> benefited. In the 50-epoch runs, the effect washes out and RandAugment is consistently slower — its
> median epoch is 14.5 s against the baseline's 13.8 s.
>
> I kept the pilot results, recorded the anomaly in the deviation log, and explained it rather than
> quietly dropping the inconvenient number. Note also that the per-epoch logs show it directly: the
> baseline seed-17 run has exactly **one** slow epoch, epoch 1 at 39.7 s, and then 49 normal ones.

### 54. 🟡 Why did you set the model to refuse to run on CPU?

> Because the whole point of the timing measurement is GPU execution. If CUDA were unavailable and the
> script silently fell back to CPU, I would have recorded a CPU wall time and reported it as a GPU
> result — a silent, catastrophic measurement error. The script raises an explicit error instead. Same
> reasoning in the evaluator. I also verified CUDA and the GTX 1650 were genuinely usable before
> starting, and recorded the GPU name and total memory in every run's config.

### 55. 🟡 Your peak memory numbers are identical for baseline and RandAugment. Is that a copy-paste error?

> No, and it is expected. `torch.cuda.max_memory_allocated` reports **peak allocated tensor memory**,
> which is driven by the model, the batch size, and the activations — and **not** by how much CPU work
> the augmentation does. Baseline and RandAugment both push 128 images through in one batch with the
> same architecture, so their activation footprint is identical. AugMix's is 2.71× larger
> (557,116,928 vs 205,514,240 bytes) because it pushes 384 images per step, roughly 3× the
> activations, and it is slightly less than 3× because the peak is not perfectly linear. That the
> augmentation's *CPU* cost does not appear in a *GPU memory* metric is exactly why wall-clock time
> and memory are separate metrics and must be reported separately.

### 56. 🟡 What is the difference between peak *allocated* and peak *reserved* GPU memory?

> **Allocated** is the bytes actually held by live tensors — the working set. **Reserved** is what
> PyTorch's caching allocator has claimed from the CUDA driver, which is larger because it keeps
> cached blocks to avoid expensive re-allocation. Reserved is the number closer to what `nvidia-smi`
> shows for the process. I report **allocated**, which is the more conservative and more comparable
> number. Neither includes the CUDA context itself, so neither is "total GPU usage" — and I never
> present it that way.

### 57. 🟡 How did you make training reproducible?

> Six measures. I seeded Python's `random`, NumPy, Torch, and CUDA from the run seed. I set
> `cudnn.deterministic = True` **and** `cudnn.benchmark = False` — this pairing matters, because
> `benchmark = True` selects convolution algorithms *by timing them*, which would make runs differ
> from each other. I gave the DataLoader its own `torch.Generator` so shuffling is reproducible and
> independent of global RNG state. I added a `worker_init_fn` to seed NumPy and Python's `random` in
> each worker — without it, all workers inherit the same default RNG state and produce **correlated**
> augmentations, a subtle and real bug. And I used `persistent_workers=True` so workers are not
> re-created every epoch.

### 58. 🟡 Is your training exactly bitwise reproducible?

> On the **same** GPU model, driver, and library versions, it should be close to bitwise, given the
> determinism settings. Across different hardware it is not guaranteed, because cuDNN may select
> different convolution algorithms for different devices. And I should be clear that **wall-clock
> timings are not reproducible at all** — that is not a flaw in my setup, it is my main empirical
> finding. One thing I did **not** do is enable `torch.use_deterministic_algorithms(True)`, which would
> enforce stricter guarantees at some performance cost; that is a reasonable addition for a follow-up.

---

## G. Results and how to interpret them

### 59. 🟢 Give me your headline numbers.

> | Condition | Clean accuracy | CIFAR-10-C accuracy | Training time |
> |---|---:|---:|---:|
> | Crop-and-flip | **86.39 ± 0.25 %** | 67.04 ± 1.63 % | 13.2 min |
> | RandAugment | 86.18 ± 0.82 % | 71.00 ± 0.06 % | 19.1 min |
> | AugMix | 86.02 ± 0.01 % | **77.82 ± 0.31 %** | 121.4 min |
>
> Means ± sample SD over seeds 17 and 23. CIFAR-10-C is the unweighted mean over 75
> corruption-severity cells.

### 60. 🔴 Is the clean-accuracy difference real? Which method is best on clean images?

> **No, the difference is not real, and I do not claim a winner.** The three means span 0.37
> percentage points (86.39, 86.18, 86.02), which is **smaller than the seed-to-seed spread within a
> single condition** — RandAugment's own SD is 0.82 points. The decisive evidence that the ordering is
> noise: RandAugment produced **both** the single highest clean accuracy of all six runs (86.76 %,
> seed 17) **and** the single lowest (85.60 %, seed 23). If the ranking were meaningful, that would not
> happen.
>
> The correct conclusion is that under this 50-epoch schedule all three policies reach equivalent clean
> accuracy, and the interesting differences are entirely in the corruption metric and the cost.

### 61. 🔴 Is AugMix's 10.78-point CIFAR-10-C gain statistically significant?

> I cannot answer "yes" with a significance test, because **n = 2 gives 1 degree of freedom**. No test
> is defensible here, and I did not run one.
>
> What I can say is that the effect is very large relative to the observed noise. The gap is 10.78
> points. The baseline's own seed-to-seed SD is 1.63 points, so the gap is about **6.6× the baseline's
> SD**, and AugMix's own SD is only 0.31 points, making the gap about **35× that**. The largest
> single-seed difference between any two AugMix runs is 0.31 points. A gap that large, that stable,
> is very unlikely to be a seed artefact. But "very unlikely" is not a p-value, and I present it as a
> descriptive observation rather than an inferential claim.

### 62. 🟡 Why is the baseline's seed-to-seed SD (1.63) so much larger than AugMix's (0.31)?

> An interesting observation rather than something I set out to measure. AugMix averages predictions
> over three views during training, and its consistency loss explicitly penalises sensitivity to input
> perturbation. So it seems plausible that heavy augmentation plus a consistency objective produces a
> **flatter, more stable** solution — less dependent on the particular initialisation and data order.
> The same pattern holds for RandAugment (SD 0.06), which is also heavily augmented.
>
> I want to be careful here: with n = 2 this is a suggestive pattern, not a finding, and I have not
> tested the mechanism. It would be a clean thing to investigate with more seeds.

### 63. 🔴 AugMix got 3× the model compute. Is its win just because it did more training? This is a confound.

> **Yes, and this is the most important limitation in the study.** AugMix performs three forward and
> backward passes' worth of gradient signal per source image, and its loss contains three cross-entropy
> terms plus the JSD term, versus one for the others. So the comparison is **policy-matched but not
> compute-matched**. Part of AugMix's advantage may come from receiving more gradient signal per
> source image rather than from a better augmentation distribution.
>
> The decisive missing experiment is a **compute-matched control**: train the baseline for three times
> as long, or at three times the throughput, at the same wall-clock budget. If the baseline at 3×
> compute closes the gap, then AugMix's advantage is mostly compute. If it does not, the augmentation
> and consistency objective are genuinely doing the work. I did not have the compute to run this, and
> I list it as the top follow-up in the paper.

### 64. 🔴 Your augmentation parameters were not tuned. Doesn't that bias against RandAugment and AugMix?

> It could, and I flag it explicitly as a limitation. `magnitude=9` and `severity=3` are library
> defaults, not values I confirmed are optimal for MobileNetV3-Small on CIFAR-10. The original RandAugment
> paper found that the best strength is dataset- and architecture-dependent. A carefully tuned
> RandAugment might close some of its 3.96-point gap.
>
> There is a real trade-off here, and I think it is worth being honest about both sides. Tuning each
> method to its own optimum is the fairest possible comparison, but it requires a **validation split**
> — and tuning on the test set would be far worse than not tuning at all. My choice of untuned defaults
> is the leakage-free option. The right follow-up is to restore the 45,000/5,000 split and report
> **both**: a default-configured comparison and a best-tuned comparison.

### 65. 🔴 Show me a case where a stronger augmentation made things WORSE.

> **RandAugment on pixelate**: 70.12 % versus the baseline's 71.74 % — **1.62 points worse**, and the
> only case in the entire 15-corruption table where a stronger method lost to the baseline.
>
> There is also a near-tie: on brightness, AugMix scored 84.56 % versus the baseline's 84.50 % — a
> 0.06-point difference, which is noise. These are the results that stop "stronger augmentation is
> always better" from being a defensible summary of my data, and they are consistent with Yin et al.
> (2019), who found that the effect of an augmentation depends on the corruption family.

### 66. 🔴 Which corruption families benefited most, and can you explain why?

> Looking at AugMix's gain over the baseline by family:
> - **Noise: +18 to +23 points** (Gaussian +23.06, impulse +20.04, shot +18.40) — by far the largest
> - **Blur: +9 to +17 points** (glass +16.94, motion +14.24, zoom +13.62, defocus +9.27)
> - **Weather: +4 to +8** (frost +7.73, snow +5.41, fog +4.19)
> - **Digital: +0.06 to +11.95** (contrast +11.95, pixelate +9.45, elastic +4.37, JPEG +2.99,
>   brightness +0.06)
>
> Two observations. **First**, the baseline is weakest exactly where the gain is largest — Gaussian
> noise 47.75 % for the baseline — so part of the effect is simply more headroom. **Second**, and more
> interesting, the operations inside AugMix's chains — solarize, posterize, contrast, equalize,
> autocontrast, sharp geometric warps — are qualitatively extreme versions of noise, blur, and
> contrast corruptions. So AugMix is being trained on inputs that resemble the *test-time* corruption
> distribution more than crop-and-flip ever could. That is a mechanism-level story consistent with the
> AugMix paper, though I did not verify it causally.

### 67. 🔴 What is the single most interesting number in your paper?

> The **severity-5 row**. Mean accuracy at the most extreme corruption level: baseline 52.17 %,
> RandAugment 57.12 %, AugMix **69.67 %**. AugMix loses only 13.62 points going from severity 1 to 5,
> while the baseline loses 27.60 — less than half the degradation.
>
> The point is that the advantage **grows with severity**: AugMinus−baseline is +3.52 points at
> severity 1 but +17.5 points at severity 5. So AugMix is not just shifting the whole curve up; it
> changes the **shape** of degradation. In deployment terms, the difference between 52 % and 70 %
> accuracy on a badly degraded image is far more consequential than a 3-point difference on a mildly
> degraded one.

### 68. 🟡 How exactly do you compute the CIFAR-10-C aggregate?

> For each checkpoint, the unweighted mean of the top-1 accuracies of all **75 cells** — 15 corruption
> types × 5 severity levels. Each cell is 10,000 images, so the per-checkpoint aggregate is over
> 750,000 images. Then I average across the two seeds. I use a plain unweighted mean because the
> standard deviation step (`statistics.stdev`) is cleanest on a flat list of per-run aggregates.
> The weighting does not matter much here because every cell has exactly the same size.

### 69. 💣 Your reported training losses are not comparable across methods. Why?

> Because for AugMix the logged loss **includes the 12 × JSD consistency term**, so it is on a
> completely different scale from a plain cross-entropy. The epoch log records the total `loss` that was
> back-propagated. Never compare AugMix's loss curve to the baseline's — it is comparing a
> cross-entropy against a cross-entropy plus a large regulariser.
>
> A related trap: the logged **training accuracy** is also not comparable, because it is computed on
> *different input distributions*. For baseline and AugMix it comes from the crop-and-flip view; for
> RandAugment it comes from RandAugment-damaged images. RandAugment's lower "training accuracy"
> (83.8 % / 82.6 % vs 88.8 %) is largely a property of *what it was measured on*, not evidence that it
> learned less — its clean test accuracy is statistically identical to the others. I would not present
> these three training-accuracy numbers as a comparison.

### 70. 🟡 What was your best single result, and your worst?

> **Best:** AugMix's CIFAR-10-C mean of 77.82 % — 10.78 points above the baseline, with the smallest
> seed-to-seed SD of any corruption result (0.31 points). A large effect that is also highly
> reproducible.
>
> **Worst:** the timing measurement, honestly. RandAugment's SD (586 s) is **larger than the entire
> mean difference** from the baseline (351 s). That comparison is simply not interpretable, and I say
> so in the paper rather than presenting 1.44× as a finding. A measurement I cannot trust is worth
> less than no measurement, and the right response was to report it and flag it, which is what I did.

### 71. 🔴 If you had to run only ONE condition, which, and why?

> It depends entirely on the constraint, which is the real finding of my paper.
> - **Unconstrained compute:** AugMix. +10.78 points of corruption robustness for −0.37 points of
>   clean accuracy is an excellent trade.
> - **Tight compute, or the corruption is mostly blur/weather:** RandAugment. It captured about a
>   third of AugMix's robustness gain (3.96 of 10.78 points) for a genuinely small overhead —
>   probably around 6 % once the timing noise is removed from the equation.
> - **Deployment is clean-image only:** crop-and-flip. All three are equivalent on clean accuracy, so
>   take the cheapest.
>
> The point is that "which augmentation is best" is the wrong question. "Which augmentation is best
> **for my accuracy-per-FLOP constraint**" is the right one, and that is why I measured cost at all.

### 72. 🟡 Can you draw a trade-off curve or compute a Pareto frontier?

> I can reason about it qualitatively from three points, but with three conditions I cannot fit a
> meaningful curve. What I can say is that crop-and-flip and RandAugment are both **dominated** in a
> strict sense: AugMix is better on corruption accuracy and only 0.37 points worse on clean accuracy,
> so if compute were free, both would be dominated. Compute is therefore the **only** reason to choose
> anything but AugMix, and quantifying that trade properly needs either a compute-matched control or
> many more conditions. That is a limitation, and plotting three points as a frontier would overstate
> what the data supports.

---

## H. Timing and performance (the hard one)

### 73. 🟢 How did you measure training time?

> With `time.perf_counter()`, starting immediately before the first epoch's loop and stopping after the
> last, with an explicit `torch.cuda.synchronize(device)` immediately before stopping the timer. That
> synchronise is the critical detail — CUDA operations are asynchronous, so without it the timer would
> stop while GPU work was still queued and I would **under-report** the time. The measurement boundary
> **includes** data loading and online augmentation, and **excludes** dataset download, model
> construction, and evaluation. Per-epoch timings are written to `epochs.csv` as they happen.

### 74. 🔴 Your timing numbers are not credible. RandAugment took 2.14× as long as the baseline seed under identical settings. What's going on?

> You are right, and the paper says exactly this. I logged it as a deviation and drew no compute-cost
> conclusion from the means. But I went back to the per-epoch logs, and the structure is informative.
> The noise is **episodic, in long contiguous blocks**, not random jitter:
>
> | Run | Median epoch | Epochs much slower than median | Pattern |
> |---|---:|---|---|
> | baseline seed 17 | 13.8 s | 1 (epoch 1: 39.7 s) | clean; one-time warm-up |
> | baseline seed 23 | 13.5 s | 11 (epochs 1–11: ~32 s) | slow block at the start |
> | randaugment seed 17 | 14.5 s | 0 | completely clean |
> | randaugment seed 23 | 22.7 s | 24 (epochs 27–50: ~47 s) | sustained slowdown from epoch 27 |
> | augmix seed 17 | 93.7 s | 16 (epochs 1–8 and 41–50: ~320 s) | slow blocks at both ends |
> | augmix seed 23 | 86.4 s | 5 (epochs 1–4: ~330 s; epoch 45: 865.9 s) | slow start + one outlier |
>
> Contiguous multi-epoch blocks are the signature of an **external, sustained change in machine
> state** — background load or thermal/power throttling — not of anything about the augmentation. And
> the medians are nearly identical for baseline and RandAugment (13.7 s vs 14.5 s) even though their
> reported totals differ by 23 % and 114 %. **The totals are dominated by interference; the medians are
> not.**

### 75. 🔴 So what IS the real augmentation overhead?

> Using median epoch times instead of totals:
>
> | Condition | Median epoch | Implied 50-epoch time | Overhead vs baseline |
> |---|---:|---:|---:|
> | Crop-and-flip | ~13.7 s | ~11.4 min | 1.00× |
> | RandAugment | ~14.5 s | ~12.1 min | **~1.06×** |
> | AugMix | ~90 s | ~75 min | **~6.6×** |
>
> So the defensible claims are: **RandAugment is essentially free** — about 5–10 % overhead, not 44 % —
> and **AugMix's real overhead is roughly 6–7×**, not the 9.2× the raw means suggest. The reported
> means are inflated by machine interference, not by the methods.
>
> I want to be careful about how I frame this. It is a **re-analysis of data I already had**, not a new
> experiment, and even the medians are contaminated by the same unknown interference. It is a
> hypothesis that a controlled re-run would test. The paper's published wording — that timing is a
> local observation, not a stable estimate — remains the correct claim, and I would not present these
> ratios as results.

### 76. 🔴 Why is AugMix ~6.6× slower when it only triples the batch? Where does the rest come from?

> Good question, and the honest answer is that **I could not determine this**, because the paper
> explicitly states I did not separate image-transform time from model-compute time. That is a
> stated limitation.
>
> But I can reason about the candidates. Three views alone predicts roughly 3×. The stride-1 stem
> already inflates per-image cost for everyone equally, so it does not explain the *difference*. That
> leaves the **input pipeline** as the leading hypothesis: with only `num_workers=2`, each source image
> triggers 2 AugMix calls, each building 3 chains of 1–3 operations — so 6 to 18 operations plus
> internal PIL-to-tensor-to-PIL round-trips, in Python, on the CPU. AugMix's epoch times are in fact
> **bimodal** — a fast mode near 80–95 s and a slow mode near 310–330 s, in both seeds — which is
> consistent with a CPU-side pipeline that is sometimes the bottleneck and sometimes not.
>
> The test would be simple: replace the model with a no-op and time the DataLoader alone, then
> subtract. I should have done it, and I say so.

### 77. 🔴 What could have caused the interference, and did you measure any of it?

> Hypotheses only — I did not log system state during the runs, which is a real limitation:
> **background load** on a shared laptop, which best explains the contiguous early-run blocks;
> **thermal or power-plan throttling**, which best explains the *late*-run blocks (RandAugment seed 23
> epochs 27–50, AugMix seed 17 epochs 41–50) on a GTX 1650; **Windows scheduler or antivirus scanning**
> the data directory; and **disk I/O**, though I mitigated that by extracting CIFAR-10-C only after
> training finished.
>
> The fix is straightforward and I specify it in the paper: fix the power plan to maximum performance,
> record CPU and GPU utilisation and temperature per epoch automatically, discard the first epoch as
> warm-up, repeat each condition three times, and pre-declare the outlier rule. Without that, any
> wall-clock claim on a shared machine is fragile — which is itself a transferable methodological
> lesson.

### 78. 🔴 Wouldn't more data-loader workers have fixed the AugMix bottleneck?

> Very likely, and it is a concrete recommendation. `num_workers=2` is low for augmentation-heavy
> pipelines — on a typical multicore laptop, 4 to 8 workers with an increased `prefetch_factor` would
> likely help substantially if the CPU pipeline is the constraint. I fixed workers at 2 across all
> conditions for fairness, which is methodologically correct but leaves performance on the table. The
> right study design would be to keep workers fixed **for the comparison** and then report a separate
> "tuned workers" cost table, so you get both a fair comparison and an honest engineering number.

### 79. 💣 Is AugMix's 121 minutes vs the baseline's 13 minutes fair, given the baseline does 3× fewer forward passes?

> It is a fair measurement of the *end-to-end cost of the recipe*, but it is **not** a fair measurement
> of the *cost of the augmentation policy alone*, and I present it as the former. The honest way to
> state it: "as implemented, in this configuration, on this hardware, training with AugMix costs about
> 6–9× what crop-and-flip costs." That is true and useful — it is exactly the number a practitioner
> needs. What it does **not** isolate is how much comes from the augmentation transforms versus from
> the 3× batching versus from the extra loss computation. Separating those needs a profiling
> experiment I did not run.

### 80. 🟡 Why is your source-images-per-second figure lower for AugMix, when the model is doing 3× the work?

> Exactly — and that is why I chose that metric. I report throughput as **source CIFAR images per
> second**, not model-inputs per second. For AugMix, one source image produces three model inputs, so
> the source-image rate of 349.6/s corresponds to roughly **1,049 model inputs per second** — still
> about 3× below the baseline's 3,186/s. Reporting source images keeps the denominator comparable to
> the dataset across all conditions, and I explicitly flag the 1-to-3 expansion in the paper so the
> number cannot be misread as understating AugMix's compute. It is a reporting-choice decision, and
> disclosing it is what matters.

### 81. 🟡 Your throughput numbers — do they match the timing?

> Yes, they are derived from it, and the arithmetic is a good consistency check. Throughput is
> `50,000 images × 50 epochs ÷ training seconds`. For the baseline seed 17:
> `2,500,000 ÷ 710.7 = 3,518 images/s`, which matches the reported 3,517.8. So throughput is not an
> independent measurement — it is a restatement of wall time, and I present it as a convenience, not
> as additional evidence.

### 82. 🔴 You never measured FLOPs, energy, or inference latency. Why not, and does it matter?

> Honest answer: scope and time. FLOPs would have been a 20-line calculation and I should have done it —
> it is the hardware-independent cost measure and it is the first thing I would add. Energy needs
> hardware instrumentation I did not set up. **Inference latency I deliberately did not report as an
> augmentation benefit**, and I want to be clear that this is a conceptual position, not an omission:
> with an identical architecture and identical inference preprocessing, training-time augmentation
> *cannot* change inference latency, so presenting it as an augmentation cost or benefit would be a
> category error. The three models I trained would have essentially identical inference cost.

### 83. 🔴 Total training was 5.12 hours on a laptop GPU. Is that a limitation or the point?

> Both, and I would frame it as the honest constraint that shaped the study. It is a limitation in
> that n = 2 and 50 epochs fell out of a compute budget, and I could not run the compute-matched
> control. But it is also the **point**: the lightweight-classifier audience for whom this research
> matters is precisely the people training on modest hardware, so measuring cost on modest hardware is
> the relevant setting, not a compromise. The trade-off I measured — roughly 6–9× the training time for
> +10.78 points of corruption robustness — is a decision a practitioner on this hardware actually faces.

---

## I. Methodology, validity, and data leakage

### 84. 🔴 What is the biggest threat to the validity of your conclusions?

> **The compute confound in Q63**, followed closely by the timing noise. AugMix received 3× the model
> compute and 3× the loss terms per source image, so I cannot cleanly attribute its 10.78-point gain to
> the augmentation distribution rather than to extra optimisation. The timing measurement is the second
> problem: for RandAugment the variability exceeds the effect, so that comparison carries no
> information. Everything else — one architecture, one dataset, synthetic corruptions, n = 2 — is a
> scope limitation that I can state honestly, but the compute confound is a genuine threat to the
> *causal* interpretation of the central claim.

### 85. 🔴 How do you know there is no data leakage?

> Five specific controls.
> **One:** the test sets were used exactly once, for final evaluation, after all six runs finished. I
> never trained on them, never tuned on them, and never selected a checkpoint on them.
> **Two:** I used the **final-epoch** checkpoint for every run, never a best-epoch checkpoint — this
> removes the most common form of leakage in augmentation studies, where people pick the epoch that
> happens to score best on test data.
> **Three:** all hyperparameters are torchvision defaults or original-paper values, frozen in a dated
> protocol **before** any evaluation. There was no search to leak.
> **Four:** the feasibility pilots that informed the schedule are stored in a separate directory and
> excluded from the accuracy analysis.
> **Five:** the test data is structurally isolated from training — CIFAR-10's train and test splits are
> disjoint, and CIFAR-10-C is derived from the *test* images, which are never trained on.
>
> I also want to note what I *did not* do: I have no empirical proof of non-leakage beyond these
> procedural controls. An audit that searched for near-duplicate images between splits would be
> stronger evidence, and I did not run one.

### 86. 🔴 Your training accuracy on the baseline is 88.8 % and test accuracy is 86.4 %. Is the model overfitting?

> There is a 2.4-point train-test gap, which is small for CIFAR-10 from scratch. With heavy augmentation
> the picture is very different: RandAugment's training accuracy is only 82.6–83.8 %, far below its
> test accuracy, because its "training accuracy" is measured on RandAugment-*damaged* images, which are
> genuinely harder. So the gap is not a clean overfitting signal — the three training-accuracy numbers
> are measured on three different input distributions (see Q69).
>
> And I genuinely cannot diagnose overfitting, because I used no validation split and evaluated only
> the final epoch. I never saw a validation curve. I can say the cosine schedule completed and the runs
> behaved normally, but I cannot make a strong claim about convergence — and 50 epochs is probably not
> converged anyway, which is a stated limitation.

### 87. 🔴 What is your internal validity — how do you know the three conditions differed only in augmentation?

> **Structurally**, which is the strongest form. I implemented all three conditions in a **single
> parameterised script**, so there is one code path for the architecture, one for the optimiser, one
> for the scheduler, one for the DataLoader, one for the timing, and one for checkpointing. Only two
> places branch on the method: the dataset's `__getitem__`, and the loss/view handling in the training
> loop. It is therefore **structurally impossible** for one condition to have received a different
> learning rate, batch size, epoch count, or checkpoint rule — there is only one code path to get wrong.
>
> On top of that: identical seeds across conditions (paired comparison), and a written frozen protocol
> with a deviation log so any change would have to be documented. The external-validity side is much
> weaker — one architecture, one dataset, synthetic corruptions.

### 88. 🟡 How would a reader reproduce your work?

> Three commands. Install the environment (Python 3.14.5, torch 2.14.0+cu130, torchvision 0.29.0+cu130,
> NumPy 2.5.2); run `src/train.py` six times with `--method {baseline,randaugment,augmix}` and
> `--seed {17,23}`; then run `src/evaluate.py` against the checkpoint root, then `src/summarize.py` to
> regenerate the analysis CSVs. The exact commands are in the README.
>
> The reproduction package contains the scripts, the frozen protocol, all configs, all per-epoch logs,
> all six checkpoints, all evaluation outputs, and the aggregate tables. The two things a user must
> supply themselves are the **datasets** — because CIFAR-10 and CIFAR-10-C terms do not permit
> redistribution — with the download URL and the **MD5 checksum**
> `56bf5dcef84df0e2308c6dcbcbbd8499` so they can verify they got the right files. One caveat: the
> *timings* will not reproduce, which is my main finding.

### 89. 🟡 You mention "pre-registered" — were you actually pre-registered?

> No, and I want to be careful with the word. I was not pre-registered with any external registry. What I
> did was **freeze a written protocol locally before running the final experiments**: a versioned,
> dated document (`pilot_protocol.md`, v0.1, 2026-09-25) stating the research question, every fixed
> hyperparameter, the data-separation rules, the checkpoint rule, the exclusion rules, and an initially
> empty deviation log. That is "pre-registration in spirit" and it gives the same internal benefit —
> the settings were committed to before the results existed, so I could not tune to the outcome. But
> it is not externally verifiable the way a registered report is, and I would not claim it is.

### 90. 🟡 Why did you extract the archive's MD5 and record the download date?

> Because a result is only reproducible if the data is identifiable. If someone re-runs this in a year
> and CIFAR-10-C has been updated, or they download a different mirror, the numbers will not match and
> there will be no way to tell why. Recording the source, the date, and the checksum pins the exact
> artifact. It is a small amount of work that makes the difference between "we ran this" and "anyone
> can verify we ran this." I also recorded the download URL in the paper.

### 91. 🔴 Your paper says it does not compare its numbers to Fadhilah et al. (2026). Why not? Wouldn't that be the obvious thing to do?

> Because their setup is different in ways that make numeric comparison invalid. From my reading, they
> **fine-tune pretrained weights** (I train from scratch), they use **DistortionMix** — a noise and
> contrast augmentation, not RandAugment or AugMix — and they describe **19 corruption types** where
> the canonical benchmark has 15. Fine-tuning versus from-scratch alone can shift CIFAR-10 accuracy by
> 5–10 points, so a side-by-side number would be misleading. I would rather report no comparison than
> a comparison that invites the wrong conclusion. What I **do** cite them for is the novelty boundary:
> they establish that this combination of ideas has already been studied, which is why I make no
> novelty claim. Verifying these details against the full paper is on my to-do list.

---

## J. Statistics and seeds

### 92. 🔴 Two seeds. How can you say anything at all?

> I do not say much, and that is the correct posture. With n = 2 the sample SD is barely meaningful —
> one degree of freedom — and no hypothesis test is defensible. So I present **descriptive statistics
> only**, and the paper says explicitly: "These descriptive statistics do not support inferential
> claims."
>
> What I *can* legitimately do is separate claims by their effect size relative to observed noise.
> **Large and stable:** AugMix's +10.78 points, which is 6.6× the baseline's SD and 35× AugMix's own
> SD, with a maximum single-seed spread of 0.31 points. That is a real effect. **Small and unstable:**
> the clean-accuracy differences, 0.37 points against an SD of up to 0.82 — no effect. **Uninterpretable:**
> the RandAugment timing, where the SD exceeds the mean difference. Grading my own claims by effect
> size is the honest way to use a small n.

### 93. 🔴 With n = 2, isn't the standard deviation almost meaningless?

> Yes, and I would not defend the SDs as precise estimates. With two observations, the sample SD
> (n − 1 denominator) is computed from a single squared deviation, so it is extremely sensitive to a
> single value. My SDs are best read as a **rough indication of run-to-run spread**, not as confidence
> intervals. If I were advising a reader: use the per-seed values, which I report in full, rather than
> leaning on the SD.

### 94. 🟡 If you had more time, what statistical test would you run?

> A **paired test on per-seed differences**, because the seeds are shared across conditions. For each
> seed I compute the difference in CIFAR-10-C accuracy between two conditions, then test whether the
> mean difference across seeds is distinguishable from zero. This is far more powerful than an unpaired
> test, because it removes between-seed variance — with n = 5 seeds and a paired design, I could detect
> differences that an unpaired design at the same n would miss.
>
> I would use a paired t-test for simplicity and report means and bootstrap confidence intervals, since
> accuracy proportions have a binomial structure that a t-test only approximates. I would also note a
> subtlety: the same 10,000 test images are evaluated by all six models, so the measurements are
> **correlated**; a proper analysis should acknowledge that rather than treating each accuracy as
> independent.

### 95. 🔴 Isn't 750,000 images per model a huge sample? Why aren't your results more certain?

> This is a really important distinction and a common confusion. The **evaluation** sample is indeed
> huge — 750,000 images gives a standard error on a single accuracy estimate of well under 0.2 points,
> so the *measurement* of a given model's accuracy is precise.
>
> But that is not the source of uncertainty in my study. My uncertainty is about **training
> variability** — how much the model would change if I trained it with a different random seed. That
> uncertainty is governed by the number of **training runs**, which is 2. A large test set makes each
> number precise; it does not make the *conclusion* well-supported, because the conclusion's variability
> comes from training stochasticity. This is why "evaluate on more data" and "train more seeds" are
> not interchangeable, and why more seeds is the fix I need.

### 96. 🟡 Did you report any confidence intervals?

> No, and with n = 2 they would be misleading. A 95 % interval from two observations spans an enormous
> range and would convey a false impression of rigour. I preferred to show the **per-seed values
> directly** and let readers see the spread. If I had 5+ seeds, I would report bootstrap intervals.

### 97. 🔴 Your RandAugment clean-accuracy SD is 0.82, larger than AugMix's 0.01. Why would one method be 80× more variable than another?

> That is a striking difference and I would not over-interpret it. With n = 2, an SD is computed from a
> single squared deviation, so a single unusual run dominates it — RandAugment's seed 23 happened to
> land at 85.60 % while seed 17 landed at 86.76 %.
>
> There is a plausible mechanism worth mentioning: AugMix's consistency objective explicitly penalises
> sensitivity to input perturbation and averages over three views, which could plausibly produce a
> flatter, more stable solution, and the corruption-accuracy SDs show the same pattern (AugMix 0.31,
> RandAugment 0.06, baseline 1.63). But with n = 2 I cannot distinguish a real effect from chance, and
> I did not test the mechanism. It is a hypothesis worth investigating with more seeds.

---

## K. Limitations and self-criticism

### 98. 🔴 What is the weakest part of this work?

> The **compute confound**, and I would rather defend that than have a professor find it. AugMix got 3×
> the model compute and 3× the loss terms per source image, and I have no compute-matched control to
> separate "better augmentation" from "more gradient signal." The entire central claim rests on an
> attribution I cannot verify. A close second is the timing measurement, which is unreliable to the
> point of being uninformative for RandAugment.

### 99. 🔴 If you could fix one thing, what would it be?

> **Add a compute-matched control.** Train the baseline (and RandAugment) for three times as long, or
> at three times the throughput, at the same wall-clock budget. That single experiment separates
> "AugMix's augmentation and consistency objective are better" from "AugMix simply did more training."
> Without it, my headline result has an alternative explanation I cannot rule out. Everything else —
> more seeds, a second architecture, CIFAR-10.1, better timing — strengthens the study, but only this
> one fixes a genuine threat to the causal claim.

### 100. 🔴 Your paper has no novelty. Why does it exist, and should I be citing it on my CV?

> On the CV: yes, with honest framing — "experimental preprint" or "technical report," not
> "published research paper," and never implying peer review. A Zenodo DOI identifies an **archived
> report**, not a quality signal, and I would say so if asked.
>
> On whether it should exist: its value is not a new finding, it is a **reproducible, fully audited,
> cost-reported measurement** of a decision practitioners actually face, with every artifact released —
> code, frozen protocol, per-epoch logs, all six checkpoints, per-run evaluations, aggregate tables,
> dataset checksums, and a deviation log. Plus one small genuine contribution: the re-analysis of my own
> timing logs showing the noise is episodic in contiguous blocks, which yields a materially better
> estimate of augmentation overhead than the headline table — and which contradicts my own paper's
> timing numbers in a useful way. Reporting a result that undercuts your own headline table is, I think,
> a legitimate reason to trust the rest of the work.

### 101. 🟡 What would a reviewer reject this paper for?

> Being blunt: **n = 2 with no significance testing**, and **one architecture on one dataset at 50
> epochs**. A second rejection reason would be the compute confound. A fourth would be the timing
> measurement's reliability. A fifth, for a real venue, would be insufficient novelty — which is why
> I targeted a preprint archive rather than a journal, and why the paper is framed as exploratory
> throughout. I think the honest response is that this study is correctly sized for its stated purpose
> and incorrectly sized for a publication, and I said so in the document rather than hoping nobody
> noticed.

### 102. 🟡 What would you tell a student who wanted to replicate this?

> Read `pilot_protocol.md` first — it is the design document, and it will tell you the intent faster
> than the code. Then read `src/train.py`, because all three conditions live in one file and the two
> `if args.method` branches show you exactly what differs. Run the one-epoch seed-99 pilots before
> anything else, so you can see the timing behaviour and the warm-up artefact. Most importantly:
> **log your system state during training.** I did not, and it cost me the ability to make a clean
> timing claim. That is the one piece of advice I would give most forcefully.

### 103. 🔴 Is there anything wrong with your own paper that you have not fixed?

> Yes, one imprecision I found while preparing these notes. The Discussion says: *"A single epoch in
> AugMix seed 23 took 865.9 seconds while most late epochs took 80–110 seconds."* That is accurate **for
> seed 23 only**. In **seed 17**, the late epochs — 41 through 50 — each took about **310–325 seconds**,
> not 80–110. So "most late epochs" is true of one seed and false of the other.
>
> The paper's conclusion is unaffected, because the point of that sentence is that timings are
> unstable, and 310 s versus 80 s makes that point even more strongly. But the sentence as written
> over-generalises from one seed. I would correct it to state the pattern per seed. I would rather
> raise this myself than have a professor find it in `epochs.csv`.

### 104. 🟡 What is the most likely way your results would not generalise?

> Three ways, in order of likelihood. **First**, to other architectures — augmentation interacts with
> model capacity and normalisation, and I have one model. **Second**, to real-world degradation —
> CIFAR-10-C corruptions are synthetic code-generated artefacts, and Mintun et al. (2021) showed that
> augmentation gains on synthetic corruptions often do not transfer to naturally collected data. **Third**,
> to longer schedules — at 200 epochs the clean accuracies might converge and compress the differences,
> and the ranking of the methods could change.

---

## L. Literature, related work, and novelty

### 105. 🔴 What is your novelty claim?

> **I make no novelty claim.** The broad question — does augmentation improve corruption robustness, and
> what does it cost — is well established. I found a 2026 paper, Fadhilah, Warsito and Faridah (JIKI,
> DOI 10.21609/jiki.v19i1.1558), that evaluates **MobileNetV3-Small on CIFAR-10 and CIFAR-10-C**,
> compares **a baseline recipe against a distortion-based augmentation**, and reports **50-epoch
> training times**. That is nearly my exact combination of ingredients.
>
> So I repositioned the work as what it honestly is: a small, controlled, matched-setting,
> replication-style pilot with explicit cost reporting and full artifact release. I state the overlap in
> the abstract-adjacent framing, in Related Work, in Limitations, and in my protocol. Overclaiming here
> would have been easy to do and would have destroyed the work's credibility.

### 106. 🟡 Summarise the key prior work.

> - **Hendrycks & Dietterich (ICLR 2019)** — created CIFAR-10-C. Motivates reporting results **per
>   corruption and per severity**, because a single average hides differences between corruption types.
>   This directly shaped my analysis: I report by-type and by-severity tables, not just an aggregate.
> - **Cubuk et al. (CVPR Workshops 2020), RandAugment** — reduces the search space for automated
>   augmentation by sampling a fixed number of ops at sampled strengths, removing the policy-search
>   phase. Relevant to my cost question because it separates *search* cost from *online* cost.
> - **Hendrycks et al. (ICLR 2020), AugMix** — combines augmentation chains with a JSD consistency
>   loss; evaluated on four CIFAR model families including CIFAR-10-C results. Established baseline,
>   and the source of the JSD weight 12 and the general recipe.
> - **Yin et al. (NeurIPS 2019)** — Fourier-domain analysis showing corruption-robustness gains vary by
>   corruption type and one augmentation can help some families while hurting others. This is the
>   closest explanation for my RandAugment-loses-on-pixelate result, and it is why I report per-corruption
>   tables rather than only an average.
> - **Mintun et al. (NeurIPS 2021)** — studies the interaction between training augmentations and test
>   corruptions, and finds robustness may degrade on corruptions dissimilar to the benchmark's. The
>   reason CIFAR-10.1 matters and the reason I limit my generalisation claims.
> - **Modas et al. (ECCV 2022), PRIME** — compares corruption-robust augmentation methods with CIFAR-10-C
>   results and reports some relative augmentation runtime. Confirms both robustness **and** cost are
>   already studied, which is why the gap had to be narrowed.
> - **Kireev et al. (UAI 2022)** — PreAct ResNet-18 on CIFAR-10/CIFAR-10-C with a wall-clock table for
>   robust-training recipes on a V100. Confirms controlled corruption evaluation with wall-clock
>   reporting already exists.
> - **Trinh et al. (NeurIPS 2024)** — a different route to corruption robustness via multiplicative
>   weight perturbations. A recent lead, not a direct comparison.
> - **Fadhilah et al. (2026)** — the close overlap described above.

### 107. 🟡 What did you contribute that is genuinely new, if not novelty?

> Three things, in descending order of substance.
> **One:** a re-analysis of my own timing logs showing the variability is **episodic in contiguous
> multi-epoch blocks** rather than random jitter, which yields a materially different and more
> defensible estimate of augmentation overhead — about 1.06× for RandAugment and 6.6× for AugMix,
> rather than 1.44× and 9.2×. That analysis is not in the paper, and it contradicts the paper's own
> headline table in a way I think is useful.
> **Two:** a concrete, per-corruption demonstration that augmentation gains are strongly
> family-dependent — AugMix gains 18–23 points on noise but only 0.06 on brightness — with a
> documented case of a stronger method *losing* to the baseline (RandAugment on pixelate).
> **Three:** the fully released, auditable artifact set: frozen protocol, per-epoch logs, all six
> checkpoints, per-run evaluations, aggregate tables, dataset checksums, and a dated deviation log.

### 108. 🟡 Why did you use a 2-day deadline as a scope constraint? Was that wise?

> It was a real constraint, and I made it explicit rather than letting it silently shape the study. I
> wrote in the protocol, before running anything, that the deliverable was "a pilot/replication-style
> comparison, not a claim of publication-level novelty" — so the deadline shaped the **framing**
> honestly rather than producing a result I then had to oversell. Given the constraint, I think
> guaranteeing a complete, reproducible, honestly-scoped study was the right call over starting a
> larger study that would not have finished. The cost is real and I list it: n = 2, one model, 50
> epochs, no compute-matched control. The full 24-week plan with more seeds, a second architecture,
> CIFAR-10.1, and proper timing infrastructure remains as the extension.

### 109. 🔴 Where would you publish this?

> Honestly, a two-seed pilot is not competitive at a peer-reviewed ML venue, and I would not try. It
> is on **Zenodo** as a versioned preprint with a reserved DOI and a CC BY 4.0 reproducibility archive —
> which gives it a citable, timestamped, permanent identifier. A student journal or an
> undergraduate/early-stage venue is a plausible next step, chosen **with my supervisor**, and the
> first thing to check would be whether the venue's reviewers would accept a study this size. What I
> would want before submitting anywhere is a defensible gap statement, which means completing the
> literature review I only partially did.

---

## M. Reproducibility, code, and tooling

### 110. 🟡 Why one training script for all three conditions?

> Because it makes the comparison **structurally** fair rather than merely intended to be fair. There is
> one code path for the architecture, the optimiser, the scheduler, the DataLoader, the timing, the
> logging, and the checkpointing. Only two places branch on the method. So it is impossible for one
> condition to have received, say, a different learning rate or a different epoch count — there is only
> one place that code lives, and all three conditions execute it.
>
> The alternative — three separate scripts — would create three chances to introduce an accidental
> difference, and I would have to argue they were equivalent. I prefer to make the mistake impossible.

### 111. 🔴 How do you know your evaluation used the same preprocessing as training?

> By construction: `src/evaluate.py` does `from train import MEAN, STD, make_model`. The evaluator
> **imports the constants and the model-construction function directly from the training script**, so
> it cannot drift. This eliminates a whole class of silent bug where an evaluation normalisation
> constant differs by a small amount from training, producing mysteriously depressed accuracy that is
> very hard to diagnose. For the corruption path I wrote the normalisation explicitly —
> `uint8 → ÷255.0 → (x − mean)/std` — which is numerically equivalent to `ToTensor()` + `Normalize()`
> but is visible in the code without needing to trust torchvision.

### 112. 🔴 How do you know evaluating a checkpoint twice gives the same answer?

> Logically I can argue it: evaluation uses `model.eval()` and `torch.inference_mode()`, the transforms
> are deterministic (`ToTensor` + `Normalize`, no random ops), the DataLoader has `shuffle=False` and
> `num_workers=0`, and there is no augmentation at test time. So there is no source of nondeterminism in
> the evaluation path.
>
> **But I did not formally verify it by running the same checkpoint twice and diffing the outputs.** My
> original plan listed that as a check and I did not execute it. That is an honest gap: I have a strong
> argument, not an empirical record. It is a five-minute check and I should have run it.

### 113. 🟡 Why did you use memory-mapping for the CIFAR-10-C arrays?

> Each corruption file is 50,000 × 32 × 32 × 3 uint8 ≈ 150 MB, and the aggregate evaluation is 750,000
> images. Loading everything into RAM at once would be wasteful and, with 19 files present, could
> approach 3 GB. `np.load(..., mmap_mode="r")` memory-maps the file so the OS pages it in on demand, and
> I slice one severity block of 10,000 images at a time. Peak memory stays low and the code stays
> simple. It also means the 150 MB file is read sequentially, which is cache-friendlier.

### 114. 🟡 Why did you log software versions automatically?

> Because manual provenance recording always fails eventually. My training script builds a
> `system_manifest()` that captures the Python version, platform string, Torch version, torchvision
> version, NumPy version, whether CUDA is available, the GPU name, and total GPU memory — and writes it
> into **every** `config.json` automatically. So I cannot forget it, and every run carries its own
> environment record. If two runs had different library versions I would see it immediately, which
> matters because a torchvision version bump could change `AugMix`'s internals and therefore my
> augmentation.

### 115. 🔴 You have no experiment tracking. Was that a mistake?

> For a study this size, no — but it is a real limitation at scale. I did not use TensorBoard, Weights &
> Biases, or MLflow. I mitigated it with per-epoch CSV logs that are **flushed after every epoch**, so
> a crash loses nothing, plus a `config.json` per run, plus a printed line per epoch to the console.
>
> The thing I would want most from a tracker is **system-state logging alongside the metrics**, which
> is exactly what I did not have and exactly what would have explained the timing anomalies. That is
> the gap. Adding per-epoch CPU/GPU utilisation and temperature to the CSV would have converted §H from
> speculation into evidence.

### 116. 🟡 Your code depends on torchvision's AugMix internals. Is that a risk?

> It is a coupling risk, and I would flag it to any user. I depend on torchvision's implementation
> matching the published AugMix algorithm. If a future torchvision version changed `AugMix`'s internals
> — the operator set, the chain construction, the mixing weights — my augmentation would silently
> change, and I would not necessarily notice. Mitigations: I pin the exact versions in the README and in
> every `config.json`; and I verified the actual defaults in the installed version rather than assuming
> them, which is how I can state with confidence that `severity=3`, `mixture_width=3`, `chain_depth=-1`,
> `alpha=1.0` are the library defaults. But yes, the results are only valid for that library version.

### 117. 🟡 Why publish a 34.5 MB archive? What is in it?

> Scripts, the frozen protocol, all six `config.json` files, all six per-epoch CSVs, all six
> checkpoints, all per-run evaluation CSVs and JSONs, the four aggregate analysis CSVs, and a README.
> The 34.5 MB is almost entirely the six model checkpoints. The **datasets are deliberately excluded**,
> because CIFAR-10 and CIFAR-10-C terms do not permit redistribution — so instead I published the
> download URL and the MD5 checksum so a user can obtain and verify the exact files themselves. That is
> the correct way to handle dataset licensing, and it is why the archive is small enough to host
> anywhere.

---

## N. Ethics, licensing, and AI use

### 118. 🟡 Ethical considerations?

> Minimal, and I would say so plainly rather than manufacture concerns. **No human participants** — the
> datasets are public image collections with no personal data being collected. **No animal or medical
> imaging.** The main ethical obligation is **data licensing and attribution**, which I handled
  properly: I did not redistribute the datasets, I cited the original CIFAR-10-C paper and the Zenodo
> record, and I provided the checksum. **Research integrity** is the substantive issue here, and the
  requirements were: do not tune on test data, do not discard inconvenient results, do not silently
  replace failed runs, and do not claim novelty you have not verified. My frozen protocol, deviation log,
  and the fact that I report a method *losing* to a baseline and a timing measurement I do not trust
  are all attempts to meet those. I would also flag that CIFAR-10-C is **synthetic** corruption data —
  describing models as "robust" based on it risks overstating real-world capability, which is a
  communication-ethics issue I try to avoid by stating the limitation prominently.

### 119. 🟡 Did you use AI tools? Should you have declared it?

> Yes — I used AI assistance to help draft and organise documents, and I would declare that in line with
  my university's and any target venue's policy. My position is that AI assistance is a **tool**, like an
  IDE or a search engine, and what matters is that the **claims are verified and the author owns the
  work**. So the discipline I applied was: every number in the paper must trace to a saved result file
  (and I generated the tables with a script rather than typing them); every citation must be checked
  against the primary source; and every generated statement about the code must be true of the code —
  which is why I re-read my own training script closely enough to notice the JSD-in-the-logged-loss
  issue and the warm-up artefact in the pilots. I would not put my name on anything I had not verified
  myself, and I would rather be asked about this than have it surface later.

### 120. 🟡 What is the compute cost / carbon footprint of this project?

> I did not measure energy, so I cannot give a number, and I would not guess one. What I can bound
  roughly: about 5.12 hours of timed training on a single GTX 1650 — a 75 W mobile GPU, so on the order
  of a few hundred watt-hours for the training itself, plus system overhead, plus evaluation. The
  dominant term is AugMix at 4.05 of those 5.12 hours. The broader point is that reporting energy and
  FLOPs alongside wall time is a genuine gap in the augmentation literature, and it is on my
  follow-up list precisely because wall-clock time on one machine is such a poor proxy.

---

## O. Future work and research vision

### 121. 🔴 What would you do next, in priority order?

> **One — compute-matched control.** Train the baseline and RandAugment at 3× the compute budget. The
> only experiment that separates "AugMix's augmentation is better" from "AugMix did more training."
> Without it my central claim has an unfalsified alternative explanation.
> **Two — fix the timing methodology.** Report median epoch time, discard the first epoch as warm-up,
> run each condition 3×, fix the power plan, log per-epoch CPU/GPU utilisation and temperature, and
> pre-declare the outlier rule.
> **Three — profile the input pipeline separately.** Time the DataLoader with a no-op model to get pure
> augmentation cost. This directly tests whether the CPU pipeline or the GPU compute dominates AugMix's
> cost, which my paper explicitly says it cannot answer.
> **Four — more seeds (5+) and a paired statistical test**, which is properly powered.
> **Five — tune each method on a validation split** and report both default and tuned comparisons.
> **Six — a second architecture** (a stride-2 MobileNetV3-Small and a small ResNet) to test whether the
> ranking is architecture-independent.
> **Seven — CIFAR-10.1 v6** to test transfer to naturally collected data.
> **Eight — report mCE, FLOPs, reserved memory, and energy**, and add calibration metrics (ECE, NLL) to
> test AugMix's uncertainty claim, which is a stated motivation I never tested.

### 122. 🟡 If you had an unlimited compute budget for one month, what study would you run?

> A clean, well-powered version of exactly this comparison, designed as a factorial rather than a
> three-point comparison. **Axes:** three augmentation policies × at least five seeds × two
> architectures (a stride-2-stem MobileNetV3-Small and a small ResNet) × three evaluation sets
> (CIFAR-10, CIFAR-10-C, CIFAR-10.1) × 200 epochs, with a **compute-matched arm** at every policy.
> Plus one axis I find genuinely interesting and did not do: **isolate the two components of AugMix** —
> multi-view averaging alone, JSD consistency alone, and both together. That would answer whether
> AugMix's advantage comes from seeing more augmented images or specifically from the consistency
> objective, which is a question my design cannot address and which I think is the most scientifically
> interesting thing about this whole topic.

### 123. 🟡 What is the most interesting open question in this area?

> Whether augmentation-induced robustness generalises **beyond the corruption distribution it was
> trained for**. Mintun et al. (2021) showed gains on synthetic benchmark corruptions often do not
> transfer to real degradation. That means the headline metric the whole field optimises — CIFAR-10-C
> accuracy — may be a partially misleading target. If a method can score 78 % on CIFAR-10-C and still
> fail on real camera degradation, then the field is optimising the wrong objective, and the right
> response is a benchmark of *real* degradations with held-out corruption families.
>
> A second question I find interesting: **the compute dimension is under-measured across the field.**
> Most augmentation papers report accuracy only. If augmentation overhead varies by an order of
> magnitude — as it does in my measurements — then reporting it should be standard practice, and
> compute-matched comparisons should be the default.

### 124. 🟡 What would make augmentation worth its cost, in your view?

> It depends on the deployment, and that is the practical conclusion of my work. If the input is
> reliably clean — a controlled imaging setup, a curated dataset — augmentation beyond crop-and-flip
> buys nothing, since all three conditions tied on clean accuracy. If the input is **unreliable**, and
> you cannot control the capture conditions, then augmentation is not optional: the baseline lost
> 27.6 points from severity 1 to severity 5, while AugMix lost 13.6. That gap is the difference between
> a model that is usable outdoors in bad weather and one that is not.
>
> The decision then becomes an economic one: is 6–9× the training time worth 10.78 points of
> robustness? For a single training run, almost certainly yes — 6 extra hours is cheap against a
> deployed model. For a large hyperparameter search over many configurations, it may not be, and
> RandAugment at roughly 1.06× overhead is then the obvious choice.

---

## P. Questions about you

### 125. 🟢 What was your specific role in this project?

> Everything, as the sole author. I did the literature screening and the source matrix; designed and
> froze the protocol; wrote all three Python scripts; set up the environment; ran the three feasibility
> pilots and all six final training runs; wrote the evaluator and evaluated all six checkpoints on
> 4.56 million images; wrote the aggregation script and generated every table; wrote the manuscript;
> and prepared and published the Zenodo release with its reproducibility archive. There was no
> co-author and no supervisor contribution to the technical work.

### 126. 🟡 What did you personally find difficult?

> **Not writing the code** — that was the easy part. The hard part was the timing measurement. I spent
> real effort trying to understand why identical configurations produced 2× different runtimes, and I
> eventually realised I could not resolve it, because I had not instrumented the system. The lesson was
> expensive and simple: **measure your environment, or your measurements will not be interpretable.**
>
> Second-hardest was resisting the temptation to overclaim. I found a paper doing almost exactly my
> study, and the honest move was to narrow my contribution and say so plainly, rather than find a
> flattering angle. That is a discipline, not a technical skill, and it was the hardest part of writing.

### 127. 🟡 What did you learn that was not in a course?

> Mostly things about **measurement honesty**. Concretely: that `torch.cuda.synchronize()` before
> stopping a timer is not optional; that cuDNN's `benchmark=True` silently destroys run-to-run
> reproducibility; that DataLoader workers need explicit per-worker seeding or their augmentations
> become correlated; that `max_memory_allocated` is not "GPU usage"; that reporting throughput per
> source image is misleading when one image becomes three model inputs; and that a two-seed study with
> no test can still yield a defensible finding if you grade your claims by effect size and say plainly
> what you cannot support.

### 128. 🟡 How would you extend this to a master's thesis?

> In four stages. **Stage 1 — make the timing trustworthy**, because without it there is no cost result
> to build on: controlled power plan, per-epoch system-state logging, 3 repetitions, median statistics,
> and a pre-declared outlier rule. **Stage 2 — add the compute-matched control** so the central claim
> is causally interpretable, and decompose AugMix into multi-view-averaging-only, JSD-only, and both,
> which answers the most interesting question in the project. **Stage 3 — generalise**: two
> architectures, five or more seeds, CIFAR-10.1 for real-data transfer, 200 epochs, and mCE alongside
> raw accuracy. **Stage 4 — broaden the cost story**: FLOPs, energy, and a sensitivity analysis over
> worker count and batch size, so the cost model is hardware-independent rather than tied to one GTX
> 1650.
>
> I would want a defensible gap statement before submitting anywhere, which means completing the
> literature review I only partially did.

### 129. 🟡 What would you do differently if you started again?

> Four things, in order. **One — log system state from the first run.** Per-epoch CPU/GPU utilisation,
> temperature, and power, written next to the loss. That single change would have converted my weakest
> section from speculation into evidence. **Two — run a 3× compute control early**, as a pilot, because
> it would have changed how I framed the whole result. **Three — restore the train/validation split**
> from the start, so hyperparameters could be tuned legitimately and overfitting could be diagnosed.
> **Four — decide the compute-matching question before, not after.** My biggest mistake was treating
> "one script, three methods" as sufficient experimental control. It controls the *code*, but not the
> *compute*, and for AugMix those are different things.

### 130. 🟡 What is the one thing you would want a professor to remember about this work?

> That I am precise about what the evidence does and does not support. The finding I would stand behind
> is narrow and solid: **under a fixed 50-epoch schedule on CIFAR-10 with a CIFAR-adapted
> MobileNetV3-Small, AugMix substantially improves CIFAR-10-C accuracy while leaving clean accuracy
> essentially unchanged, at several times the training cost** — and the per-corruption breakdown shows
> that gain is strongly family-dependent, with one case of a stronger method doing worse. The finding I
> explicitly do **not** stand behind is any precise cost multiplier, because my timing measurement is
> not reliable enough to support one. I would rather a professor think I know the difference between
> those two kinds of claim than think I oversold my results.

---

## Q. Rapid-fire one-liners

Memorise these. If you can answer all of them without hesitating, you are ready.

| # | Question | Answer |
|---:|---|---|
| 1 | How many training runs? | **6** (3 methods × 2 seeds), 50 epochs each, 0 failures, 300 epochs total |
| 2 | Total training time? | **18,438.9 s ≈ 5.12 h**; AugMix alone 14,565 s ≈ 4.05 h |
| 3 | Architecture? | torchvision **MobileNetV3-Small**, stride 2→1, 10-class head, from scratch |
| 4 | Parameter count? | **1,528,106** (verified in PyTorch) |
| 5 | Training data? | All **50,000** CIFAR-10 train images, native 32×32 |
| 6 | Test data? | 10,000 official test + **750,000** corrupted = 760,000 per checkpoint |
| 7 | Total eval forwards? | **4,560,000** |
| 8 | Corruption types? | **15** canonical (archive had 19; 4 excluded for comparability) |
| 9 | Severities? | **5** (1–5) → 75 corruption-severity cells |
| 10 | Best clean accuracy? | Crop-and-flip, **86.39 ± 0.25 %** |
| 11 | Best CIFAR-10-C accuracy? | AugMix, **77.82 ± 0.31 %** |
| 12 | AugMix's corruption gain? | **+10.78 pp** over baseline |
| 13 | RandAugment's corruption gain? | **+3.96 pp** |
| 14 | Clean-accuracy cost? | **−0.37 pp** (AugMix), **−0.21 pp** (RandAugment) — within noise |
| 15 | Clean-accuracy spread? | **0.37 pp**, smaller than the within-condition SD (up to 0.82) |
| 16 | Cheapest? | Crop-and-flip, **13.2 min** mean |
| 17 | Most expensive? | AugMix, **121.4 min** mean |
| 18 | Reported time ratio? | 9.2× (AugMix), 1.44× (RandAugment) — **not reliable** |
| 19 | Median-epoch time ratio? | **~6.6×** (AugMix), **~1.06×** (RandAugment) — better estimate |
| 20 | Peak GPU memory? | 205,514,240 B baseline/RandAugment; **557,116,928 B** AugMix (**2.71×**) |
| 21 | Memory metric? | `torch.cuda.max_memory_allocated` — **not** total GPU usage |
| 22 | Hardest corruption? | **Gaussian noise** (baseline 47.75 %) |
| 23 | Easiest corruption? | **Brightness** (baseline 84.50 %) |
| 24 | Where did AugMix gain most? | **Noise: +18 to +23 pp** |
| 25 | Where did it gain least? | **Brightness: +0.06 pp** (a tie) |
| 26 | Any method that LOST? | **Yes** — RandAugment on pixelate, 70.12 vs 71.74 (−1.62 pp) |
| 27 | Severity 5 accuracy? | Baseline 52.17 / RandAugment 57.12 / **AugMix 69.67 %** |
| 28 | Severity 1→5 degradation? | Baseline **−27.60**, RandAugment −24.42, **AugMix −13.62** |
| 29 | Seeds? | **17 and 23** (99 = 1-epoch feasibility pilot, excluded) |
| 30 | Batch / workers / epochs? | **128 / 2 / 50** |
| 31 | Optimiser? | **SGD**, lr 0.05, momentum 0.9, weight decay 5e-4, **cosine annealing** |
| 32 | RandAugment params? | `num_ops=2, magnitude=9` — **torchvision defaults, untuned** |
| 33 | AugMix params? | `severity=3, mixture_width=3, chain_depth=−1, alpha=1.0` — **defaults** |
| 34 | AugMix JSD weight? | **12** (from the original AugMix paper) |
| 35 | AugMix views? | **3** — one crop-flip + two independent AugMix views |
| 36 | AugMix loss? | mean CE over 3 views **+ 12 × JSD** |
| 37 | JSD formula? | `[KL(p₁‖M) + KL(p₂‖M) + KL(p₃‖M)] / 3`, `M = (p₁+p₂+p₃)/3` |
| 38 | Why JSD not KL? | **Symmetric and bounded in [0, ln 2]** — a safe regulariser |
| 39 | Checkpoint rule? | **Final epoch only** — no best-test selection |
| 40 | Test used for tuning? | **No** — frozen protocol, no validation split, no early stopping |
| 41 | GPU? | **NVIDIA GeForce GTX 1650**, 4 GB (4,294,639,616 bytes) |
| 42 | Software? | Windows 11 · Python 3.14.5 · torch 2.14.0+cu130 · torchvision 0.29.0+cu130 · NumPy 2.5.2 |
| 43 | Pretrained weights? | **No** — `weights=None`, from random init |
| 44 | CIFAR-10-C checksum? | MD5 **`56bf5dcef84df0e2308c6dcbcbbd8499`**, Zenodo record 2535967 |
| 45 | Datasets redistributed? | **No** — download URL + checksum provided instead |
| 46 | Published? | Zenodo **preprint** v1.0, 2026-09-25, DOI `10.5281/zenodo.22961521`, CC BY 4.0 |
| 47 | Peer-reviewed? | **No.** A Zenodo DOI is an archived report, not a quality signal |
| 48 | Novelty claim? | **None.** Fadhilah et al. (2026) already covers this combination |
| 49 | Closest prior work? | Fadhilah, Warsito & Faridah, JIKI 2026, DOI `10.21609/jiki.v19i1.1558` |
| 50 | Biggest limitation? | **No compute-matched control** — AugMix got 3× the compute |
| 51 | Second limitation? | **n = 2** — no significance test possible |
| 52 | Third limitation? | **Timing unreliable** — SD exceeds the effect for RandAugment |
| 53 | Not measured? | mCE, FLOPs, energy, inference latency, calibration, CIFAR-10.1, 2nd architecture |
| 54 | AugMix's extra compute? | **3× model inputs** per source image (3 × 128 = 384) |
| 55 | AugMix throughput? | 349.6 **source** img/s ≈ **1,049 model inputs/s** |
| 56 | Scripts? | `train.py` (235 lines), `evaluate.py` (145), `summarize.py` (109) |
| 57 | Key design choice? | **One parameterised training script** — only 2 branches on method |
| 58 | Leakage guard? | Evaluator **imports** `MEAN`, `STD`, `make_model` from `train.py` |
| 59 | Warm-up artefact? | Pilot baseline 42.2 s vs RandAugment 20.9 s — **first-run GPU init** |
| 60 | Timing anomaly? | AugMix seed 23 epoch 45 = **865.9 s**; slow epochs come in **contiguous blocks** |

---

## R. The 12 hardest questions, ranked

If a professor asks only one thing, it will likely be one of these. Rehearse these answers until they
are fluent.

| Rank | Question | What they are testing | One-line answer |
|---:|---|---|---|
| 1 | **AugMix got 3× the compute. Isn't the win just extra training?** | Whether you see your own central confound | "Yes, and I cannot rule it out. A compute-matched control is my top follow-up." |
| 2 | **Two seeds — what can you possibly conclude?** | Statistical honesty | "Only large, stable effects. AugMix's +10.78 pp is 35× its own SD; the 0.37 pp clean difference is not real." |
| 3 | **Your timings vary 2× under identical settings. Why should I believe them?** | Measurement rigour | "I don't, and I say so. The medians show the noise is episodic; my reported means are inflated by interference, not by the methods." |
| 4 | **What is your novelty?** | Research honesty | "None. A 2026 paper covers this exact combination, so I repositioned as a reproducible pilot." |
| 5 | **Did you implement AugMix, or just the transform?** | Technical depth | "The full method: 3 views per image plus the 12 × JSD consistency loss. The transform alone is not AugMix." |
| 6 | **Why is your training loss/accuracy not comparable across methods?** | Whether you understand your own logs | "AugMix's logged loss includes 12 × JSD, and train accuracy is measured on different input distributions." |
| 7 | **How do you know there is no test-set leakage?** | Experimental hygiene | "Final-epoch checkpoint, no tuning, defaults frozen in a dated protocol, no validation split, test touched once." |
| 8 | **Why did you change the first conv stride, and what does it cost?** | Architecture understanding | "To avoid destroying 32×32 inputs. It makes early feature maps 4× larger, inflating all timings — but equally across conditions." |
| 9 | **Your archive has 19 corruptions; why 15?** | Whether you cherry-picked | "The canonical benchmark has 15. The extra 4 are from a later release. Hard-coded set, errors if any canonical file is missing." |
| 10 | **Is 86 % good for MobileNetV3-Small?** | Calibration of claims | "Reasonable for 50 epochs with untuned defaults — and all three conditions tie, so the ranking is unaffected." |
| 11 | **Why not AdamW / why lr 0.05 / why 2 seeds / why one model?** | Whether choices were justified or arbitrary | "All were chosen for comparability with the CIFAR literature or the compute budget — never for the result." |
| 12 | **Is there anything wrong with your own paper?** | Intellectual honesty | "Yes — 'most late epochs took 80–110 s' is true for seed 23 but false for seed 17, whose late epochs took ~310–325 s." |

---

## Final checklist before any interview

- [ ] I can state the research question in one sentence.
- [ ] I can recite all 60 numbers in §Q without looking.
- [ ] I can explain AugMix's JSD loss from memory, including why JSD and not KL.
- [ ] I can say **"yes, AugMix got 3× the compute and that is a confound"** unprompted.
- [ ] I can say **"no, I make no novelty claim"** unprompted.
- [ ] I can explain the timing bimodality and the median-epoch re-analysis.
- [ ] I can volunteer the §Q103 imprecision in my own paper.
- [ ] I can describe what I personally did, in detail, without vague phrasing.
- [ ] I never call the Zenodo preprint peer-reviewed.
- [ ] I never present the training-accuracy numbers as a comparison.
- [ ] When I don't know, I say so and say what I would do to find out.

---

*Companion document: `overview.md` — the full plain-language explanation of the research, all result
tables, tools and their usage, limitations, and follow-up plans.*
