# Endoscopy Enhancement — Paper A vs. Paper B vs. Combined, Experimental Report

> **Correction (2026-08-15):** the full Paper A text (not just the abstract) revealed that `enhance.py` had CLAHE and Hu-WBI **swapped** — it ran Hu-WBI-then-CLAHE, but the paper's abstract, phase list, and Fig. 1 all agree on CLAHE-then-Hu-WBI (Hu-WBI's whole stated purpose is fixing "artifact issues associated with CLAHE," which requires CLAHE to run first). Fixed in `enhance.py`. Configs **C, D, and F** additionally had this same order hand-duplicated in `experiments/paper_comparison.py` and `experiments/search_best_pipeline.py` rather than calling the shared (now-fixed) function, so they didn't inherit the fix automatically — those were fixed too. **Every table below was regenerated after the fix**; numbers differ somewhat from earlier versions of this report but the overall conclusions (which config wins, why) are unchanged.

## 1. What this covers

Two independent, fully-specified enhancement pipelines were run and compared on
a real endoscopy image, along with a per-stage ablation of Paper A's pipeline
and a hand-built combination of the two papers' ideas.

| | |
|---|---|
| Test image | `dataset/downloaded/polyp_wikimedia.jpeg` — a **real** colon-polyp endoscopy photo downloaded from Wikimedia Commons ([File:Polyp.jpeg](https://commons.wikimedia.org/wiki/File:Polyp.jpeg)), CC‑BY‑2.5, credit Stephen Holland, M.D. (Naperville Gastroenterology). Not one of the sample images already sitting in the repo. |
| Work size | 512×512 (matches Paper A's own working resolution) |
| Metrics | `enhance.compute_metrics()` — PSNR, SSIM, ΔE2000 (fidelity vs. original) and entropy/EME/CII/colourfulness (no-reference, on the enhanced image) — same metric code used for every config below, so all numbers are directly comparable to each other |

---

## 2. The two papers

### Paper A — the project's existing pipeline
**Jagarajan & Jayaraman (2026)**, *Multimedia Tools and Applications* 85:477, DOI 10.1007/s11042-026-21507-z.
Already implemented in [`enhance.py`](../enhance.py) before this session: HSV → normalize → **invert** → gamma correction (γ=0.8) → Hu-WBI 2× upsample → CLAHE → downsample → unsharp mask → SRGAN → colour transfer (Eq. 6). `srgan_v.pth` here is a checkpoint trained by this project's own `train_srgan_v.py` (paper's original weights were never released).

### Paper C — a third paper, added in a follow-up round
**D. Sheet, H. Garud, A. Suveer, M. Mahadevappa, J. Chatterjee**, *"Brightness Preserving Dynamic Fuzzy Histogram Equalization (BPDFHE),"* IEEE Trans. Consumer Electronics 56(4), 2475–2480, 2010.
Its per-partition equalization step (Eq.6–11) is itself the **Dynamic Histogram Equalization (DHE)** technique from a fourth paper: **M. Abdullah-Al-Wadud, M. H. Kabir, M. A. A. Dewan, O. Chae**, *"A Dynamic Histogram Equalization for Image Contrast Enhancement,"* IEEE Trans. Consumer Electronics 53(2), 593–600, 2007 — BPDFHE's own contribution is computing the histogram in the *fuzzy* domain first (Eq.1–2, triangular membership, so noisy bins don't fragment the local-maxima partitioning) and a final brightness-restoring normalization (Eq.12).

Implemented in [`experiments/paper_c_bpdfhe.py`](paper_c_bpdfhe.py) as a drop-in contrast-method function. Sanity-checked against a reference Python port ([msrinivaskgp/BPDFHE-Python](https://github.com/msrinivaskgp/BPDFHE-Python)) and the original PDF (included in that same repo) for the equations. On the real downloaded image it detects 17 local maxima (a noisier histogram than the paper's own smooth photographic test images), so it ends up close to an identity map on its own (mean pixel shift 1.8/255) — a real, correctly-implemented behaviour of the algorithm on this image, not a bug; see Section 4b.

### Paper B — newly researched and implemented this session
**Rezvan Ezatian, Donya Khaledyan, Kian Jafari, Morteza Heidari, Abolfazl Zargari Khuzani, Najmeh Mashhadi**, *"Image quality enhancement in wireless capsule endoscopy with Adaptive Fraction Gamma Transformation and Unsharp Masking filter"*, [arXiv:2009.12631](https://arxiv.org/pdf/2009.12631).
Its core nonlinearity (AFGT, Eq. 2–3) is itself cited from an earlier paper: **M. Long, Z. Lan, X. Xie, G. Li, Z. Wang**, *"Image Enhancement Method Based on Adaptive Fraction Gamma Transformation and Color Restoration for Wireless Capsule Endoscopy,"* IEEE BioCAS 2018. Ezatian et al.'s own contribution is the adaptive-β derivation from the image's histogram (Eq. 4–10), the Unsharp Masking filter (Eq. 11), and combining the three stages into one pipeline.

Pipeline: RGB → HSI, intensity `I` → normalize (Eq.1) → **AFGT** (Eq.2‑3, per-pixel adaptive γ = 1+arctan(Iₙ−0.5), no inversion needed — the arctan-gamma already brightens dark regions directly) → **Unsharp Masking** (Eq.11) → **colour restoration** by ratio scaling (Eq.12) + per-channel min-max stretch (Eq.13).

Implemented from scratch, equation-by-equation, in [`experiments/paper_b_afgt.py`](paper_b_afgt.py). It reuses `enhance.py`'s existing `split_intensity/merge_intensity(space="hsi")` for Eq.12, since that ratio-restore logic already existed in this codebase for an unrelated reason (the "hsi"/"gray" colour-space option) and turned out to implement exactly what Paper B's Eq.12 specifies. No training/weights needed — every step is closed-form.

Paper B's own claimed numbers (Table I, KID capsule-endoscopy dataset, 300 images, 360×360, vs. their **Proposed**): PSNR 28.91 dB, SSIM 0.95 — for context, not directly comparable since it's a different dataset.

---

## 3. Head-to-head: Paper A vs. Paper B vs. combined (real image, 512×512)

Script: [`experiments/paper_comparison.py`](paper_comparison.py) · Data: [`paper_comparison.csv`](paper_comparison.csv) · Side-by-side PNGs: `experiments/cmp_*.png`

| Config | PSNR | SSIM | Entropy | EME | CII | Colourfulness | ΔE2000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: Paper A default (HSV, invert+γ, CLAHE, Hu-WBI, SRGAN-v, colour xfer) | 23.01 | 0.826 | 7.564 | **44.02** | 0.932 | 61.05 | 5.282 |
| B: Paper B standalone (HSI, AFGT, UM, ratio colour restore) | 24.61 | 0.925 | 7.551 | 21.85 | 0.879 | 65.59 | 5.127 |
| C: Hybrid — A's structure, AFGT swapped in for invert+γ | 21.44 | 0.814 | 7.585 | 47.31 | 0.902 | 59.62 | 6.729 |
| **D: Hybrid — A's structure, AFGT swapped in for invert+γ, AGCWD instead of CLAHE** | **26.06** | **0.934** | 7.539 | 36.04 | **0.981** | **69.60** | **4.402** |

**Reading it:** Paper B alone already beats Paper A's default on fidelity (PSNR/SSIM/ΔE) here — plausible, since Paper A's `srgan_v.pth` is a self-trained proxy for weights the original authors never released, while Paper B needs no learned weights at all and so reproduces identically for anyone. Naively swapping AFGT into Paper A's shell (C) does **not** help — it actually loses fidelity, because CLAHE combined with an already-brightened (non-inverted) channel over-stretches contrast. Also swapping CLAHE → AGCWD (D) fixes that: **config D reaches PSNR 26.06 / SSIM 0.934, close to Paper A's own claimed 26.4 dB / 0.923** — obtained here from two closed-form, un-trained equations (AFGT + AGCWD) plus the trained SRGAN, rather than relying on the paper's unreleased weights. D also wins on CII, colourfulness and ΔE2000 outright.

Caveat: EME (local-contrast) is highest for A/C, lower for B/D — B and D stay closer to the original structurally (which is what PSNR/SSIM reward) while pushing local contrast less hard than CLAHE-based configs. Which is "better" depends on whether the goal is fidelity or aggressive contrast boosting; see `enhance.py`'s own `METRIC_INFO` notes on this tension.

A follow-up single try (**E: HSV + `stretch` contrast + Hu-WBI-**literal** + SRGAN + colour transfer**, i.e. Paper A's own gamma tone-curve, not AFGT) scored PSNR 25.86 / SSIM 0.9204 / CII 0.991 / ΔE2000 4.629 — beats Paper A's default and lands near D, but EME drops to 14.01 (vs. D's 36.04): visually it barely changes the image. Simple percentile stretch is gentle by construction, so this win is partly "changed the image less," not purely "enhanced it more."

### 3b. Broader search — is there a config that's both high-fidelity AND visibly enhanced?

Script: [`experiments/search_best_pipeline.py`](search_best_pipeline.py) · Data: [`search_best_pipeline.csv`](search_best_pipeline.csv) — full **2 tone-curves × 7 contrast methods × 2 upsample methods = 28-combo** grid (SRGAN-v + colour transfer fixed on throughout), adding Paper C (BPDFHE) as another contrast option alongside CLAHE/AGCWD/stretch/HE/BBHE/DSIHE.

Top result by a combined PSNR+SSIM score:

| Config | PSNR | SSIM | Entropy | EME | CII | ΔE2000 |
|---|---:|---:|---:|---:|---:|---:|
| **F: afgt+stretch+hu_wbi_literal** | **33.99** | 0.952 | 7.561 | **38.39** | 0.981 | **1.737** |
| afgt+bpdfhe+hu_wbi | 32.35 | 0.958 | 7.543 | 34.22 | 0.956 | 2.251 |
| gamma+stretch+hu_wbi | 31.16 | 0.952 | 7.245 | 15.79 | 0.993 | 2.799 |
| afgt+stretch+hu_wbi | 30.24 | **0.973** | 7.544 | 18.88 | 0.974 | 2.268 |
| gamma+bpdfhe+hu_wbi | 30.41 | 0.949 | 7.455 | 15.33 | 0.979 | 2.905 |

**Config F** (AFGT tone-curve + `stretch` contrast + Hu-WBI-**literal** upsample + SRGAN-v + colour transfer) is the standout: it has the *highest PSNR in the whole 28+5-config search* (33.99, vs. the paper's claimed 26.4) **and** keeps EME high (38.39 — comparable to the visually-aggressive CLAHE configs, not a near-identity map like plain "stretch" was under the gamma tone-curve). Checked visually ([`cmp_F_afgt_stretch_huwbiliteral.png`](cmp_F_afgt_stretch_huwbiliteral.png)) — it's still a fairly subtle-looking change, so EME's near-doubling reflects a genuine local micro-contrast/texture shift (likely from how SRGAN's sharpening interacts with the AFGT-then-stretch input) more than an obviously "punchier" picture.

**So, honestly: no single config wins on every axis.**
- Want to match/beat the paper's claimed PSNR/SSIM as closely as possible, with a real (not trivial) EME gain → **config F**.
- Want a visually obvious, more aggressively enhanced-looking result → **config D** (AFGT + AGCWD), which still lands close to the paper's claim (26.1/0.934) while looking clearly brighter/higher-contrast in the side-by-side.

---

## 4. Paper A per-stage ablation (real image, everything else = paper default)

Scripts: [`experiments/real_image_ablation.py`](real_image_ablation.py), [`real_sr_and_gamma.py`](real_sr_and_gamma.py) — paper default row repeated in each table for reference.

**Colour space** ([csv](real_ablation_color_space.csv)) — after the fix, HSV (paper) is now actually the PSNR winner (23.01) among all 7 spaces, though Lab/HSI edge it slightly on SSIM; YIQ/YCrCb/YUV score noticeably better on colourfulness. HSV keeps the lowest ΔE.
**Upsample** ([csv](real_ablation_upsample.csv)) — the whole axis is much flatter now than before the fix: Hu-WBI (paper, 23.01) sits within ~0.1 dB of the best option (bilinear, 23.09), instead of ~1 dB behind `nearest`/`none` as it was pre-fix. Makes sense — Hu-WBI now runs *after* CLAHE has already done the contrast work, so which interpolation kernel finishes the upsample matters much less.
**Contrast** ([csv](real_ablation_contrast.csv)) — CLAHE is still far from the strongest option on this image: `stretch` and `none` top PSNR/SSIM (expected — they change the image least), while `he`/`bbhe`/`dsihe` land a strong PSNR≈29/SSIM≈0.91 with real contrast gain. `clahe_sk` and retinex (`ssr`/`msr`) are the clear losers.
**SR method** ([csv](real_ablation_sr.csv)) — all SR variants still land within ~0.3 dB of each other and of `none`; SR contributes far less than the contrast-method choice does, same conclusion as before the fix.

## 5. Gamma parameter sweep

`experiments/gamma_sweep.png` (paper-figure image) and `experiments/real_gamma_sweep.png` (real downloaded image) — same experiment, two images, paper's full pipeline otherwise, γ swept 0.2→2.0.

On both images, PSNR is still **not maximised at the paper's γ=0.8** — it now peaks around γ≈1.3 (fig6 image, PSNR 28.56) to γ≈1.5 (real downloaded image, PSNR 24.48), while SSIM keeps climbing well past γ=2.0 on both. Same conclusion as before the pipeline-order fix: γ=0.8 sits on the rising part of both curves rather than at either metric's optimum — it looks like a choice made for qualitative brightening rather than to maximise these particular metrics.

---

## 6. Cross-validation on a second real image (Kvasir v2)

Everything in Section 3 was re-run on a **second, independently-sourced real image**: `dataset/downloaded/kvasir_dyed-lifted-polyps.jpg`, a genuine image from the **official Kvasir v2** dataset (Simula Research Laboratory) — class "dyed-lifted-polyps", 720×576. `datasets.simula.no`'s own TLS certificate chain is broken (`unable to get local issuer certificate`), so it was pulled through the dataset's official Hugging Face mirror (`San-D/Kvasir_V2`) instead. Script: [`experiments/kvasir_full_test.py`](kvasir_full_test.py).

| Config | PSNR | SSIM | Entropy | EME | CII | ΔE2000 |
|---|---:|---:|---:|---:|---:|---:|
| A: Paper A default | 24.48 | 0.860 | 6.705 | 34.28 | 0.984 | 3.737 |
| B: Paper B (AFGT+UM) | 28.19 | 0.931 | 6.661 | **58.61** | 1.003 | 3.041 |
| C: Hybrid (A+AFGT) | 24.57 | 0.830 | 7.000 | 74.22 | 0.960 | 4.003 |
| D: Hybrid (AFGT+AGCWD) | 25.08 | 0.922 | 6.856 | 56.19 | **1.084** | 4.334 |
| **E: HSV+stretch+HuWBI-literal** | **27.42** | 0.868 | 6.623 | 24.03 | 1.017 | **2.968** |
| F: afgt+stretch+HuWBI-literal | 27.04 | **0.935** | 6.744 | 55.41 | 1.018 | 3.344 |

**On this image it's a close call between E and F** — E edges out F on PSNR (27.42 vs. 27.04) and ΔE, F wins SSIM and keeps EME much higher (55 vs. 24, a real contrast gain, not just a gentler change). Both clearly beat Paper A's default and land ahead of D. Paper B alone is again consistently strong (2nd/3rd place both times), reinforcing Section 3's point that a closed-form, no-training-required pipeline can beat Paper A's self-trained SRGAN proxy.

One thing that did **not** transfer: AGCWD, which was the standout contrast method on the Wikimedia image (Config D), does poorly on its own here (PSNR 20.49 in the per-axis contrast table) — a reminder that any single result on one image can be image-dependent, which is exactly why this cross-check matters. `stretch`/`none` again top the fidelity ranking in the per-axis ablation, for the same reason noted in Section 4 (they change the image least). Full ablation tables: `kvasir_ablation_{color_space,upsample_method,contrast_method,sr}.csv`, gamma sweep: `kvasir_gamma_sweep.png`, side-by-sides: `kvasir_cmp_D.png`, `kvasir_cmp_F.png`, `kvasir_paper_default_comparison.png`.

---

## 7. File index

| File | What it is |
|---|---|
| `paper_b_afgt.py` | Paper B, implemented from its equations (importable: `enhance_afgt()`) |
| `paper_c_bpdfhe.py` | Paper C (BPDFHE, built on DHE), implemented from its equations (importable: `bpdfhe()`) |
| `paper_comparison.py` / `.csv` | Section 3 — A vs. B vs. Hybrid C/D, plus config E |
| `search_best_pipeline.py` / `.csv` | Section 3b — 28-combo tone×contrast×upsample grid, winner = config F |
| `cmp_E_*.png`, `cmp_F_*.png` | Side-by-side comparison images for configs E and F |
| `real_image_ablation.py`, `real_sr_and_gamma.py` | Section 4 — per-stage ablation on the real downloaded image |
| `real_ablation_{color_space,upsample,contrast,sr}.csv` | Section 4 raw data |
| `real_paper_default.csv`, `real_paper_default_comparison.png` | Paper A default on the real image |
| `gamma_sweep.py` / `.png`, `real_gamma_sweep.png` | Section 5 |
| `fig6_compare.csv`, `fig6_*.png` | Earlier session: paper-default vs. toggles (SR/colour-transfer/visual-punch) on the paper's own Fig. 6 image, checked against its claimed R‑B redness (93.8) — see `enhance.py`'s `apply_visual_punch()` docstring |
| `cmp_*.png` | Side-by-side comparison images for configs A–D |
| `dataset/downloaded/polyp_wikimedia.jpeg` | Test image 1 (CC‑BY‑2.5, Stephen Holland M.D., via Wikimedia Commons) |
| `kvasir_full_test.py` | Section 6 — full battery re-run on test image 2 |
| `dataset/downloaded/kvasir_dyed-lifted-polyps.jpg` | Test image 2 (official Kvasir v2 dataset, via its Hugging Face mirror) |
| `kvasir_configs_A-F.csv`, `kvasir_ablation_*.csv`, `kvasir_gamma_sweep.png`, `kvasir_cmp_*.png` | Section 6 raw data and figures |
