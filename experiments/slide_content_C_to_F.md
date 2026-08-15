# Slide content — Configs A, C, D, E, F

Test image (same for all): `dataset/downloaded/polyp_wikimedia.jpeg` — a real
colon-polyp endoscopy photo (CC-BY-2.5, Stephen Holland M.D., via Wikimedia
Commons), resized to 512×512. Same image, same metric code
(`enhance.compute_metrics()`), for every config below — numbers are directly
comparable across slides.

Every slide below shows **Original vs. Output** — the original image's own
measurements are identical across all five slides (same input image every
time), included on each slide so it stands alone.

---

## Slide: Config A — Paper's own method (baseline)

**Image:** [`output_A_original_vs_enhanced.png`](output_A_original_vs_enhanced.png)

**Pipeline:**
1. RGB → HSV, extract V channel
2. Normalize → invert → gamma correction ([Paper A](https://doi.org/10.1007/s11042-026-21507-z), γ = 0.8, Eq. 1–3)
3. CLAHE contrast enhancement (Paper A)
4. Hu-WBI 2× upsample (Paper A, Eq. 4)
5. Downsample back to original size
6. Unsharp mask (Eq. 7)
7. SRGAN-v super-resolution
8. Invert back, merge to RGB
9. Colour transfer (Eq. 6)

**Original vs. Output:**

| Metric | Original | Output | Change |
|---|---:|---:|---:|
| Brightness | 84.85 | 84.19 | ≈ same |
| Contrast (std-dev) | 59.19 | 57.74 | ≈ same |
| Entropy | 7.557 | 7.594 | ≈ same |
| EME (local contrast) | 20.95 | 49.38 | ↑ 2.4× |
| Avg. gradient | 17.63 | 32.91 | ↑ 1.9× |
| Laplacian variance (sharpness) | 32.84 | 222.19 | ↑ 6.8× |
| Colourfulness | 66.67 | 63.32 | ↓ |
| Edge density (%) | 0.95 | 3.14 | ↑ 3.3× |
| Michelson contrast | 0.980 | 1.000 | ↑ |

**Fidelity (Original ↔ Output difference):** PSNR 23.29 dB · SSIM 0.812 · LPIPS 0.226 · ΔE2000 5.197 · MAE 11.86 · UQI 0.964

**Note:** numbers updated 2026-08-15 after fixing colour-transfer strength to 1.0 (full Eq. 6, no blending — the old 0.6 default was a tuning knob not defined in the paper).

---

## Slide: Config C — AFGT + CLAHE

**Image:** [`output_C_original_vs_enhanced.png`](output_C_original_vs_enhanced.png)

**What it is:** [Paper A](https://doi.org/10.1007/s11042-026-21507-z)'s pipeline shape, with [Paper B](https://arxiv.org/pdf/2009.12631)'s AFGT tone-curve substituted for Paper A's invert+gamma step. Contrast stage left as Paper A's own CLAHE.

**Pipeline:**
1. RGB → HSV, extract V channel
2. AFGT tone-curve ([Paper B / Long et al. 2018](https://ieeexplore.ieee.org/document/8584793)) — replaces invert + gamma
3. CLAHE contrast enhancement (Paper A)
4. Hu-WBI 2× upsample (Paper A)
5. Downsample back to original size
6. Unsharp mask
7. SRGAN-v super-resolution
8. Merge back to RGB
9. Colour transfer (Eq. 6, Paper A)

**Original vs. Output:**

| Metric | Original | Output | Change |
|---|---:|---:|---:|
| Brightness | 84.85 | 83.76 | ↓ |
| Contrast (std-dev) | 59.19 | 53.42 | ↓ |
| Entropy | 7.557 | 7.585 | ≈ same |
| EME (local contrast) | 20.95 | 47.31 | ↑ 2.3× |
| Avg. gradient | 17.63 | 32.96 | ↑ 1.9× |
| Laplacian variance (sharpness) | 32.84 | 213.28 | ↑ 6.5× |
| Colourfulness | 66.67 | 59.62 | ↓ |
| Edge density (%) | 0.95 | 3.11 | ↑ 3.3× |
| Michelson contrast | 0.980 | 0.990 | ↑ |

**Fidelity (Original ↔ Output difference):** PSNR 21.44 dB · SSIM 0.814 · LPIPS 0.236 · ΔE2000 6.729 · MAE 14.90 · UQI 0.942

**Note:** weakest of the four on fidelity — CLAHE over-stretches contrast on the already-brightened (non-inverted) AFGT output, visible as a purple/blue color-cast artifact in the output image.

---

## Slide: Config D — AFGT + AGCWD

**Image:** [`output_D_original_vs_enhanced.png`](output_D_original_vs_enhanced.png)

**What it is:** Same shape as C, but CLAHE swapped for AGCWD ([Huang, Cheng & Chiu, IEEE TIP 2013](https://doi.org/10.1109/TIP.2012.2226047)).

**Pipeline:**
1. RGB → HSV, extract V channel
2. AFGT tone-curve ([Paper B / Long et al. 2018](https://ieeexplore.ieee.org/document/8584793))
3. AGCWD contrast enhancement ([Huang et al. 2013](https://doi.org/10.1109/TIP.2012.2226047))
4. Hu-WBI 2× upsample (Paper A)
5. Downsample back to original size
6. Unsharp mask
7. SRGAN-v super-resolution
8. Merge back to RGB
9. Colour transfer (Eq. 6, Paper A)

**Original vs. Output:**

| Metric | Original | Output | Change |
|---|---:|---:|---:|
| Brightness | 84.85 | 92.39 | ↑ |
| Contrast (std-dev) | 59.19 | 58.09 | ≈ same |
| Entropy | 7.557 | 7.539 | ≈ same |
| EME (local contrast) | 20.95 | 36.04 | ↑ 1.7× |
| Avg. gradient | 17.63 | 19.08 | ↑ |
| Laplacian variance (sharpness) | 32.84 | 66.26 | ↑ 2.0× |
| Colourfulness | 66.67 | 69.60 | ↑ |
| Edge density (%) | 0.95 | 0.96 | ≈ same |
| Michelson contrast | 0.980 | 0.990 | ↑ |

**Fidelity (Original ↔ Output difference):** PSNR 26.06 dB · SSIM 0.934 · LPIPS 0.109 · ΔE2000 4.402 · MAE 9.67 · UQI 0.986

**Note:** closest to the paper's own claimed PSNR/SSIM (26.4 / 0.923), and the only config that visibly, obviously enhances the image — brightness genuinely up (not just less lost), shadows lifted, more surface detail, warmer color. (Unaffected by the colour-transfer fix below — D hardcodes its own strength=0.6.)

---

## Slide: Config E — stretch + Hu-WBI-literal

**Image:** [`output_E_original_vs_enhanced.png`](output_E_original_vs_enhanced.png)

**What it is:** Paper A's own gamma tone-curve (not AFGT), contrast swapped to a plain percentile stretch, and Hu-WBI's literal (as-printed) equation instead of the corrected/averaged version.

**Pipeline:**
1. RGB → HSV, extract V channel
2. Invert + gamma correction (Paper A, γ = 0.8)
3. Percentile stretch contrast (generic, not paper-derived)
4. Hu-WBI-literal 2× upsample (Eq. 4 as literally printed)
5. Downsample back to original size
6. Unsharp mask
7. SRGAN-v super-resolution
8. Invert back
9. Merge back to RGB
10. Colour transfer (Eq. 6, Paper A)

**Original vs. Output:**

| Metric | Original | Output | Change |
|---|---:|---:|---:|
| Brightness | 84.85 | 83.79 | ≈ same |
| Contrast (std-dev) | 59.19 | 58.08 | ≈ same |
| Entropy | 7.557 | 7.005 | ↓ |
| EME (local contrast) | 20.95 | 11.65 | ↓ |
| Avg. gradient | 17.63 | 17.29 | ≈ same |
| Laplacian variance (sharpness) | 32.84 | 88.73 | ↑ 2.7× |
| Colourfulness | 66.67 | 63.22 | ↓ |
| Edge density (%) | 0.95 | 1.32 | ↑ |
| Michelson contrast | 0.980 | 0.838 | ↓ |

**Fidelity (Original ↔ Output difference):** PSNR 28.94 dB · SSIM 0.922 · LPIPS 0.137 · ΔE2000 3.363 · MAE 6.73 · UQI 0.990

**Note:** numbers updated 2026-08-15 after the colour-transfer fix (strength 0.6→1.0) — E's PSNR jumped from 25.86 to 28.94, now *beating* D (26.06). EME and entropy still drop below the original either way — the output barely differs visually and by some measures is less locally-detailed than the input. `stretch` is gentle by construction, so this remains a "changed the image least" win, not a real enhancement win, even though the fidelity numbers improved further.

---

## Slide: Config F — AFGT + stretch + Hu-WBI-literal

**Image:** [`output_F_original_vs_enhanced.png`](output_F_original_vs_enhanced.png)

**What it is:** Combines AFGT ([Paper B](https://arxiv.org/pdf/2009.12631)) with the percentile stretch contrast and the literal Hu-WBI equation ([Paper A](https://doi.org/10.1007/s11042-026-21507-z), Eq. 4). Found via a 28-combination grid search as the best-scoring config overall.

**Pipeline:**
1. RGB → HSV, extract V channel
2. AFGT tone-curve ([Paper B / Long et al. 2018](https://ieeexplore.ieee.org/document/8584793))
3. Percentile stretch contrast (generic, not paper-derived)
4. Hu-WBI-literal 2× upsample (Eq. 4 as literally printed)
5. Downsample back to original size
6. Unsharp mask
7. SRGAN-v super-resolution
8. Merge back to RGB
9. Colour transfer (Eq. 6, Paper A)

**Original vs. Output:**

| Metric | Original | Output | Change |
|---|---:|---:|---:|
| Brightness | 84.85 | 84.75 | ≈ same |
| Contrast (std-dev) | 59.19 | 58.06 | ≈ same |
| Entropy | 7.557 | 7.561 | ≈ same |
| EME (local contrast) | 20.95 | 38.39 | ↑ 1.8× |
| Avg. gradient | 17.63 | 18.99 | ↑ |
| Laplacian variance (sharpness) | 32.84 | 80.93 | ↑ 2.5× |
| Colourfulness | 66.67 | 66.41 | ≈ same |
| Edge density (%) | 0.95 | 1.08 | ≈ same |
| Michelson contrast | 0.980 | 1.000 | ↑ |

**Fidelity (Original ↔ Output difference):** PSNR 33.99 dB · SSIM 0.952 · LPIPS 0.054 · ΔE2000 1.737 · MAE 3.67 · UQI 0.997

**Note:** best PSNR, SSIM, and ΔE2000 of every config tested (including the paper's own claimed numbers) — brightness/contrast/colourfulness stay almost exactly at the original's level while EME and sharpness genuinely rise, so unlike E this is a real (if visually subtle) local-contrast gain, not just a near-identity map.

---

## Quick comparison table (fidelity metrics, all five)

*(A and E updated 2026-08-15 — see colour-transfer fix note below)*

| Config | PSNR | SSIM | LPIPS | ΔE2000 | EME (orig → out) |
|---|---:|---:|---:|---:|---|
| A: Paper's own method | 23.29 | 0.812 | 0.226 | 5.197 | 20.95 → 49.38 |
| C: AFGT + CLAHE | 21.44 | 0.814 | 0.236 | 6.729 | 20.95 → 47.31 |
| D: AFGT + AGCWD | 26.06 | 0.934 | 0.109 | 4.402 | 20.95 → 36.04 |
| E: stretch + Hu-WBI-literal | 28.94 | 0.922 | 0.137 | 3.363 | 20.95 → 11.65 |
| **F: AFGT + stretch + Hu-WBI-literal** | **33.99** | **0.952** | **0.054** | **1.737** | 20.95 → 38.39 |
| Paper's claimed numbers (CVC-Clinic / real-time) | 26.44 | 0.923 / 0.955 | 0.132 / 0.131 | — | — |

**Colour-transfer fix (2026-08-15):** `color_transfer()` (Eq. 6) defaulted to `strength=0.6` — a partial blend not defined anywhere in the paper. Fixed to `strength=1.0` (full Eq. 6, as printed) in `enhance.py`/`app.py`/the UI. This changed **A** and **E** (both call `enhance()` without overriding the parameter); **C, D, F** are unaffected since their scripts hardcode `strength=0.6` explicitly as a deliberate choice for those hybrids. Net effect: E's PSNR rose from 25.86 to 28.94, overtaking D.

---

## References (paper links)

| Paper | Role | Link |
|---|---|---|
| Jagarajan & Jayaraman (2026), *Multimedia Tools and Applications* 85:477 | Paper A — the base pipeline (HSV, invert+γ, CLAHE, Hu-WBI, SRGAN, colour transfer) | https://doi.org/10.1007/s11042-026-21507-z |
| Ezatian, Khaledyan, Jafari, Heidari, Zargari Khuzani, Mashhadi, arXiv:2009.12631 | Paper B — AFGT + Unsharp Masking pipeline (used in C, D, F) | https://arxiv.org/pdf/2009.12631 |
| Long, Lan, Xie, Li, Wang, IEEE BioCAS 2018 | AFGT's core equations (cited by Paper B) | https://ieeexplore.ieee.org/document/8584793 |
| Huang, Cheng & Chiu, IEEE TIP 22(3), 2013 | AGCWD contrast method (used in D) | https://doi.org/10.1109/TIP.2012.2226047 |
| Sheet, Garud, Suveer, Mahadevappa, Chatterjee, IEEE TCE 56(4), 2010 | BPDFHE (explored earlier in the broader search) | https://ieeexplore.ieee.org/abstract/document/5681130 |
| Abdullah-Al-Wadud, Kabir, Dewan, Chae, IEEE TCE 53(2), 2007 | DHE (cited by BPDFHE for its per-partition equalization) | https://dl.acm.org/doi/10.1109/tce.2007.381734 |
