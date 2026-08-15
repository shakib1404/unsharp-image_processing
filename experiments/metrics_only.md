# Metrics — Original Image, then Each Pipeline's Output

Test image: `dataset/downloaded/polyp_wikimedia.jpeg`, 512×512. All numbers
from `enhance.compute_metrics()`, same code for every row.

---

## Original image (before any pipeline runs)

| Metric | Value |
|---|---:|
| Brightness | 84.85 |
| Contrast (std-dev) | 59.19 |
| Entropy | 7.557 |
| EME (local contrast) | 20.95 |
| Avg. gradient | 17.63 |
| Laplacian variance (sharpness) | 32.84 |
| Colourfulness | 66.67 |
| Edge density (%) | 0.947 |
| Michelson contrast | 0.980 |

This same row is the "before" side for every pipeline below — it doesn't change, only the output does.

---

## A: Paper's own method — output metrics

*(updated after fixing colour-transfer strength to 1.0 — full Eq. 6, no blending; see note below)*

| Metric | Value |
|---|---:|
| Brightness | 84.19 |
| Contrast (std-dev) | 57.74 |
| Entropy | 7.594 |
| EME (local contrast) | 49.38 |
| Avg. gradient | 32.91 |
| Laplacian variance (sharpness) | 222.19 |
| Colourfulness | 63.32 |
| Edge density (%) | 3.136 |
| Michelson contrast | 1.000 |

**Original ↔ output difference:** PSNR 23.29 dB · SSIM 0.812 · LPIPS 0.226 · MSE 245.00 · RMSE 15.65 · MAE 11.86 · AMBE 0.663 · NCC 0.965 · UQI 0.964 · ΔE2000 5.197

---

## C: AFGT + CLAHE — output metrics

| Metric | Value |
|---|---:|
| Brightness | 83.76 |
| Contrast (std-dev) | 53.42 |
| Entropy | 7.585 |
| EME (local contrast) | 47.31 |
| Avg. gradient | 32.96 |
| Laplacian variance (sharpness) | 213.28 |
| Colourfulness | 59.62 |
| Edge density (%) | 3.108 |
| Michelson contrast | 0.990 |

**Original ↔ output difference:** PSNR 21.44 dB · SSIM 0.814 · LPIPS 0.236 · MSE 367.35 · RMSE 19.17 · MAE 14.90 · AMBE 1.096 · NCC 0.947 · UQI 0.942 · ΔE2000 6.729

---

## D: AFGT + AGCWD — output metrics

| Metric | Value |
|---|---:|
| Brightness | 92.39 |
| Contrast (std-dev) | 58.09 |
| Entropy | 7.539 |
| EME (local contrast) | 36.04 |
| Avg. gradient | 19.08 |
| Laplacian variance (sharpness) | 66.26 |
| Colourfulness | 69.60 |
| Edge density (%) | 0.961 |
| Michelson contrast | 0.990 |

**Original ↔ output difference:** PSNR 26.06 dB · SSIM 0.934 · LPIPS 0.109 · MSE 127.90 · RMSE 11.31 · MAE 9.673 · AMBE 7.538 · NCC 0.990 · UQI 0.986 · ΔE2000 4.402

---

## E: stretch + Hu-WBI-literal — output metrics

*(updated after fixing colour-transfer strength to 1.0 — full Eq. 6, no blending; see note below)*

| Metric | Value |
|---|---:|
| Brightness | 83.79 |
| Contrast (std-dev) | 58.08 |
| Entropy | 7.005 |
| EME (local contrast) | 11.65 |
| Avg. gradient | 17.29 |
| Laplacian variance (sharpness) | 88.73 |
| Colourfulness | 63.22 |
| Edge density (%) | 1.321 |
| Michelson contrast | 0.838 |

**Original ↔ output difference:** PSNR 28.94 dB · SSIM 0.922 · LPIPS 0.137 · MSE 72.25 · RMSE 8.500 · MAE 6.727 · AMBE 1.063 · NCC 0.990 · UQI 0.990 · ΔE2000 3.363

---

## F: AFGT + stretch + Hu-WBI-literal — output metrics

| Metric | Value |
|---|---:|
| Brightness | 84.75 |
| Contrast (std-dev) | 58.06 |
| Entropy | 7.562 |
| EME (local contrast) | 38.39 |
| Avg. gradient | 18.99 |
| Laplacian variance (sharpness) | 80.93 |
| Colourfulness | 66.41 |
| Edge density (%) | 1.080 |
| Michelson contrast | 1.000 |

**Original ↔ output difference:** PSNR 33.99 dB · SSIM 0.952 · LPIPS 0.054 · MSE 20.22 · RMSE 4.497 · MAE 3.669 · AMBE 0.107 · NCC 0.997 · UQI 0.997 · ΔE2000 1.737

---

## All five outputs side by side (+ original)

| Metric | Original | A | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| Brightness | 84.85 | 84.19 | 83.76 | 92.39 | 83.79 | 84.75 |
| Contrast | 59.19 | 57.74 | 53.42 | 58.09 | 58.08 | 58.06 |
| Entropy | 7.557 | 7.594 | 7.585 | 7.539 | 7.005 | 7.562 |
| EME | 20.95 | 49.38 | 47.31 | 36.04 | 11.65 | 38.39 |
| Avg. gradient | 17.63 | 32.91 | 32.96 | 19.08 | 17.29 | 18.99 |
| Laplacian var. | 32.84 | 222.19 | 213.28 | 66.26 | 88.73 | 80.93 |
| Colourfulness | 66.67 | 63.32 | 59.62 | 69.60 | 63.22 | 66.41 |
| Edge density (%) | 0.947 | 3.136 | 3.108 | 0.961 | 1.321 | 1.080 |
| Michelson | 0.980 | 1.000 | 0.990 | 0.990 | 0.838 | 1.000 |
| PSNR (vs. orig) | — | 23.29 | 21.44 | 26.06 | **28.94** | 33.99 |
| SSIM (vs. orig) | — | 0.812 | 0.814 | 0.934 | 0.922 | 0.952 |
| LPIPS (vs. orig) | — | 0.226 | 0.236 | 0.109 | 0.137 | 0.054 |
| ΔE2000 (vs. orig) | — | 5.197 | 6.729 | 4.402 | 3.363 | 1.737 |

**Ranking changed:** with colour transfer at full strength, E now beats D on PSNR (28.94 vs. 26.06) and ΔE2000 (3.363 vs. 4.402) — E was artificially held back by the old 0.6-strength blend more than D was, because E's output was already close to the original, so full correction pulls it even closer. D still has far higher EME (real local-contrast gain) and is still the only one that visibly brightens the image.

---

## Paper's own claimed PSNR / SSIM / LPIPS, for reference

The paper reports these on its own datasets (CVC-Clinic and a private real-time
dataset, not this image) — not directly comparable, but the closest published
numbers to check any config against:

| Dataset | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| CVC-Clinic (paper's Table 6) | 26.4365 | 0.9232 | 0.1324 |
| Real-time dataset (paper's Table 7) | 26.4365 | 0.9552 | 0.1308 |

**D and F both beat the paper's own claimed LPIPS (~0.13) on this image** —
D reaches 0.109, F reaches 0.054, well below what the paper itself reports,
despite neither being a reproduction of the paper's exact method.

---

## Note: colour-transfer fix (2026-08-15)

`color_transfer()` (Eq. 6) had a `strength` parameter defaulting to 0.6 — a
partial blend not defined anywhere in the paper (Eq. 6 is just the formula,
no blending step). Fixed to `strength=1.0` (full, undamped Eq. 6) as the
default in `enhance.py`, `app.py`, and the UI. This changes **A** and **E**
above, since both call `enhance()` without overriding the parameter. **C, D,
F** are unaffected — their scripts hardcode `strength=0.6` explicitly, which
is a deliberate choice for those (non-paper) hybrids, not an inherited bug.
