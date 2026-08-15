"""
Paper B — a second, independent endoscopy enhancement pipeline, implemented
from its own equations for direct comparison against Paper A (the project's
main pipeline in enhance.py).

Rezvan Ezatian, Donya Khaledyan, Kian Jafari, Morteza Heidari, Abolfazl
Zargari Khuzani, Najmeh Mashhadi — "Image quality enhancement in wireless
capsule endoscopy with Adaptive Fraction Gamma Transformation and Unsharp
Masking filter", arXiv:2009.12631.
  - AFGT's core transform (Eq. 2-3 below) is itself cited from an earlier
    paper: M. Long, Z. Lan, X. Xie, G. Li, Z. Wang, "Image Enhancement Method
    Based on Adaptive Fraction Gamma Transformation and Color Restoration for
    Wireless Capsule Endoscopy," IEEE BioCAS 2018.
  - The adaptive-beta derivation (Eq. 4-10), the Unsharp Masking filter
    (Eq. 11) and the full 3-stage pipeline are Ezatian et al.'s contribution.

PIPELINE (paper's Fig. 1):
  RGB -> HSI, take Intensity I
  Eq 1   normalize:  I_n = I / I_max
  Eq 2-3 AFGT:        L  = ( I_n^gamma / (2 - I_n^gamma) )^beta(r_N)
                       gamma(i,j) = 1 + arctan(I_n(i,j) - 0.5)
  Eq 4-10 beta(r_N) derived from the image's own (smoothed) intensity
          histogram -- one adaptive parameter per intensity level, not a
          global constant.
  Eq 11  Unsharp Masking:  r_S = L + 0.8 * (I_n - LPF(I_n))
  Eq 12  Color restoration:  S_c = S_I * (r_S / I_n)   -- ratio scaling,
          reuses enhance.py's split_intensity/merge_intensity(space="hsi"),
          which already implements exactly this ratio-restore rule.
  Eq 13  S_c,m = per-channel min-max stretch back to full range.

No training, no learned weights -- every step is closed-form, so it runs
identically for anyone, unlike Paper A's SRGAN stage.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
from enhance import merge_intensity, split_intensity  # noqa: E402


def _beta_lut(i_n: np.ndarray, i_max_orig: float, bins: int = 256) -> np.ndarray:
    """Eq. 4-10: adaptive beta(r_N), one value per intensity level r_N in [0,1]."""
    hist, edges = np.histogram(i_n, bins=bins, range=(0.0, 1.0))
    pdf = hist.astype(np.float64) / max(i_n.size, 1)          # Eq. 4
    r_n = 0.5 * (edges[:-1] + edges[1:])                       # bin centres

    r_bar = float(np.sum(r_n * pdf))                           # Eq. 8
    variance = float(np.sum((r_n - r_bar) ** 2 * pdf))
    tau = i_max_orig * np.sqrt(max(variance, 0.0)) * 0.01       # Eq. 7

    nz = pdf[pdf > 0]
    if nz.size == 0:
        return np.full(bins, 0.5, dtype=np.float64)
    pdf_min, pdf_max = float(nz.min()), float(pdf.max())
    if pdf_max <= pdf_min:
        pdfs = pdf.copy()
    else:
        norm = np.clip((pdf - pdf_min) / (pdf_max - pdf_min), 0.0, 1.0)
        pdfs = pdf_max * np.power(norm, tau)
        pdfs[pdf <= 0] = 0.0                                    # Eq. 6

    cdf_s = np.cumsum(pdfs) / max(pdfs.sum(), 1e-12)             # Eq. 9
    beta = 1.0 / (1.0 + cdf_s)                                   # Eq. 10
    return beta


def afgt(i_n: np.ndarray, i_max_orig: float) -> np.ndarray:
    """Eq. 2-3: Adaptive Fraction Gamma Transformation on normalized intensity [0,1]."""
    gamma = 1.0 + np.arctan(i_n - 0.5)                           # Eq. 3
    i_n_g = np.power(np.clip(i_n, 1e-6, 1.0), gamma)
    ratio = i_n_g / (2.0 - i_n_g)                                # Eq. 2 (base)

    beta_lut = _beta_lut(i_n, i_max_orig)
    idx = np.clip((i_n * (len(beta_lut) - 1)).round().astype(np.int32),
                   0, len(beta_lut) - 1)
    beta = beta_lut[idx]

    return np.power(np.clip(ratio, 1e-6, None), beta)            # Eq. 2


def unsharp_mask(i_n: np.ndarray, l_afgt: np.ndarray, sigma: float = 1.0,
                  factor: float = 0.8) -> np.ndarray:
    """Eq. 11: r_S = L + factor * (I_n - LPF(I_n)); LPF = Gaussian (paper's threshold factor 0.8)."""
    lpf = cv2.GaussianBlur(i_n, (0, 0), sigma)
    return l_afgt + factor * (i_n - lpf)


def color_restoration(r_s_uint8: np.ndarray, ctx: dict) -> np.ndarray:
    """Eq. 12 (ratio restore, via enhance.py's HSI merge) + Eq. 13 (per-channel min-max stretch)."""
    bgr = merge_intensity(r_s_uint8, ctx).astype(np.float64)
    out = np.empty_like(bgr)
    for c in range(3):
        ch = bgr[..., c]
        lo, hi = ch.min(), ch.max()
        out[..., c] = (ch - lo) * (255.0 / (hi - lo)) if hi > lo else ch
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_afgt(image_bgr: np.ndarray, sigma: float = 1.0,
                  um_factor: float = 0.8) -> np.ndarray:
    """Full Paper-B pipeline: HSI split -> normalize -> AFGT -> UM -> colour restoration."""
    i_u8, ctx = split_intensity(image_bgr, "hsi")
    i_max_orig = float(i_u8.max()) or 1.0
    i_n = i_u8.astype(np.float64) / 255.0

    l_afgt = afgt(i_n, i_max_orig)
    r_s = unsharp_mask(i_n, l_afgt, sigma=sigma, factor=um_factor)
    r_s_u8 = np.clip(r_s * 255.0, 0, 255).astype(np.uint8)

    return color_restoration(r_s_u8, ctx)


if __name__ == "__main__":
    img = cv2.imread("/home/shakib_islam/Desktop/image_processing /dataset/downloaded/polyp_wikimedia.jpeg")
    img = cv2.resize(img, (512, 512))
    out = enhance_afgt(img)
    cv2.imwrite("/home/shakib_islam/Desktop/image_processing /experiments/paper_b_afgt_output.png", out)
    print("wrote experiments/paper_b_afgt_output.png")
