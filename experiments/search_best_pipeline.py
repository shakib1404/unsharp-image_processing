"""
Broader search: {tone-curve} x {contrast method} x {upsample method}, SRGAN-v
and colour transfer fixed on (matching every earlier config), real downloaded
image. Finds the best combination found so far across A/B/C/D/E and this
session's new BPDFHE paper.

tone-curve options:
  gamma  -- Paper A: invert -> gamma(0.8)
  afgt   -- Paper B / Long et al.: Adaptive Fraction Gamma Transformation

contrast options: clahe (paper), agcwd, stretch, bpdfhe (new), he, bbhe, dsihe
upsample options: hu_wbi (paper), hu_wbi_literal
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing /experiments")
import enhance as E
from paper_b_afgt import afgt
from paper_c_bpdfhe import bpdfhe

IMG_PATH = "/home/shakib_islam/Desktop/image_processing /dataset/downloaded/polyp_wikimedia.jpeg"
OUT_DIR = "/home/shakib_islam/Desktop/image_processing /experiments"
WORK_SIZE = 512


def tone_curve(v_u8, kind, gamma=0.8):
    if kind == "gamma":
        v_norm = v_u8.astype(np.float32) / 255.0
        v_inv = 1.0 - v_norm
        v_g = np.power(np.clip(v_inv, 0.0, 1.0), gamma)
        return np.clip(v_g * 255, 0, 255).astype(np.uint8), True  # needs_invert_back
    if kind == "afgt":
        i_n = v_u8.astype(np.float64) / 255.0
        l_afgt = afgt(i_n, float(v_u8.max()) or 1.0)
        return np.clip(l_afgt * 255.0, 0, 255).astype(np.uint8), False
    raise ValueError(kind)


def contrast_step(v_u8, kind):
    if kind == "bpdfhe":
        return bpdfhe(v_u8)
    return E.apply_contrast(v_u8, kind)


def run_config(img, tone, contrast, upsample, cn=0.85, sigma=1.0):
    v_u8, ctx = E.split_intensity(img, "hsv")
    v_tone, needs_invert = tone_curve(v_u8, tone)

    v_c_pre = contrast_step(v_tone, contrast)
    v_c = E.upsample_2x(v_c_pre, upsample)
    h, w = v_tone.shape
    v_down = cv2.resize(v_c, (w, h), interpolation=cv2.INTER_LINEAR) \
        if v_c.shape != v_tone.shape else v_c

    lpf = cv2.GaussianBlur(v_down.astype(np.float32), (0, 0), sigma)
    ps = (v_down.astype(np.float32) - lpf) * cn
    v_sharp = np.clip(v_down.astype(np.float32) + ps, 0, 255).astype(np.uint8)

    sr_method = E.resolve_sr_method("srgan_v")
    v_sr = E.apply_sr_intensity(v_sharp, sr_method)
    v_final = (255 - v_sr) if needs_invert else v_sr

    enhanced = E.merge_intensity(v_final, ctx)
    enhanced = E.color_transfer(img, enhanced, strength=0.6)
    return enhanced


def main():
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))

    tones = ["gamma", "afgt"]
    contrasts = ["clahe", "agcwd", "stretch", "bpdfhe", "he", "bbhe", "dsihe"]
    upsamples = ["hu_wbi", "hu_wbi_literal"]

    rows = []
    print(f"{'tone':<8}{'contrast':<10}{'upsample':<16}{'PSNR':>8}{'SSIM':>8}"
          f"{'Entropy':>9}{'EME':>9}{'CII':>7}{'Colour':>10}{'dE2000':>9}")
    print("-" * 95)
    for tone in tones:
        for contrast in contrasts:
            for up in upsamples:
                out = run_config(img, tone, contrast, up)
                m = E.compute_metrics(img, out, include_delta_e=True)
                label = f"{tone}+{contrast}+{up}"
                print(f"{tone:<8}{contrast:<10}{up:<16}{m['psnr']:>8.2f}{m['ssim']:>8.4f}"
                      f"{m['enhanced']['entropy']:>9.3f}{m['enhanced']['eme']:>9.2f}"
                      f"{m['gain']['cii']:>7.3f}{m['enhanced']['colorfulness']:>10.2f}"
                      f"{m['delta_e']:>9.3f}")
                rows.append({"tone": tone, "contrast": contrast, "upsample": up,
                             "psnr": m["psnr"], "ssim": m["ssim"],
                             "entropy": m["enhanced"]["entropy"], "eme": m["enhanced"]["eme"],
                             "cii": m["gain"]["cii"], "colorfulness": m["enhanced"]["colorfulness"],
                             "delta_e": m["delta_e"]})

    with open(os.path.join(OUT_DIR, "search_best_pipeline.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # rank by PSNR+SSIM*20 (put both fidelity metrics on comparable footing) as a simple composite
    ranked = sorted(rows, key=lambda r: r["psnr"] + r["ssim"] * 20, reverse=True)
    print("\nTop 5 by PSNR + 20*SSIM:")
    for r in ranked[:5]:
        print(f"  {r['tone']}+{r['contrast']}+{r['upsample']:<16} "
              f"PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} EME={r['eme']:.2f} dE={r['delta_e']:.3f}")


if __name__ == "__main__":
    main()
