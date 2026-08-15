"""
Three-way comparison on the real downloaded image (dataset/downloaded/polyp_wikimedia.jpeg):

  A. Paper A default   -- Jagarajan & Jayaraman (2026): HSV, invert+gamma,
                           Hu-WBI upsample, CLAHE, SRGAN-v, unsharp mask,
                           colour transfer.
  B. Paper B (AFGT+UM)  -- Ezatian et al. / Long et al.: HSI, Adaptive
                           Fraction Gamma Transformation (no invert needed --
                           AFGT brightens dark regions directly), its own
                           unsharp mask, ratio-based colour restoration.
  C. Hybrid             -- Paper A's structure (HSV, Hu-WBI, CLAHE, SRGAN-v,
                           colour transfer) with Paper A's invert+gamma
                           tone-curve step swapped out for Paper B's AFGT
                           nonlinearity (Eq. 2-3 only -- Paper A's own Step 8
                           already does the unsharp mask, so AFGT's Eq. 11 is
                           skipped here to avoid sharpening twice).
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
import enhance as E
from paper_b_afgt import afgt, enhance_afgt

IMG_PATH = "/home/shakib_islam/Desktop/image_processing /dataset/downloaded/polyp_wikimedia.jpeg"
OUT_DIR = "/home/shakib_islam/Desktop/image_processing /experiments"
WORK_SIZE = 512


def hybrid_enhance(image_bgr: np.ndarray, gamma_unused=None,
                    cn: float = 0.85, sigma: float = 1.0,
                    contrast_method: str = "clahe"):
    """Paper A's pipeline shape, Paper B's AFGT swapped in for invert+gamma."""
    v_u8, ctx = E.split_intensity(image_bgr, "hsv")
    i_n = v_u8.astype(np.float64) / 255.0
    l_afgt = afgt(i_n, float(v_u8.max()) or 1.0)          # Paper B Eq. 2-3
    v_uint8 = np.clip(l_afgt * 255.0, 0, 255).astype(np.uint8)

    v_contrast_pre = E.apply_contrast(v_uint8, contrast_method)   # Paper A Step 5 (CLAHE first)
    v_contrast = E.upsample_2x(v_contrast_pre, "hu_wbi")           # Paper A Step 6 (then Hu-WBI)
    h, w = v_uint8.shape
    v_down = cv2.resize(v_contrast, (w, h), interpolation=cv2.INTER_LINEAR) \
        if v_contrast.shape != v_uint8.shape else v_contrast  # Step 7

    lpf = cv2.GaussianBlur(v_down.astype(np.float32), (0, 0), sigma)   # Step 8
    ps = (v_down.astype(np.float32) - lpf) * cn
    v_sharp = np.clip(v_down.astype(np.float32) + ps, 0, 255).astype(np.uint8)

    sr_method = E.resolve_sr_method("srgan_v")
    v_final = E.apply_sr_intensity(v_sharp, sr_method)      # Step 9 (no invert -- AFGT never inverted)

    enhanced_bgr = E.merge_intensity(v_final, ctx)
    enhanced_bgr = E.color_transfer(image_bgr, enhanced_bgr, strength=0.6)  # Eq. 6
    return enhanced_bgr, {"sr_actual": sr_method}


def report(label, img, out, config, rows):
    m = E.compute_metrics(img, out, include_delta_e=True)
    print(f"{label:<28}{m['psnr']:>8.2f}{m['ssim']:>8.4f}"
          f"{m['enhanced']['entropy']:>9.3f}{m['enhanced']['eme']:>9.2f}"
          f"{m['gain']['cii']:>7.3f}{m['enhanced']['colorfulness']:>10.2f}"
          f"{m['delta_e']:>9.3f}")
    rows.append({"config": label, "psnr": m["psnr"], "ssim": m["ssim"],
                 "entropy": m["enhanced"]["entropy"], "eme": m["enhanced"]["eme"],
                 "cii": m["gain"]["cii"], "colorfulness": m["enhanced"]["colorfulness"],
                 "delta_e": m["delta_e"]})
    cmp_img = E.build_comparison_image(img, out, m, config if isinstance(config, dict) else None)
    safe = label.replace(" ", "_").replace("(", "").replace(")", "")
    cv2.imwrite(os.path.join(OUT_DIR, f"cmp_{safe}.png"), cmp_img)
    return m


def main():
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))

    print(f"{'config':<28}{'PSNR':>8}{'SSIM':>8}{'Entropy':>9}{'EME':>9}"
          f"{'CII':>7}{'Colour':>10}{'dE2000':>9}")
    print("-" * 96)

    rows = []
    out_a, cfg_a = E.enhance(img, return_config=True, use_srgan=True,
                             color_space="hsv", upsample_method="hu_wbi",
                             contrast_method="clahe", sr_method="srgan_v",
                             use_color_transfer=True)
    report("A: Paper A (Jagarajan)", img, out_a, cfg_a, rows)

    out_b = enhance_afgt(img)
    report("B: Paper B (AFGT+UM)", img, out_b, None, rows)

    out_c, cfg_c = hybrid_enhance(img)
    report("C: Hybrid (A + AFGT)", img, out_c, cfg_c, rows)

    out_d, cfg_d = hybrid_enhance(img, contrast_method="agcwd")
    report("D: Hybrid (AFGT+AGCWD)", img, out_d, cfg_d, rows)

    csv_path = os.path.join(OUT_DIR, "paper_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path}")
    print("Comparison PNGs: experiments/cmp_*.png")


if __name__ == "__main__":
    main()
