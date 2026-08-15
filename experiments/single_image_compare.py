"""
One image (the paper's own Fig. 6 original), CLAHE as the fixed base, and a
few things toggled on top of it (SR on/off, colour transfer on/off, visual
punch on/off). Full metrics per combo, printed against the paper's own
claimed numbers (PSNR 26.4 dB, SSIM 0.923, R-B redness 93.8 on this exact
image — see enhance.py's module docstring and apply_visual_punch() comment).
"""
import csv
import os
import sys

import cv2

sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
import enhance as E

IMG_PATH = "/home/shakib_islam/Desktop/image_processing /uploads/fig6_original.png"
OUT_DIR = "/home/shakib_islam/Desktop/image_processing /experiments"
WORK_SIZE = 512  # paper's own working resolution

PAPER_CLAIMS = {"psnr": 26.4, "ssim": 0.923, "r_minus_b": 93.8}

BASE = dict(color_space="hsv", upsample_method="hu_wbi", contrast_method="clahe")

CONFIGS = [
    ("CLAHE only (no SR, no colour xfer)",
     dict(BASE, sr_method="none", use_color_transfer=False)),
    ("CLAHE + colour transfer",
     dict(BASE, sr_method="none", use_color_transfer=True)),
    ("CLAHE + SRGAN-v",
     dict(BASE, sr_method="srgan_v", use_color_transfer=False)),
    ("CLAHE + SRGAN-v + colour transfer  [paper defaults]",
     dict(BASE, sr_method="srgan_v", use_color_transfer=True)),
    ("CLAHE + SRGAN-v + colour transfer + visual punch",
     dict(BASE, sr_method="srgan_v", use_color_transfer=True,
          use_visual_punch=True)),
    ("CLAHE + Lanczos-detail SR (classical, no GAN) + colour transfer",
     dict(BASE, sr_method="lanczos_detail", use_color_transfer=True)),
]


def r_minus_b(bgr):
    """Paper's own redness proxy: mean(R) - mean(B)."""
    b, g, r = cv2.split(bgr.astype("float32"))
    return float(r.mean() - b.mean())


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise SystemExit(f"could not read {IMG_PATH}")
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))
    orig_rb = r_minus_b(img)

    rows = []
    print(f"Image: {os.path.basename(IMG_PATH)}  ({WORK_SIZE}x{WORK_SIZE})")
    print(f"Original R-B redness = {orig_rb:.2f}")
    print(f"Paper claims (whole-dataset avg unless noted): "
          f"PSNR={PAPER_CLAIMS['psnr']} dB, SSIM={PAPER_CLAIMS['ssim']}, "
          f"R-B redness (this exact Fig.6 image, 'Proposed')={PAPER_CLAIMS['r_minus_b']}")
    print("=" * 118)
    hdr = f"{'config':<58}{'PSNR':>8}{'SSIM':>8}{'Entropy':>9}{'EME':>9}{'CII':>7}{'R-B':>9}{'dE2000':>9}"
    print(hdr)
    print("-" * 118)

    for name, cfg in CONFIGS:
        use_srgan = cfg.get("sr_method", "none") != "none"
        out, config = E.enhance(img, use_srgan=use_srgan, return_config=True, **cfg)
        m = E.compute_metrics(img, out, include_delta_e=True)
        rb = r_minus_b(out)
        row = {
            "config": name,
            "psnr": m["psnr"], "ssim": m["ssim"],
            "entropy": m["enhanced"]["entropy"], "eme": m["enhanced"]["eme"],
            "cii": m["gain"]["cii"], "r_minus_b": round(rb, 4),
            "delta_e": m["delta_e"], "sr_actual": config["sr_actual"],
        }
        rows.append(row)
        print(f"{name:<58}{m['psnr']:>8.2f}{m['ssim']:>8.4f}"
              f"{m['enhanced']['entropy']:>9.3f}{m['enhanced']['eme']:>9.2f}"
              f"{m['gain']['cii']:>7.3f}{rb:>9.2f}{m['delta_e']:>9.3f}")

        cmp_img = E.build_comparison_image(img, out, m, config)
        safe = name.split("[")[0].strip().replace(" ", "_").replace("/", "-")
        cv2.imwrite(os.path.join(OUT_DIR, f"fig6_{safe}.png"), cmp_img)

    print("-" * 118)
    print(f"{'PAPER CLAIM':<58}{PAPER_CLAIMS['psnr']:>8.2f}{PAPER_CLAIMS['ssim']:>8.4f}"
          f"{'':>9}{'':>9}{'':>7}{PAPER_CLAIMS['r_minus_b']:>9.2f}{'':>9}")

    csv_path = os.path.join(OUT_DIR, "fig6_compare.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path}")
    print("Comparison PNGs written to experiments/fig6_*.png")


if __name__ == "__main__":
    main()
