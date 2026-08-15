"""Remaining two pieces for the real downloaded image: SR-method table + gamma sweep."""
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
import enhance as E

IMG_PATH = "/home/shakib_islam/Desktop/image_processing /dataset/downloaded/polyp_wikimedia.jpeg"
OUT_DIR = "/home/shakib_islam/Desktop/image_processing /experiments"
WORK_SIZE = 512
PAPER = dict(color_space="hsv", upsample_method="hu_wbi", contrast_method="clahe",
             sr_method="srgan_v", use_color_transfer=True)


def run(img, **overrides):
    cfg = {**PAPER, **overrides}
    use_srgan = cfg.get("sr_method", "none") != "none"
    out, config = E.enhance(img, use_srgan=use_srgan, return_config=True, **cfg)
    m = E.compute_metrics(img, out, include_delta_e=True)
    return out, config, m


def flatten(label, config, m):
    return {"config": label, "psnr": m["psnr"], "ssim": m["ssim"],
            "entropy": m["enhanced"]["entropy"], "eme": m["enhanced"]["eme"],
            "cii": m["gain"]["cii"], "colorfulness": m["enhanced"]["colorfulness"],
            "delta_e": m["delta_e"], "sr_actual": config.get("sr_actual")}


def main():
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))

    print("SR METHOD  (everything else = paper default)")
    print(f"{'config':<28}{'PSNR':>8}{'SSIM':>8}{'Entropy':>9}{'EME':>9}{'CII':>7}{'Colour':>10}{'dE2000':>9}")
    rows = []
    avail = E.available_sr_methods()
    for sr in E.SR_METHODS:
        if not avail.get(sr, True):
            continue
        out, cfg, m = run(img, sr_method=sr)
        label = f"{sr}{'  [paper]' if sr == 'srgan_v' else ''}"
        print(f"{label:<28}{m['psnr']:>8.2f}{m['ssim']:>8.4f}{m['enhanced']['entropy']:>9.3f}"
              f"{m['enhanced']['eme']:>9.2f}{m['gain']['cii']:>7.3f}"
              f"{m['enhanced']['colorfulness']:>10.2f}{m['delta_e']:>9.3f}")
        rows.append(flatten(label, cfg, m))
    with open(os.path.join(OUT_DIR, "real_ablation_sr.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  -> experiments/real_ablation_sr.csv")

    print("\nGAMMA SWEEP")
    gammas = np.round(np.arange(0.2, 2.01, 0.1), 2)
    psnr, ssim, entropy, eme, brightness, cii = [], [], [], [], [], []
    for g in gammas:
        out, cfg, m = run(img, gamma=float(g))
        psnr.append(m["psnr"]); ssim.append(m["ssim"])
        entropy.append(m["enhanced"]["entropy"]); eme.append(m["enhanced"]["eme"])
        brightness.append(m["enhanced"]["brightness"]); cii.append(m["gain"]["cii"])
        print(f"gamma={g:.1f}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Gamma sweep on real downloaded image — paper flow otherwise "
                 "(dashed = paper's gamma=0.8)", fontsize=12)
    panels = [(axes[0, 0], psnr, "PSNR (dB)", "tab:blue"),
              (axes[0, 1], ssim, "SSIM", "tab:green"),
              (axes[0, 2], entropy, "Entropy (bits)", "tab:orange"),
              (axes[1, 0], eme, "EME", "tab:red"),
              (axes[1, 1], brightness, "Brightness", "tab:purple"),
              (axes[1, 2], cii, "CII (contrast gain x)", "tab:brown")]
    for ax, vals, title, color in panels:
        ax.plot(gammas, vals, marker="o", color=color)
        ax.axvline(0.8, color="grey", linestyle="--", linewidth=1)
        ax.set_title(title); ax.set_xlabel("gamma"); ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(OUT_DIR, "real_gamma_sweep.png")
    plt.savefig(out_png, dpi=130)
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
