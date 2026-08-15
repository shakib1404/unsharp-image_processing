"""
Sweep gamma (Eq. 3 correction exponent, paper default 0.8) on the paper's own
Fig. 6 image, full paper pipeline otherwise (HSV + Hu-WBI + CLAHE + SRGAN-v +
colour transfer), and plot how PSNR/SSIM/entropy/EME/brightness respond.
"""
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
import enhance as E

IMG_PATH = "/home/shakib_islam/Desktop/image_processing /uploads/fig6_original.png"
OUT_PNG = "/home/shakib_islam/Desktop/image_processing /experiments/gamma_sweep.png"
WORK_SIZE = 512
PAPER_GAMMA = 0.8

GAMMAS = np.round(np.arange(0.2, 2.01, 0.1), 2)


def main():
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))

    psnr, ssim, entropy, eme, brightness, cii = [], [], [], [], [], []
    for g in GAMMAS:
        out = E.enhance(img, gamma=float(g), color_space="hsv",
                         upsample_method="hu_wbi", contrast_method="clahe",
                         sr_method="srgan_v", use_srgan=True,
                         use_color_transfer=True)
        m = E.compute_metrics(img, out, include_delta_e=False)
        psnr.append(m["psnr"]); ssim.append(m["ssim"])
        entropy.append(m["enhanced"]["entropy"]); eme.append(m["enhanced"]["eme"])
        brightness.append(m["enhanced"]["brightness"]); cii.append(m["gain"]["cii"])
        print(f"gamma={g:.1f}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
              f"entropy={m['enhanced']['entropy']:.3f}  EME={m['enhanced']['eme']:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Gamma sweep — {WORK_SIZE}x{WORK_SIZE}, paper pipeline otherwise "
                 f"(dashed line = paper's gamma={PAPER_GAMMA})", fontsize=12)

    panels = [
        (axes[0, 0], psnr, "PSNR (dB)", "tab:blue"),
        (axes[0, 1], ssim, "SSIM", "tab:green"),
        (axes[0, 2], entropy, "Entropy (bits)", "tab:orange"),
        (axes[1, 0], eme, "EME", "tab:red"),
        (axes[1, 1], brightness, "Brightness", "tab:purple"),
        (axes[1, 2], cii, "CII (contrast gain x)", "tab:brown"),
    ]
    for ax, vals, title, color in panels:
        ax.plot(GAMMAS, vals, marker="o", color=color)
        ax.axvline(PAPER_GAMMA, color="grey", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("gamma")
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
