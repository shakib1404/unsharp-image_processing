"""
One real, freshly-downloaded endoscopy image (dataset/downloaded/polyp_wikimedia.jpeg
-- CC-BY-2.5, Stephen Holland M.D., via Wikimedia Commons -- not one of the
project's pre-existing sample/paper-figure images).

1. Run the paper's normal flow (HSV + Hu-WBI + CLAHE + SRGAN-v + colour
   transfer) and report its metrics.
2. Four one-axis-at-a-time tables, full process otherwise identical to the
   paper flow: colour space, upsample method, contrast method, SR method.
3. A gamma parameter sweep graph (paper flow, gamma varied).
"""
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


def print_row(label, config, m, width=42):
    print(f"{label:<{width}}{m['psnr']:>8.2f}{m['ssim']:>8.4f}"
          f"{m['enhanced']['entropy']:>9.3f}{m['enhanced']['eme']:>9.2f}"
          f"{m['gain']['cii']:>7.3f}{m['enhanced']['colorfulness']:>10.2f}"
          f"{m['delta_e']:>9.3f}   {config.get('sr_actual', '')}")


def table_header(width=42):
    print(f"{'config':<{width}}{'PSNR':>8}{'SSIM':>8}{'Entropy':>9}{'EME':>9}"
          f"{'CII':>7}{'Colour':>10}{'dE2000':>9}")
    print("-" * (width + 68))


def write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def flatten(label, config, m):
    return {"config": label, "psnr": m["psnr"], "ssim": m["ssim"],
            "entropy": m["enhanced"]["entropy"], "eme": m["enhanced"]["eme"],
            "cii": m["gain"]["cii"], "colorfulness": m["enhanced"]["colorfulness"],
            "delta_e": m["delta_e"], "sr_actual": config.get("sr_actual")}


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise SystemExit(f"could not read {IMG_PATH}")
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))
    print(f"Real image: {os.path.basename(IMG_PATH)} (CC-BY-2.5, Stephen Holland "
          f"M.D. / Wikimedia Commons) resized to {WORK_SIZE}x{WORK_SIZE}\n")

    # ── 0. paper's normal flow ───────────────────────────────────────────
    print("=" * 110)
    print("0. PAPER'S NORMAL FLOW  (HSV + Hu-WBI + CLAHE + SRGAN-v + colour transfer)")
    print("=" * 110)
    out0, cfg0, m0 = run(img)
    table_header()
    print_row("paper default", cfg0, m0)
    write_csv("real_paper_default.csv", [flatten("paper default", cfg0, m0)])
    cv2.imwrite(os.path.join(OUT_DIR, "real_paper_default_comparison.png"),
                E.build_comparison_image(img, out0, m0, cfg0))

    # ── 1. colour space ──────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("1. COLOUR SPACE  (everything else = paper default)")
    print("=" * 110)
    table_header()
    rows = []
    for cs in E.COLOR_SPACES:
        out, cfg, m = run(img, color_space=cs)
        label = f"{cs}{'  [paper]' if cs == 'hsv' else ''}"
        print_row(label, cfg, m)
        rows.append(flatten(label, cfg, m))
    write_csv("real_ablation_color_space.csv", rows)

    # ── 2. upsample method ───────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("2. UPSAMPLE METHOD  (everything else = paper default)")
    print("=" * 110)
    table_header()
    rows = []
    for um in E.UPSAMPLE_METHODS:
        out, cfg, m = run(img, upsample_method=um)
        label = f"{um}{'  [paper]' if um == 'hu_wbi' else ''}"
        print_row(label, cfg, m)
        rows.append(flatten(label, cfg, m))
    write_csv("real_ablation_upsample.csv", rows)

    # ── 3. contrast method ───────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("3. CONTRAST METHOD  (everything else = paper default)")
    print("=" * 110)
    table_header()
    rows = []
    for cm in E.CONTRAST_METHODS:
        out, cfg, m = run(img, contrast_method=cm)
        label = f"{cm}{'  [paper]' if cm == 'clahe' else ''}"
        print_row(label, cfg, m)
        rows.append(flatten(label, cfg, m))
    write_csv("real_ablation_contrast.csv", rows)

    # ── 4. SR / SRGAN method ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("4. SR METHOD  (everything else = paper default)")
    print("=" * 110)
    table_header()
    rows = []
    avail = E.available_sr_methods()
    for sr in E.SR_METHODS:
        if not avail.get(sr, True):
            continue
        out, cfg, m = run(img, sr_method=sr)
        label = f"{sr}{'  [paper]' if sr == 'srgan_v' else ''}"
        print_row(label, cfg, m)
        rows.append(flatten(label, cfg, m))
    write_csv("real_ablation_sr.csv", rows)

    # ── 5. gamma parameter sweep -> graph ────────────────────────────────
    print("\n" + "=" * 110)
    print("5. GAMMA PARAMETER SWEEP  (paper flow, gamma varied) -> graph")
    print("=" * 110)
    gammas = np.round(np.arange(0.2, 2.01, 0.1), 2)
    psnr, ssim, entropy, eme, brightness, cii = [], [], [], [], [], []
    for g in gammas:
        out, cfg, m = run(img, gamma=float(g))
        psnr.append(m["psnr"]); ssim.append(m["ssim"])
        entropy.append(m["enhanced"]["entropy"]); eme.append(m["enhanced"]["eme"])
        brightness.append(m["enhanced"]["brightness"]); cii.append(m["gain"]["cii"])
        print(f"gamma={g:.1f}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
              f"entropy={m['enhanced']['entropy']:.3f}  EME={m['enhanced']['eme']:.2f}")

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
