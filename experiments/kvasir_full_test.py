"""
Full test battery, run on a real Kvasir v2 image (dataset/downloaded/kvasir_dyed-lifted-polyps.jpg,
official Kvasir v2 dataset, class "dyed-lifted-polyps", fetched via the dataset's
Hugging Face mirror since datasets.simula.no's own TLS chain is broken).

Repeats, on this image, everything done earlier on the Wikimedia polyp photo:
  1. Paper A default
  2. Configs A-F (Paper A / Paper B / hybrids)
  3. Per-stage ablation: colour space, upsample, contrast, SR method
  4. Gamma sweep
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
sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing /experiments")
import enhance as E
from paper_b_afgt import afgt, enhance_afgt
from paper_comparison import hybrid_enhance
from search_best_pipeline import run_config

IMG_PATH = "/home/shakib_islam/Desktop/image_processing /dataset/downloaded/kvasir_dyed-lifted-polyps.jpg"
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


def flatten(label, m, extra=None):
    row = {"config": label, "psnr": m["psnr"], "ssim": m["ssim"],
           "entropy": m["enhanced"]["entropy"], "eme": m["enhanced"]["eme"],
           "cii": m["gain"]["cii"], "colorfulness": m["enhanced"]["colorfulness"],
           "delta_e": m["delta_e"]}
    if extra:
        row.update(extra)
    return row


def print_row(label, m, width=42):
    print(f"{label:<{width}}{m['psnr']:>8.2f}{m['ssim']:>8.4f}"
          f"{m['enhanced']['entropy']:>9.3f}{m['enhanced']['eme']:>9.2f}"
          f"{m['gain']['cii']:>7.3f}{m['enhanced']['colorfulness']:>10.2f}{m['delta_e']:>9.3f}")


def table_header(width=42):
    print(f"{'config':<{width}}{'PSNR':>8}{'SSIM':>8}{'Entropy':>9}{'EME':>9}"
          f"{'CII':>7}{'Colour':>10}{'dE2000':>9}")
    print("-" * (width + 68))


def write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  -> {path}")


def main():
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))
    print(f"Kvasir v2 image: {os.path.basename(IMG_PATH)} ({WORK_SIZE}x{WORK_SIZE})\n")

    # ── 0. paper default ─────────────────────────────────────────────────
    print("=" * 110); print("0. PAPER A DEFAULT"); print("=" * 110)
    out0, cfg0, m0 = run(img)
    table_header(); print_row("paper default", m0)
    cv2.imwrite(os.path.join(OUT_DIR, "kvasir_paper_default_comparison.png"),
                E.build_comparison_image(img, out0, m0, cfg0))
    rows0 = [flatten("paper default", m0)]
    write_csv("kvasir_paper_default.csv", rows0)

    # ── 1. configs A-F ───────────────────────────────────────────────────
    print("\n" + "=" * 110); print("1. CONFIGS A-F"); print("=" * 110)
    table_header()
    rows = []
    print_row("A: Paper A default", m0); rows.append(flatten("A: Paper A default", m0))

    out_b = enhance_afgt(img)
    m_b = E.compute_metrics(img, out_b, include_delta_e=True)
    print_row("B: Paper B (AFGT+UM)", m_b); rows.append(flatten("B: Paper B (AFGT+UM)", m_b))

    out_c, _ = hybrid_enhance(img)
    m_c = E.compute_metrics(img, out_c, include_delta_e=True)
    print_row("C: Hybrid (A+AFGT)", m_c); rows.append(flatten("C: Hybrid (A+AFGT)", m_c))

    out_d, _ = hybrid_enhance(img, contrast_method="agcwd")
    m_d = E.compute_metrics(img, out_d, include_delta_e=True)
    print_row("D: Hybrid (AFGT+AGCWD)", m_d); rows.append(flatten("D: Hybrid (AFGT+AGCWD)", m_d))
    cv2.imwrite(os.path.join(OUT_DIR, "kvasir_cmp_D.png"),
                E.build_comparison_image(img, out_d, m_d, {"sr_actual": "srgan_v"}))

    out_e, cfg_e, m_e = run(img, upsample_method="hu_wbi_literal", contrast_method="stretch")
    print_row("E: HSV+stretch+HuWBIliteral", m_e)
    rows.append(flatten("E: HSV+stretch+HuWBIliteral", m_e))

    out_f = run_config(img, "afgt", "stretch", "hu_wbi_literal")
    m_f = E.compute_metrics(img, out_f, include_delta_e=True)
    print_row("F: afgt+stretch+HuWBIliteral", m_f)
    rows.append(flatten("F: afgt+stretch+HuWBIliteral", m_f))
    cv2.imwrite(os.path.join(OUT_DIR, "kvasir_cmp_F.png"),
                E.build_comparison_image(img, out_f, m_f, {"sr_actual": "srgan_v"}))
    write_csv("kvasir_configs_A-F.csv", rows)

    # ── 2. ablations ─────────────────────────────────────────────────────
    for axis_name, table, key in [("COLOUR SPACE", E.COLOR_SPACES, "color_space"),
                                   ("UPSAMPLE METHOD", E.UPSAMPLE_METHODS, "upsample_method"),
                                   ("CONTRAST METHOD", E.CONTRAST_METHODS, "contrast_method")]:
        print("\n" + "=" * 110); print(f"{axis_name}  (everything else = paper default)"); print("=" * 110)
        table_header()
        arows = []
        for val in table:
            _, cfg, m = run(img, **{key: val})
            label = f"{val}{'  [paper]' if val in ('hsv','hu_wbi','clahe') else ''}"
            print_row(label, m)
            arows.append(flatten(label, m))
        write_csv(f"kvasir_ablation_{key}.csv", arows)

    print("\n" + "=" * 110); print("SR METHOD  (everything else = paper default)"); print("=" * 110)
    table_header()
    srows = []
    avail = E.available_sr_methods()
    for sr in E.SR_METHODS:
        if not avail.get(sr, True):
            continue
        _, cfg, m = run(img, sr_method=sr)
        label = f"{sr}{'  [paper]' if sr == 'srgan_v' else ''}"
        print_row(label, m)
        srows.append(flatten(label, m))
    write_csv("kvasir_ablation_sr.csv", srows)

    # ── 3. gamma sweep ───────────────────────────────────────────────────
    print("\n" + "=" * 110); print("GAMMA SWEEP"); print("=" * 110)
    gammas = np.round(np.arange(0.2, 2.01, 0.1), 2)
    psnr, ssim, entropy, eme, brightness, cii = [], [], [], [], [], []
    for g in gammas:
        _, cfg, m = run(img, gamma=float(g))
        psnr.append(m["psnr"]); ssim.append(m["ssim"])
        entropy.append(m["enhanced"]["entropy"]); eme.append(m["enhanced"]["eme"])
        brightness.append(m["enhanced"]["brightness"]); cii.append(m["gain"]["cii"])
        print(f"gamma={g:.1f}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Gamma sweep — Kvasir v2 image, paper flow otherwise (dashed = paper's gamma=0.8)")
    panels = [(axes[0, 0], psnr, "PSNR (dB)", "tab:blue"), (axes[0, 1], ssim, "SSIM", "tab:green"),
              (axes[0, 2], entropy, "Entropy (bits)", "tab:orange"), (axes[1, 0], eme, "EME", "tab:red"),
              (axes[1, 1], brightness, "Brightness", "tab:purple"), (axes[1, 2], cii, "CII (x)", "tab:brown")]
    for ax, vals, title, color in panels:
        ax.plot(gammas, vals, marker="o", color=color)
        ax.axvline(0.8, color="grey", linestyle="--", linewidth=1)
        ax.set_title(title); ax.set_xlabel("gamma"); ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, "kvasir_gamma_sweep.png"), dpi=130)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'kvasir_gamma_sweep.png')}")


if __name__ == "__main__":
    main()
