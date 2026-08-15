"""
Empirical check: over the whole dataset, is there a configuration that always
wins? Runs one-axis-at-a-time variations around the paper's own configuration
on every dataset image and dumps a tidy CSV for analysis.
"""
import csv, glob, os, sys, time
import cv2
sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
import enhance as E

PAPER = dict(color_space="hsv", upsample_method="hu_wbi",
             contrast_method="clahe", sr_method="srgan_v")

CONFIGS = [("PAPER", dict(PAPER))]
for cs in ["yiq", "ycrcb", "lab", "hsi"]:
    CONFIGS.append((f"space:{cs}", {**PAPER, "color_space": cs}))
for cm in ["clahe_sk", "he", "bbhe", "dsihe", "agcwd", "msr", "stretch"]:
    CONFIGS.append((f"contrast:{cm}", {**PAPER, "contrast_method": cm}))
for um in ["hu_wbi_literal", "bilinear", "bicubic", "lanczos4", "nearest", "none"]:
    CONFIGS.append((f"upsample:{um}", {**PAPER, "upsample_method": um}))
for sr in ["none", "lanczos_detail", "ibp", "unsharp_sr"]:
    CONFIGS.append((f"sr:{sr}", {**PAPER, "sr_method": sr}))

images = sorted(glob.glob("/home/shakib_islam/Desktop/image_processing /dataset/*/*.png"))
out = "/tmp/claude-1000/-home-shakib-islam-Desktop-image-processing-/17fd1c0d-b8a1-49e3-bfed-d7247df7446d/scratchpad/results.csv"

rows = []
t0 = time.time()
for ip, path in enumerate(images):
    img = cv2.resize(cv2.imread(path), (256, 256))
    for name, cfg in CONFIGS:
        out_img = E.enhance(img, use_srgan=(cfg["sr_method"] != "none"), **cfg)
        m = E.compute_metrics(img, out_img)
        flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
        flat.update({f"enh_{k}": v for k, v in m["enhanced"].items()})
        flat.update({f"orig_{k}": v for k, v in m["original"].items()})
        flat.update(m["gain"])
        rows.append({"image": os.path.basename(path),
                     "class": os.path.basename(os.path.dirname(path)),
                     "config": name, **flat})
    print(f"[{ip+1}/{len(images)}] {os.path.basename(path)}  {time.time()-t0:.0f}s", flush=True)

with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("done", len(rows), "rows in", round(time.time() - t0), "s")
