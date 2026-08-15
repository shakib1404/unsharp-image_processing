"""Aggregate the sweep: win rates, per-image stability, metric agreement."""
import csv, collections, itertools
import numpy as np
from scipy.stats import spearmanr, wilcoxon

CSV = "/tmp/claude-1000/-home-shakib-islam-Desktop-image-processing-/17fd1c0d-b8a1-49e3-bfed-d7247df7446d/scratchpad/results.csv"
rows = list(csv.DictReader(open(CSV)))
for r in rows:
    for k, v in r.items():
        if k not in ("image", "class", "config"):
            r[k] = float(v)

configs = list(dict.fromkeys(r["config"] for r in rows))
images  = list(dict.fromkeys(r["image"] for r in rows))
by = {(r["image"], r["config"]): r for r in rows}

# metric -> direction ("higher"/"lower" better)
METRICS = {
    "psnr": "higher", "ssim": "higher", "mse": "lower", "ambe": "lower",
    "delta_e": "lower", "uqi": "higher",
    "enh_entropy": "higher", "enh_contrast": "higher", "enh_eme": "higher",
    "enh_avg_gradient": "higher", "enh_colorfulness": "higher",
    "enh_edge_density": "higher", "cii": "higher", "sharpness_gain": "higher",
}

print("=" * 100)
print("1. WIN RATE — how often each config ranks #1 across the 20 images")
print("=" * 100)
wins = collections.defaultdict(lambda: collections.Counter())
for m, direction in METRICS.items():
    for img in images:
        vals = [(by[(img, c)][m], c) for c in configs]
        best = (max if direction == "higher" else min)(vals)[1]
        wins[m][best] += 1

hdr = f"{'metric':<20}" + "".join(f"{c.split(':')[-1][:9]:>11}" for c in configs)
print(hdr)
for m in METRICS:
    line = f"{m:<20}"
    for c in configs:
        n = wins[m][c]
        line += f"{(str(n) if n else '.'):>11}"
    print(line)

print()
print("configs that NEVER rank #1 on any metric:",
      [c for c in configs if not any(wins[m][c] for m in METRICS)])
print("configs that win a clean sweep (20/20) on some metric:",
      {m: c for m in METRICS for c in configs if wins[m][c] == len(images)})

print()
print("=" * 100)
print("2. PAPER CONFIG vs EACH ALTERNATIVE — mean over images, and Wilcoxon p")
print("=" * 100)
key_metrics = ["psnr", "ssim", "enh_entropy", "enh_eme", "cii", "sharpness_gain", "delta_e"]
print(f"{'config':<24}" + "".join(f"{m.replace('enh_',''):>16}" for m in key_metrics))
for c in configs:
    line = f"{c:<24}"
    for m in key_metrics:
        vals = np.array([by[(i, c)][m] for i in images])
        line += f"{vals.mean():>10.3f}±{vals.std():<5.2f}"
    print(line)

print()
print("Wilcoxon signed-rank vs PAPER (n=20 paired images); * = p<0.05")
print(f"{'config':<24}" + "".join(f"{m.replace('enh_',''):>18}" for m in key_metrics))
for c in configs:
    if c == "PAPER":
        continue
    line = f"{c:<24}"
    for m in key_metrics:
        a = np.array([by[(i, "PAPER")][m] for i in images])
        b = np.array([by[(i, c)][m] for i in images])
        d = b - a
        if np.allclose(d, 0):
            line += f"{'identical':>18}"
            continue
        p = wilcoxon(a, b).pvalue
        line += f"{('%+.3f' % d.mean()) + ('*' if p < 0.05 else ' '):>18}"
    print(line)

print()
print("=" * 100)
print("3. DO THE METRICS EVEN AGREE? Spearman rank correlation across all runs")
print("=" * 100)
mnames = list(METRICS)
mat = np.array([[r[m] for m in mnames] for r in rows])
rho = spearmanr(mat).statistic
pairs = [(mnames[i], mnames[j], rho[i, j])
         for i, j in itertools.combinations(range(len(mnames)), 2)]
pairs.sort(key=lambda t: t[2])
print("strongest DISAGREEMENTS (negative rank correlation):")
for a, b, r in pairs[:8]:
    print(f"  {a:<20} vs {b:<20} rho = {r:+.3f}")
print("strongest agreements:")
for a, b, r in pairs[-5:]:
    print(f"  {a:<20} vs {b:<20} rho = {r:+.3f}")

print()
print("=" * 100)
print("4. IS THE BEST CONFIG STABLE PER IMAGE? (best config per image, by metric)")
print("=" * 100)
for m in ["psnr", "ssim", "enh_entropy", "enh_eme", "cii"]:
    per_img = []
    for img in images:
        vals = [(by[(img, c)][m], c) for c in configs]
        per_img.append((max if METRICS[m] == "higher" else min)(vals)[1])
    cnt = collections.Counter(per_img)
    print(f"{m:<18} distinct winners: {len(cnt):<3} -> {dict(cnt.most_common())}")

print()
print("per-class best (entropy / EME / CII):")
classes = sorted(set(r["class"] for r in rows))
for cls in classes:
    imgs = [r["image"] for r in rows if r["class"] == cls]
    imgs = list(dict.fromkeys(imgs))
    line = f"  {cls:<24}"
    for m in ["enh_entropy", "enh_eme", "cii"]:
        means = {c: np.mean([by[(i, c)][m] for i in imgs]) for c in configs}
        line += f"{m.replace('enh_','')}={max(means, key=means.get):<22}"
    print(line)
