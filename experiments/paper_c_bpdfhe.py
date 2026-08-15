"""
Paper C — Brightness Preserving Dynamic Fuzzy Histogram Equalization (BPDFHE).

D. Sheet, H. Garud, A. Suveer, M. Mahadevappa, J. Chatterjee, "Brightness
Preserving Dynamic Fuzzy Histogram Equalization," IEEE Trans. Consumer
Electronics, 56(4), 2475-2480, 2010.

BPDFHE's per-partition equalization (Eq. 6-11 below) is explicitly the
Dynamic Histogram Equalization (DHE) technique, reused from an earlier paper:
M. Abdullah-Al-Wadud, M. H. Kabir, M. A. A. Dewan, O. Chae, "A Dynamic
Histogram Equalization for Image Contrast Enhancement," IEEE Trans. Consumer
Electronics, 53(2), 593-600, 2007. BPDFHE's own contribution is computing
that histogram in the *fuzzy* domain (Eq. 1-2) so the local-maxima detection
that drives the partitioning (Eq. 3-5) doesn't fragment on histogram noise,
plus a final brightness-restoring normalization (Eq. 12).

Implemented as a standalone contrast-method function, same uint8-in/uint8-out
shape as enhance.py's apply_contrast() so it can drop into any pipeline slot
that currently takes "clahe"/"agcwd"/etc.
"""
import numpy as np


def _fuzzy_histogram(channel: np.ndarray, L: int = 256, radius: int = 4) -> np.ndarray:
    """Eq. 1-2: triangular-membership fuzzy histogram, support width 2*radius-1."""
    crisp = np.bincount(channel.ravel(), minlength=L).astype(np.float64)
    k = np.arange(-(radius - 1), radius)
    w = np.clip(1.0 - np.abs(k) / radius, 0.0, None)
    return np.convolve(crisp, w, mode="same")


def _local_maxima(h: np.ndarray) -> list:
    """Eq. 3-5, simplified: strict local maxima of the (already-smooth) fuzzy histogram."""
    L = len(h)
    peaks = [i for i in range(1, L - 1) if h[i] > h[i - 1] and h[i] > h[i + 1]]
    return peaks or [int(np.argmax(h))]


def bpdfhe(channel: np.ndarray, L: int = 256) -> np.ndarray:
    flat = channel.ravel()
    h = _fuzzy_histogram(channel, L=L)
    maxima = sorted(_local_maxima(h))

    i_min, i_max = int(flat.min()), int(flat.max())
    if i_max <= i_min:
        return channel.copy()

    # Partitions: [Imin,m0],[m0+1,m1],...,[mn+1,Imax]  (Section II.B.2)
    bounds = [i_min] + [m for m in maxima if i_min < m < i_max] + [i_max]
    bounds = sorted(set(bounds))
    partitions = []
    lo = i_min
    for m in bounds[1:-1]:
        partitions.append((lo, m))
        lo = m + 1
    partitions.append((lo, i_max))
    partitions = [(a, b) for a, b in partitions if b >= a]
    if not partitions:
        partitions = [(i_min, i_max)]

    # C.1 Mapping partitions to a dynamic range (Eq. 6-10)
    spans, factors, pops = [], [], []
    for lo, hi in partitions:
        pop = float(h[lo:hi + 1].sum())
        pops.append(pop)
        span = hi - lo
        factor = span * np.log10(max(pop, 1.0))
        spans.append(span)
        factors.append(max(factor, 0.0))
    total_factor = sum(factors)
    if total_factor <= 0:
        ranges = [(L - 1) / len(partitions)] * len(partitions)
    else:
        ranges = [(L - 1) * f / total_factor for f in factors]

    n = len(partitions)
    starts, stops = [0.0] * n, [0.0] * n
    cum = 0.0
    for i, r in enumerate(ranges):
        starts[i] = 0.0 if i == 0 else stops[i - 1] + 1.0
        stops[i] = (L - 1) if i == n - 1 else starts[i] + r
        cum += r

    # C.2 Equalizing each sub-histogram (Eq. 11) -> build a full 0..255 LUT
    lut = np.zeros(L, dtype=np.float64)
    for (lo, hi), start_i, stop_i, pop in zip(partitions, starts, stops, pops):
        range_i = stop_i - start_i
        if pop <= 0:
            lut[lo:hi + 1] = start_i
            continue
        cdf = np.cumsum(h[lo:hi + 1]) / pop
        lut[lo:hi + 1] = start_i + range_i * cdf

    dhe_out = np.clip(lut[flat], 0, L - 1).reshape(channel.shape)

    # D. Normalization of image brightness (Eq. 12)
    m_i = float(channel.mean())
    m_o = float(dhe_out.mean()) or 1e-6
    g = np.clip(dhe_out * (m_i / m_o), 0, L - 1)
    return g.astype(np.uint8)


if __name__ == "__main__":
    import sys
    import cv2
    sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
    import enhance as E

    img = cv2.imread("/home/shakib_islam/Desktop/image_processing /dataset/downloaded/polyp_wikimedia.jpeg")
    img = cv2.resize(img, (512, 512))
    v_u8, ctx = E.split_intensity(img, "hsv")
    out_v = bpdfhe(v_u8)
    out = E.merge_intensity(out_v, ctx)
    cv2.imwrite("/home/shakib_islam/Desktop/image_processing /experiments/paper_c_bpdfhe_output.png", out)
    m = E.compute_metrics(img, out, include_delta_e=True)
    print(f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  entropy={m['enhanced']['entropy']:.3f}  "
          f"EME={m['enhanced']['eme']:.2f}  CII={m['gain']['cii']:.3f}  dE2000={m['delta_e']:.3f}")
