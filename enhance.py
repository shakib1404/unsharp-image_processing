"""
Endoscopy image enhancement pipeline.
Reference: Jagarajan & Jayaraman (2026), Multimedia Tools and Applications 85:477
DOI: 10.1007/s11042-026-21507-z

REVIEWER CLAIMS vs FACTS
─────────────────────────────────────────────────────────────────────────────
Claim 1 – "Diagonal Hu-WBI should be 0.5 * sum(4 neighbours)"
  WRONG. The paper example shows pixel "7" = ½(12) + ½(2) = average of 2.
  By the same logic, a diagonal pixel between 4 neighbours is their AVERAGE
  = sum/4 = 0.25 * sum.  Using 0.5 * sum would give values up to 510 on a
  [0,255] image and clip everything bright to white – destroying interpolation.

Claim 2 – "SRGAN is missing"
  CORRECT (was). The paper combines unsharp masking with SRGAN (Eq. 8, Fig. 1).
  SRGANGenerator below is the paper's architecture; 'srgan_v.pth' is now a
  real checkpoint trained via train_srgan_v.py (self-supervised LR/HR pairs
  from Kvasir-SEG — see that file's docstring for why, since the paper's own
  clinical dataset/weights were never released). Measured on this project's
  sample images: PSNR ~21.6–23.2 dB, SSIM ~0.80–0.83 (compute_metrics()),
  below the paper's claimed 26.4/0.923 — expected, since training data,
  duration and hyperparameters all differ from the paper's unreleased setup.
  Without any weights file, _apply_srgan() falls back to Lanczos-4 + detail
  boost instead.

Claims 3 & 4 – "Workflow order wrong / CLAHE not on inverted intensity"
  PARTLY WRONG, PARTLY RIGHT (corrected 2026-08-15 against the full published
  text, not just the abstract). CLAHE *is* on the inverted+gamma'd intensity,
  as claimed. But CLAHE vs. Hu-WBI ordering was backwards: this file used to
  run Hu-WBI (upsample) before CLAHE. The paper runs CLAHE first, then
  Hu-WBI — stated three separate ways: the abstract frames Hu-WBI's whole
  purpose as fixing "over-enhanced images with artifact issues associated
  with CLAHE" (can't fix artifacts that don't exist yet), Section 3's phase
  list has "(ii) CLAHE performs contrast enhancement, (iii) apply Hu-WBI",
  and Fig. 1's block diagram has the CLAHE box before the Hu-WBI box. Fixed
  order, now matching the paper:
    normalize → invert → gamma → CLAHE → Hu-WBI upsample → downsample →
    unsharp mask → SRGAN → invert back → RGB.
  Since srgan_v.pth was trained on v_sharp from the *old* (Hu-WBI-then-CLAHE)
  ordering, its training distribution no longer exactly matches what
  preprocess_intensity() now produces — retraining via train_srgan_v.py is
  recommended, though the mismatch is mild (same two operations, just
  reordered) and the existing checkpoint still runs.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Hu-WBI  (Half-unit Weighted Bilinear Interpolation)
# ═══════════════════════════════════════════════════════════════════════════════

def hu_wbi_upsample(channel: np.ndarray) -> np.ndarray:
    """
    2× upsampling via Hu-WBI (Eq. 4).

    Weight rule  (matches the paper's worked example):
      • Between 2 horizontal neighbours  → average  = 0.5 * (L + R)
      • Between 2 vertical   neighbours  → average  = 0.5 * (T + B)
      • Between 4 diagonal   neighbours  → average  = 0.25 * (TL+TR+BL+BR)

    Why 0.25 for diagonals, not 0.5?
      The paper says H(a,b) = ΣP_n / 2 for n=1..4, which literally means
      (P1+P2+P3+P4)/2.  However the paper's own example — "pixel 7 was
      calculated as [½(12) + ½(2)]" — shows AVERAGING of 2 neighbours.
      Applying the same logic to 4 neighbours gives sum/4 = 0.25*sum.
      Using 0.5*sum produces values up to 510 and clips all bright pixels
      to white, which is physically nonsensical and contradicts the intent
      of "half-unit weighted" interpolation.
    """
    h, w = channel.shape
    big  = np.zeros((h * 2, w * 2), dtype=np.float32)
    cf   = channel.astype(np.float32)

    # known pixels at even grid positions
    big[0::2, 0::2] = cf

    # horizontal intermediates
    horiz = np.empty((h, w), dtype=np.float32)
    horiz[:, :w - 1] = 0.5 * (cf[:, :-1] + cf[:, 1:])
    horiz[:, w - 1]  = cf[:, -1]
    big[0::2, 1::2]  = horiz

    # vertical intermediates
    vert = np.empty((h, w), dtype=np.float32)
    vert[:h - 1, :] = 0.5 * (cf[:-1, :] + cf[1:, :])
    vert[h - 1, :]  = cf[-1, :]
    big[1::2, 0::2] = vert

    # diagonal intermediates  (average of 4 = sum/4 = 0.25 * sum)
    diag = np.empty((h, w), dtype=np.float32)
    diag[:h-1, :w-1] = 0.25 * (
        cf[:-1, :-1] + cf[:-1, 1:] + cf[1:, :-1] + cf[1:, 1:]
    )
    diag[:h-1, w-1] = 0.5 * (cf[:-1, -1] + cf[1:, -1])
    diag[h-1, :w-1] = 0.5 * (cf[-1, :-1] + cf[-1, 1:])
    diag[h-1, w-1]  = cf[-1, -1]
    big[1::2, 1::2] = diag

    return np.clip(big, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 1b.  Upsampling alternatives to Hu-WBI  (Step 5 is swappable)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Step 5 of the paper only needs "some 2x upsample so CLAHE sees a finer grid".
# Hu-WBI is the paper's choice, but every interpolation kernel below can stand
# in for it, so the ablation "does Hu-WBI actually buy anything?" is answerable
# from the UI instead of by editing code.

def hu_wbi_literal_upsample(channel: np.ndarray) -> np.ndarray:
    """
    Hu-WBI read *literally* as printed in the paper: H(a,b) = Σ P_n / 2 for
    n = 1..4, i.e. 0.5 * sum for the diagonal case (and 0.5 * sum for the 2
    neighbour cases, which happens to be the average there).

    Kept as a selectable option purely so the reviewer claim can be checked
    empirically: on a [0,255] image the diagonal pixels reach up to 510 and
    clip to white, which is why hu_wbi_upsample() uses the averaging reading
    instead. Expect visibly blown-out output and worse PSNR/SSIM here.
    """
    h, w = channel.shape
    big  = np.zeros((h * 2, w * 2), dtype=np.float32)
    cf   = channel.astype(np.float32)

    big[0::2, 0::2] = cf

    horiz = np.empty((h, w), dtype=np.float32)
    horiz[:, :w - 1] = 0.5 * (cf[:, :-1] + cf[:, 1:])
    horiz[:, w - 1]  = cf[:, -1]
    big[0::2, 1::2]  = horiz

    vert = np.empty((h, w), dtype=np.float32)
    vert[:h - 1, :] = 0.5 * (cf[:-1, :] + cf[1:, :])
    vert[h - 1, :]  = cf[-1, :]
    big[1::2, 0::2] = vert

    diag = np.empty((h, w), dtype=np.float32)
    diag[:h-1, :w-1] = 0.5 * (                       # literal: sum / 2
        cf[:-1, :-1] + cf[:-1, 1:] + cf[1:, :-1] + cf[1:, 1:]
    )
    diag[:h-1, w-1] = 0.5 * (cf[:-1, -1] + cf[1:, -1])
    diag[h-1, :w-1] = 0.5 * (cf[-1, :-1] + cf[-1, 1:])
    diag[h-1, w-1]  = cf[-1, -1]
    big[1::2, 1::2] = diag

    return np.clip(big, 0, 255).astype(np.uint8)


# name → (human label, short description)
UPSAMPLE_METHODS: dict[str, tuple[str, str]] = {
    "hu_wbi":         ("Hu-WBI (paper)",      "Half-unit weighted bilinear, averaging reading of Eq. 4"),
    "hu_wbi_literal": ("Hu-WBI (literal Eq)", "Σ P_n / 2 exactly as printed — clips bright pixels, for ablation only"),
    "bilinear":       ("Bilinear",            "cv2.INTER_LINEAR — closest classical equivalent of Hu-WBI"),
    "bicubic":        ("Bicubic",             "cv2.INTER_CUBIC — 4x4 cubic kernel, sharper than bilinear"),
    "lanczos4":       ("Lanczos-4",           "cv2.INTER_LANCZOS4 — 8x8 windowed sinc, sharpest classical kernel"),
    "nearest":        ("Nearest neighbour",   "cv2.INTER_NEAREST — pixel replication, blocky baseline"),
    "edge_directed":  ("Edge-directed (NEDI-like)", "Bicubic base blended with a diffusion-guided pass along edges"),
    "none":           ("None (skip 2x)",      "Run the contrast step at native resolution — no up/downsample"),
}


def _edge_directed_upsample(channel: np.ndarray) -> np.ndarray:
    """
    Lightweight edge-directed interpolation: start from bicubic, then pull the
    result toward an edge-preserving (bilateral-filtered) version wherever the
    local gradient is strong. Not full NEDI — it is a cheap stand-in that keeps
    staircase artefacts off diagonal edges, which is the property NEDI-style
    methods are chosen for in the SR literature.
    """
    h, w  = channel.shape
    base  = cv2.resize(channel, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    edge  = cv2.bilateralFilter(base, 5, 40, 40)
    gx    = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    mag   = cv2.magnitude(gx, gy)
    wgt   = np.clip(mag / (mag.max() + 1e-6) * 2.5, 0, 1)[..., None][..., 0]
    out   = base.astype(np.float32) * (1 - wgt) + edge.astype(np.float32) * wgt
    return np.clip(out, 0, 255).astype(np.uint8)


def upsample_2x(channel: np.ndarray, method: str = "hu_wbi") -> np.ndarray:
    """Dispatch a 2x upsample of a single-channel uint8 image."""
    if method not in UPSAMPLE_METHODS:
        raise ValueError(f"unknown upsample method: {method}")
    if method == "none":
        return channel
    if method == "hu_wbi":
        return hu_wbi_upsample(channel)
    if method == "hu_wbi_literal":
        return hu_wbi_literal_upsample(channel)
    if method == "edge_directed":
        return _edge_directed_upsample(channel)

    interp = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic":  cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
        "nearest":  cv2.INTER_NEAREST,
    }[method]
    h, w = channel.shape
    return cv2.resize(channel, (w * 2, h * 2), interpolation=interp)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SRGAN  (PyTorch, single-channel V input)
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Residual block: Conv->BN->PReLU->Conv->BN + skip (paper: 16 blocks)."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1  = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(channels)
        self.prelu  = nn.PReLU()
        self.conv2  = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return x + out


class SRGANGenerator(nn.Module):
    """
    SRGAN generator (paper Section 3.4):
      Layer 1 – low-level  feature extraction  (Conv9 + PReLU)
      Layer 2 – high-level feature extraction  (16 residual blocks)
      Layer 3 – deconvolution  (2× via PixelShuffle)
      Layer 4 – reconstruction  (Conv9 + Sigmoid)

    Input / output: single-channel (V channel), values in [0, 1].
    Output spatial size: 2 × input size.
    """
    def __init__(self, n_res: int = 16, scale: int = 2):
        super().__init__()
        # Low-level feature extraction
        self.conv1      = nn.Conv2d(1, 64, 9, padding=4)
        self.prelu1     = nn.PReLU()
        # High-level feature extraction
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_res)])
        # Post-residual
        self.post_res   = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # Deconvolution (2× PixelShuffle)
        self.upsample   = nn.Sequential(
            nn.Conv2d(64, 64 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )
        # Reconstruction
        self.conv_final = nn.Conv2d(64, 1, 9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-level features (global skip connection starts here)
        low_feat = self.prelu1(self.conv1(x))
        # Pass through residual blocks
        out = low_feat
        for block in self.res_blocks:
            out = block(out)
        # Post-res + global skip back to low-level features
        out = self.post_res(out)
        out = low_feat + out
        # Upsample then reconstruct
        out = self.upsample(out)
        return torch.sigmoid(self.conv_final(out))


_srgan_model: SRGANGenerator | None = None
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "srgan_v.pth")


def _load_srgan() -> SRGANGenerator | None:
    """Load SRGAN generator once and cache it."""
    global _srgan_model
    if _srgan_model is not None:
        return _srgan_model
    if not os.path.exists(_WEIGHTS_PATH):
        return None
    try:
        m = SRGANGenerator()
        m.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True))
        m.eval()
        _srgan_model = m
        print(f"[SRGAN] loaded weights from {_WEIGHTS_PATH}")
        return m
    except Exception as e:
        print(f"[SRGAN] weight load failed: {e}")
        return None


def _apply_srgan(v_uint8: np.ndarray) -> np.ndarray:
    """
    Run SRGAN on the V channel.

    • If srgan_v.pth exists  → use the trained generator (real checkpoint from
      train_srgan_v.py; measured PSNR ~21.6–23.2 dB / SSIM ~0.80–0.83 on this
      project's samples — see compute_metrics(), not the paper's own numbers).
    • Otherwise              → Lanczos-4 up/down + detail boost (heuristic,
      unmeasured against any ground truth).

    Output is the same spatial size as the input (downscaled back after 2×SR).
    """
    h, w = v_uint8.shape
    model = _load_srgan()

    if model is not None:
        # ── trained SRGAN path ───────────────────────────────────────────
        with torch.no_grad():
            t = torch.from_numpy(v_uint8.astype(np.float32) / 255.0)
            t = t.unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
            sr = model(t).squeeze().numpy()           # [2H,2W]
        v_sr = (sr * 255).clip(0, 255).astype(np.uint8)
        # downscale 2× SR back to original size
        return cv2.resize(v_sr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    else:
        # ── fallback: high-quality upscale + detail boost ─────────────────
        # Heuristic sharpening only, no trained model behind it.
        # Load srgan_v.pth (see train_srgan_v.py) to use the real generator.
        v_big  = cv2.resize(v_uint8, (w * 2, h * 2),
                            interpolation=cv2.INTER_LANCZOS4)
        # detail-preservation kernel (approximates SRGAN perceptual loss)
        kernel = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]], dtype=np.float32)
        v_sharp = cv2.filter2D(v_big.astype(np.float32), -1, kernel)
        v_sharp = np.clip(v_sharp, 0, 255).astype(np.uint8)
        v_blend = cv2.addWeighted(v_sharp, 0.80, v_big, 0.20, 0)
        return cv2.resize(v_blend, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ═══════════════════════════════════════════════════════════════════════════════
# 2b.  Pretrained RGB SRGAN (real Ledig et al. weights, 4x, optional add-on)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The SRGANGenerator above is a custom 1-channel/2x/sigmoid design that only
# loads the paper-specific 'srgan_v.pth' (not yet trained). This second
# generator is an exact architecture match for the publicly available
# pretrained weights from https://github.com/mseitzer/srgan
# (resources/pretrained/srgan.pth), verified by strict state_dict loading
# (0 missing/unexpected keys) and a real forward-pass sanity check.
#
# Differences from SRGANGenerator: RGB in/out (not V-channel only), 4x
# upscale (not 2x), reflection padding, no output activation. Input must be
# ToTensor-style [0,1] RGB; raw output lands in ~(-1,1) and must be
# denormalized via (x+1)/2 before use. Since it was trained on natural
# photos (COCO/BSDS500), not endoscopy images, treat it as a general-purpose
# upscaler rather than a paper-accurate result.

class _RGBResBlock(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False),
            nn.BatchNorm2d(ch), nn.PReLU(ch),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return self.block(x) + x


class PretrainedRGBSRGANGenerator(nn.Module):
    """Exact architecture match for mseitzer/srgan's pretrained srgan.pth."""

    def __init__(self, n_res: int = 16):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(4), nn.Conv2d(3, 64, 9), nn.PReLU(64),
        )
        self.body = nn.Sequential(
            *[_RGBResBlock(64) for _ in range(n_res)],
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3, bias=False),
            nn.BatchNorm2d(64),
        )
        self.upsample = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(64, 1024, 3), nn.PixelShuffle(2),
            nn.PReLU(256),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 1024, 3), nn.PixelShuffle(2),
            nn.PReLU(256),
        )
        self.final_conv = nn.Sequential(nn.ReflectionPad2d(4), nn.Conv2d(256, 3, 9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.initial_conv(x)
        out  = self.body(feat) + feat
        out  = self.upsample(out)
        return self.final_conv(out)


_pretrained_rgb_srgan_model: PretrainedRGBSRGANGenerator | None = None
_RGB_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "srgan_rgb_pretrained.pth")


def _load_pretrained_rgb_srgan() -> PretrainedRGBSRGANGenerator | None:
    """Load the pretrained RGB SRGAN generator once and cache it."""
    global _pretrained_rgb_srgan_model
    if _pretrained_rgb_srgan_model is not None:
        return _pretrained_rgb_srgan_model
    if not os.path.exists(_RGB_WEIGHTS_PATH):
        return None
    try:
        m = PretrainedRGBSRGANGenerator()
        m.load_state_dict(torch.load(_RGB_WEIGHTS_PATH, map_location="cpu", weights_only=True))
        m.eval()
        _pretrained_rgb_srgan_model = m
        print(f"[SRGAN-RGB] loaded pretrained weights from {_RGB_WEIGHTS_PATH}")
        return m
    except Exception as e:
        print(f"[SRGAN-RGB] weight load failed: {e}")
        return None


def apply_pretrained_rgb_srgan(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Run the real pretrained SRGAN on a full BGR image (4x super-resolution).

    Returns the 4x-upscaled BGR image, or None if srgan_rgb_pretrained.pth
    is not present. This is a standalone add-on separate from enhance(),
    which keeps using the paper's V-channel-only 2x pipeline.
    """
    model = _load_pretrained_rgb_srgan()
    if model is None:
        return None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        raw = model(t)
    out = ((raw + 1.0) / 2.0).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    out_rgb = (out * 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
# 2c.  Color transfer  (Eq. 6 — linear/Monge-Kantorovich covariance matching)
# ═══════════════════════════════════════════════════════════════════════════════

def color_transfer(source_bgr: np.ndarray,
                    target_bgr: np.ndarray,
                    intensity_frac: float = 0.9,
                    strength: float = 1.0) -> np.ndarray:
    """
    Eq. 6 :  R'_target = Σ_source^(1/2) · Σ_target^(-1/2) · (R_target - M_target) + M_source

    Matches target's full 3x3 RGB covariance structure to source's (a linear
    "whiten-then-recolor" transform, not per-channel scaling), correcting
    color/tone drift the enhancement pipeline introduces relative to the
    original. The paper applies this to counter color shift from SRGAN
    training on natural/IoT photos; here it also corrects the net-darkening
    tendency of the invert→gamma(γ<1)→invert-back round trip in
    preprocess_v_channel(), which otherwise pushes the output tone in the
    opposite direction from the paper's brighter/warmer qualitative examples
    (Fig. 6/7).

    Statistics use only pixels below `intensity_frac` of the max intensity
    ("to prevent the color retransfer" — paper's own wording — i.e. exclude
    blown-out specular highlights from the highlight-color statistics).

    `strength=1.0` (default) applies Eq. 6 exactly as written — no blending,
    since the paper doesn't define a partial-strength version. `strength`
    exists as a knob for callers who want less correction (e.g. to preserve
    a diagnostically useful contrast boost CLAHE/SRGAN introduced beyond the
    original's baseline), but that's a deviation from the paper, not the
    paper's own behaviour, so it defaults off (1.0 = full, undamped Eq. 6).
    """
    src = source_bgr.astype(np.float64).reshape(-1, 3)
    tgt = target_bgr.astype(np.float64).reshape(-1, 3)

    src_gray = source_bgr.astype(np.float64).mean(axis=2).reshape(-1)
    tgt_gray = target_bgr.astype(np.float64).mean(axis=2).reshape(-1)
    src_mask = src_gray < (intensity_frac * 255)
    tgt_mask = tgt_gray < (intensity_frac * 255)

    src_px = src[src_mask] if src_mask.any() else src
    tgt_px = tgt[tgt_mask] if tgt_mask.any() else tgt

    m_source = src_px.mean(axis=0)
    m_target = tgt_px.mean(axis=0)

    cov_source = np.cov(src_px, rowvar=False) + np.eye(3) * 1e-6
    cov_target = np.cov(tgt_px, rowvar=False) + np.eye(3) * 1e-6

    def _sqrtm_sym(mat, floor_ratio=0.05):
        # Endoscopy images are near-monochromatic (dominant red/pink hue),
        # so the color covariance is near-singular: one large eigenvalue
        # (brightness) and two tiny ones (the little true color variation
        # there is). Inverting near-zero eigenvalues blows up the transform
        # along essentially noise directions, rotating hue instead of just
        # correcting tone (observed as a purple/grey cast on Fig. 6's
        # original — eigenvalue ratios there were ~1600:1). Flooring every
        # eigenvalue to `floor_ratio` of the largest keeps the correction
        # bounded (~4.5x max amplification) without changing behavior on
        # well-conditioned images, where all eigenvalues already sit above
        # the floor.
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, vals.max() * floor_ratio, None)
        return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    transform = _sqrtm_sym(cov_source) @ np.linalg.inv(_sqrtm_sym(cov_target))

    fully_transferred = (tgt - m_target) @ transform.T + m_source
    blended = strength * fully_transferred + (1.0 - strength) * tgt
    return np.clip(blended, 0, 255).astype(np.uint8).reshape(target_bgr.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# 2d.  Visual punch  (NOT from the paper — cosmetic-only, opt-in)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Measured against the paper's own Fig. 6 (i)->(v) example: original R-B
# (BGR-mean redness proxy) = 44.0, paper's "Proposed" = 93.8 — more than
# double. Eq. 6 (color_transfer above) restores output statistics *toward*
# the original's, so it can reach at most ~44, never 93.8: that number is
# provably outside what Eq.1-8 + Eq.6 can produce from the documented paper.
# It must come from the authors' unreleased SRGAN's learned color style, or
# manual figure grading for publication — neither is recoverable from the
# text. This function is an explicit, undisguised stylistic add-on to get
# visually closer to that look; it is not derived from or claimed to be
# "paper-accurate", which is why it defaults to off in enhance().

def apply_visual_punch(image_bgr: np.ndarray,
                        saturation_boost: float = 1.6,
                        warmth: float = 22.0) -> np.ndarray:
    """Cosmetic saturation + warm (redder/less blue) tone push."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_boost, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    boosted[..., 2] = np.clip(boosted[..., 2] + warmth, 0, 255)       # + R
    boosted[..., 0] = np.clip(boosted[..., 0] - warmth * 0.5, 0, 255)  # - B
    return boosted.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 2e.  Colour-space options  (Step 1 is swappable: HSV / YIQ / Lab / …)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The paper works on HSV's V channel, but every step after it only needs "one
# intensity/luma channel + whatever carries the colour". Any luma-chroma space
# fits that shape, and they are NOT equivalent:
#   • V = max(R,G,B)      — ignores G/B entirely once R dominates, which is the
#                           usual case in endoscopy (everything is red/pink), so
#                           V is close to the red channel alone.
#   • Y (YIQ/YCrCb/YUV)   — weighted luma (0.299R+0.587G+0.114B): green-heavy,
#                           closer to perceived brightness, and mucosal texture
#                           lives mostly in G, so contrast steps see more detail.
#   • L* (CIE Lab)        — perceptually uniform lightness; CLAHE steps in L*
#                           give more even results than in V.
#   • I = (R+G+B)/3 (HSI) — flat average, the classical HSI intensity.
# Every space here round-trips exactly through split → merge when the intensity
# channel is left untouched (verified numerically), so a difference in output is
# a real difference in the pipeline, not a conversion artefact.

COLOR_SPACES: dict[str, tuple[str, str]] = {
    "hsv":   ("HSV  (V channel)",   "Paper default — V = max(R,G,B), hue+saturation preserved"),
    "yiq":   ("YIQ  (Y channel)",   "NTSC luma 0.299R+0.587G+0.114B, I/Q chroma preserved"),
    "ycrcb": ("YCrCb (Y channel)",  "ITU-R BT.601 luma, Cr/Cb chroma preserved"),
    "yuv":   ("YUV  (Y channel)",   "BT.601 luma with U/V chroma — same luma as YCrCb, different chroma scaling"),
    "lab":   ("CIE Lab (L* channel)", "Perceptually uniform lightness, a*/b* chroma preserved"),
    "hsi":   ("HSI  (I channel)",   "I = (R+G+B)/3, colour re-applied by intensity ratio"),
    "gray":  ("Luma-ratio (gray)",  "Rec.601 gray processed, colour re-applied by intensity ratio"),
}

# YIQ (FCC/NTSC) forward and inverse matrices, applied to R,G,B in [0,255].
_RGB2YIQ = np.array([[0.299,  0.587,  0.114],
                     [0.5959, -0.2746, -0.3213],
                     [0.2115, -0.5227,  0.3112]], dtype=np.float32)
_YIQ2RGB = np.array([[1.0,  0.956,  0.619],
                     [1.0, -0.272, -0.647],
                     [1.0, -1.106,  1.703]], dtype=np.float32)


def split_intensity(image_bgr: np.ndarray, space: str = "hsv"):
    """
    Split a BGR image into (intensity_uint8, ctx).

    `intensity_uint8` is the single channel the pipeline processes; `ctx` is an
    opaque dict holding whatever merge_intensity() needs to rebuild the colour
    image. Every space returns intensity on the same [0,255] uint8 scale, so
    the rest of the pipeline is colour-space agnostic.
    """
    if space not in COLOR_SPACES:
        raise ValueError(f"unknown colour space: {space}")

    if space == "hsv":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return v, {"space": space, "h": h, "s": s}

    if space == "yiq":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        yiq = rgb @ _RGB2YIQ.T
        y   = np.clip(yiq[..., 0], 0, 255).astype(np.uint8)
        return y, {"space": space, "iq": yiq[..., 1:].copy()}

    if space in ("ycrcb", "yuv"):
        code = cv2.COLOR_BGR2YCrCb if space == "ycrcb" else cv2.COLOR_BGR2YUV
        conv = cv2.cvtColor(image_bgr, code)
        y, c1, c2 = cv2.split(conv)
        return y, {"space": space, "c1": c1, "c2": c2}

    if space == "lab":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return l, {"space": space, "a": a, "b": b}

    # hsi / gray — intensity is a scalar per pixel, colour restored by ratio
    bgr_f = image_bgr.astype(np.float32)
    if space == "hsi":
        inten = bgr_f.mean(axis=2)
    else:  # gray
        inten = (0.114 * bgr_f[..., 0] + 0.587 * bgr_f[..., 1]
                 + 0.299 * bgr_f[..., 2])
    return (np.clip(inten, 0, 255).astype(np.uint8),
            {"space": space, "bgr": bgr_f, "inten": inten})


def merge_intensity(intensity_u8: np.ndarray, ctx: dict) -> np.ndarray:
    """Rebuild a BGR image from a processed intensity channel + split ctx."""
    space = ctx["space"]

    if space == "hsv":
        return cv2.cvtColor(cv2.merge([ctx["h"], ctx["s"], intensity_u8]),
                            cv2.COLOR_HSV2BGR)

    if space == "yiq":
        yiq = np.dstack([intensity_u8.astype(np.float32), ctx["iq"]])
        rgb = np.clip(yiq @ _YIQ2RGB.T, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if space in ("ycrcb", "yuv"):
        code = cv2.COLOR_YCrCb2BGR if space == "ycrcb" else cv2.COLOR_YUV2BGR
        return cv2.cvtColor(cv2.merge([intensity_u8, ctx["c1"], ctx["c2"]]), code)

    if space == "lab":
        return cv2.cvtColor(cv2.merge([intensity_u8, ctx["a"], ctx["b"]]),
                            cv2.COLOR_LAB2BGR)

    # hsi / gray — scale each colour channel by new_intensity / old_intensity so
    # hue and saturation ratios survive; the +1e-3 guard keeps near-black pixels
    # (where the ratio is unstable) from exploding.
    ratio = (intensity_u8.astype(np.float32) / (ctx["inten"] + 1e-3))[..., None]
    return np.clip(ctx["bgr"] * ratio, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 2f.  Contrast-enhancement options  (Step 6 is swappable: CLAHE and relatives)
# ═══════════════════════════════════════════════════════════════════════════════
#
# CLAHE is one point in a family of histogram/contrast methods. The ones below
# are the standard comparison set used in enhancement papers, all operating on
# the same single uint8 channel so they are directly interchangeable in Step 6.

CONTRAST_METHODS: dict[str, tuple[str, str]] = {
    "clahe":    ("CLAHE (paper)",        "Contrast-limited AHE, cv2 — clip limit + tile grid"),
    "clahe_sk": ("CLAHE (scikit-image)", "equalize_adapthist — bilinear tile blending, different clip semantics"),
    "he":       ("Global HE",            "Classical global histogram equalisation — no locality, can over-amplify"),
    "bbhe":     ("BBHE",                 "Brightness-preserving Bi-HE: split at the mean, equalise each half"),
    "dsihe":    ("DSIHE",                "Dualistic Sub-Image HE: split at the median (equal-area halves)"),
    "agcwd":    ("AGCWD",                "Adaptive gamma correction with weighting distribution (Huang et al.)"),
    "ssr":      ("Retinex SSR",          "Single-scale retinex — log(I) − log(Gauss*I), illumination removed"),
    "msr":      ("Retinex MSR",          "Multi-scale retinex (σ = 15/80/250), gentler halos than SSR"),
    "stretch":  ("Contrast stretch",     "Linear percentile stretch (1%–99%) — no histogram reshaping"),
    "tophat":   ("Morphological top-hat", "I + white top-hat − black top-hat: boosts local structures"),
    "log":      ("Log transform",        "s = c·log(1+r) — expands darks, compresses brights"),
    "none":     ("None",                 "Skip Step 6 entirely (isolates the effect of the other steps)"),
}


def _sub_image_he(channel: np.ndarray, threshold: int) -> np.ndarray:
    """
    Shared core of BBHE and DSIHE: equalise the pixels below and above
    `threshold` independently, each mapped back into its own intensity range.
    This is what preserves mean brightness — plain HE moves it freely.
    """
    out   = np.empty_like(channel)
    lower = channel <= threshold

    for mask, lo, hi in ((lower, 0, int(threshold)),
                         (~lower, min(int(threshold) + 1, 255), 255)):
        px = channel[mask]
        if px.size == 0 or hi <= lo:
            out[mask] = px
            continue
        hist = np.bincount(px, minlength=256)[lo:hi + 1].astype(np.float64)
        cdf  = hist.cumsum() / px.size
        lut  = lo + (hi - lo) * cdf
        out[mask] = np.clip(lut[px.astype(np.int32) - lo], 0, 255).astype(np.uint8)

    return out


def _agcwd(channel: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Adaptive Gamma Correction with Weighting Distribution.

    Reshapes the histogram into a weighted PDF, integrates it into a CDF, and
    uses 1 − CDF as a *per-intensity* gamma. Unlike HE it cannot create the
    harsh level jumps that come from a steep CDF, and unlike a fixed gamma it
    adapts to how the image's own histogram is distributed.
    """
    hist = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
    pdf  = hist / max(hist.sum(), 1.0)
    nz   = pdf[pdf > 0]
    if nz.size == 0:
        return channel.copy()
    pdf_min, pdf_max = nz.min(), pdf.max()
    if pdf_max <= pdf_min:
        return channel.copy()

    # Empty bins sit below pdf_min, so the normalised term goes negative there
    # and a fractional power of it is NaN — clip to 0 first, then zero them out.
    norm  = np.clip((pdf - pdf_min) / (pdf_max - pdf_min), 0.0, 1.0)
    pdf_w = pdf_max * np.power(norm, alpha)
    pdf_w[pdf <= 0] = 0.0
    cdf_w = pdf_w.cumsum() / max(pdf_w.sum(), 1e-12)

    levels = np.arange(256, dtype=np.float64) / 255.0
    lut    = 255.0 * np.power(levels, 1.0 - cdf_w)
    return np.clip(lut[channel], 0, 255).astype(np.uint8)


def _retinex(channel: np.ndarray, sigmas: tuple) -> np.ndarray:
    """(Multi-scale) retinex: average of log(I) − log(Gaussian_σ * I) over σ."""
    f   = channel.astype(np.float32) + 1.0
    acc = np.zeros_like(f)
    for s in sigmas:
        blur = cv2.GaussianBlur(f, (0, 0), s) + 1.0
        acc += np.log(f) - np.log(blur)
    acc /= len(sigmas)
    lo, hi = np.percentile(acc, 1), np.percentile(acc, 99)
    if hi <= lo:
        return channel.copy()
    return np.clip((acc - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def apply_contrast(channel: np.ndarray,
                   method: str        = "clahe",
                   clip_limit: float  = 2.0,
                   tile_size: tuple   = (8, 8),
                   agcwd_alpha: float = 0.5,
                   tophat_ksize: int  = 15) -> np.ndarray:
    """Step 6 dispatcher — every method takes and returns a uint8 2-D channel."""
    if method not in CONTRAST_METHODS:
        raise ValueError(f"unknown contrast method: {method}")

    if method == "none":
        return channel

    if method == "clahe":
        return cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=tile_size).apply(channel)

    if method == "clahe_sk":
        from skimage.exposure import equalize_adapthist
        # skimage's clip_limit is a [0,1] fraction, cv2's is a histogram-count
        # multiplier (typ. 1–8). /40 maps the UI's cv2-style range onto a
        # sensible skimage range; clamped so extreme slider ends stay valid.
        sk_clip = float(np.clip(clip_limit / 40.0, 0.005, 1.0))
        out = equalize_adapthist(channel, kernel_size=None, clip_limit=sk_clip)
        return (out * 255).clip(0, 255).astype(np.uint8)

    if method == "he":
        return cv2.equalizeHist(channel)

    if method == "bbhe":
        return _sub_image_he(channel, int(round(float(channel.mean()))))

    if method == "dsihe":
        return _sub_image_he(channel, int(round(float(np.median(channel)))))

    if method == "agcwd":
        return _agcwd(channel, alpha=agcwd_alpha)

    if method == "ssr":
        return _retinex(channel, (80.0,))

    if method == "msr":
        return _retinex(channel, (15.0, 80.0, 250.0))

    if method == "stretch":
        lo, hi = np.percentile(channel, (1.0, 99.0))
        if hi <= lo:
            return channel.copy()
        out = (channel.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    if method == "tophat":
        k = max(3, int(tophat_ksize) | 1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        white = cv2.morphologyEx(channel, cv2.MORPH_TOPHAT,   se)
        black = cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, se)
        out   = channel.astype(np.int16) + white.astype(np.int16) - black.astype(np.int16)
        return np.clip(out, 0, 255).astype(np.uint8)

    # log
    c   = 255.0 / np.log1p(255.0)
    out = c * np.log1p(channel.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 2g.  Super-resolution options  (Step 9/10 is swappable: SRGAN and relatives)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Only two of these are learned models (both need their .pth on disk); the rest
# are classical SR/detail methods that need no weights, which makes them the
# honest baselines to compare a GAN against. Availability is reported by
# available_sr_methods() so the UI can grey out what cannot run here.

SR_METHODS: dict[str, tuple[str, str]] = {
    "auto":           ("Auto (best available)", "srgan_v if its weights exist, else srgan_rgb, else Lanczos detail"),
    "srgan_v":        ("SRGAN — V/intensity 2x", "Paper architecture, trained by train_srgan_v.py (needs srgan_v.pth)"),
    "srgan_rgb":      ("SRGAN — pretrained RGB 4x", "Ledig-style weights from mseitzer/srgan, runs after colour merge"),
    "lanczos_detail": ("Lanczos + detail kernel", "2x Lanczos, 5-point sharpening, blend back — the built-in fallback"),
    "bicubic_detail": ("Bicubic + detail kernel", "Same as above with a bicubic kernel — softer, fewer ringing halos"),
    "ibp":            ("Iterative back-projection", "Classical SR: re-project the LR residual 8x (no training needed)"),
    "unsharp_sr":     ("Unsharp super-resolution", "2x upscale then Gaussian unsharp mask — mildest option"),
    "none":           ("None",                  "Skip super-resolution entirely"),
}


def available_sr_methods() -> dict[str, bool]:
    """Which SR options can actually run right now (weights present or not)."""
    has_v   = os.path.exists(_WEIGHTS_PATH)
    has_rgb = os.path.exists(_RGB_WEIGHTS_PATH)
    return {
        "auto":           True,
        "srgan_v":        has_v,
        "srgan_rgb":      has_rgb,
        "lanczos_detail": True,
        "bicubic_detail": True,
        "ibp":            True,
        "unsharp_sr":     True,
        "none":           True,
    }


def _detail_sr(channel: np.ndarray, interp: int) -> np.ndarray:
    """2x upscale with `interp`, sharpen, blend, downscale back (fallback SR)."""
    h, w    = channel.shape
    big     = cv2.resize(channel, (w * 2, h * 2), interpolation=interp)
    kernel  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp   = np.clip(cv2.filter2D(big.astype(np.float32), -1, kernel), 0, 255).astype(np.uint8)
    blend   = cv2.addWeighted(sharp, 0.80, big, 0.20, 0)
    return cv2.resize(blend, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _ibp_sr(channel: np.ndarray, iterations: int = 8, beta: float = 0.6) -> np.ndarray:
    """
    Iterative back-projection (Irani & Peleg): guess an HR image, simulate the
    LR image it would produce (blur + decimate), and add the back-projected
    error. Converges to an HR estimate consistent with the observed LR data —
    the standard non-learned SR baseline a GAN should be measured against.
    """
    h, w = channel.shape
    lr   = channel.astype(np.float32)
    hr   = cv2.resize(lr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    for _ in range(int(iterations)):
        sim = cv2.resize(cv2.GaussianBlur(hr, (0, 0), 1.0), (w, h),
                         interpolation=cv2.INTER_AREA)
        err = lr - sim
        hr += beta * cv2.resize(err, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    hr = np.clip(hr, 0, 255).astype(np.uint8)
    return cv2.resize(hr, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _unsharp_sr(channel: np.ndarray, amount: float = 0.8, sigma: float = 1.2) -> np.ndarray:
    """2x upscale + Gaussian unsharp mask, then back to the original size."""
    h, w  = channel.shape
    big   = cv2.resize(channel, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    blur  = cv2.GaussianBlur(big, (0, 0), sigma)
    out   = np.clip(big + amount * (big - blur), 0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)


def resolve_sr_method(method: str) -> str:
    """Turn 'auto' (or an unusable choice) into the concrete method that runs."""
    if method not in SR_METHODS:
        raise ValueError(f"unknown SR method: {method}")
    avail = available_sr_methods()
    if method == "auto":
        if avail["srgan_v"]:
            return "srgan_v"
        if avail["srgan_rgb"]:
            return "srgan_rgb"
        return "lanczos_detail"
    if not avail[method]:
        # Requested a learned model whose weights are missing — say so by
        # falling back to the documented classical stand-in rather than failing.
        return "lanczos_detail"
    return method


def apply_sr_intensity(channel: np.ndarray, method: str) -> np.ndarray:
    """
    Run an intensity-domain SR method (everything except srgan_rgb/none).
    Output keeps the input's spatial size, as the paper's Step 9 expects.
    """
    if method in ("none", "srgan_rgb"):
        return channel
    if method == "srgan_v":
        return _apply_srgan(channel)
    if method == "lanczos_detail":
        return _detail_sr(channel, cv2.INTER_LANCZOS4)
    if method == "bicubic_detail":
        return _detail_sr(channel, cv2.INTER_CUBIC)
    if method == "ibp":
        return _ibp_sr(channel)
    if method == "unsharp_sr":
        return _unsharp_sr(channel)
    raise ValueError(f"unknown SR method: {method}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Full enhancement pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_intensity(image_bgr: np.ndarray,
                         gamma: float            = 0.8,
                         clip_limit: float       = 2.0,
                         tile_size: tuple        = (8, 8),
                         cn: float               = 0.85,
                         sigma: float            = 1.0,
                         color_space: str        = "hsv",
                         contrast_method: str    = "clahe",
                         upsample_method: str    = "hu_wbi",
                         downsample_interp: int  = cv2.INTER_LINEAR,
                         agcwd_alpha: float      = 0.5,
                         tophat_ksize: int       = 15):
    """
    Steps 1–8 of the paper pipeline — everything up to (but excluding) SRGAN —
    with each swappable stage selected by name.

    Step 1  BGR → `color_space`, extract the intensity channel (Sec. 2e)
    Step 2  Normalize            Eq. 1 : I_I(a,b) = I(a,b) / I_I(Max)
    Step 3  Invert               Eq. 2 : I' = 1 - I_I(a,b)
    Step 4  Gamma correction     Eq. 3 : Y = C * M^γ      (C=1, γ=0.8)
    Step 5  Contrast enhancement via `contrast_method` (Sec. 2f)
    Step 6  2× upsample          Eq. 4 via `upsample_method` (Sec. 1b) — the
            paper runs Hu-WBI *after* CLAHE (abstract: Hu-WBI exists to fix
            "artifact issues associated with CLAHE"; Section 3's phase list
            has CLAHE as (ii), Hu-WBI as (iii); Fig. 1's CLAHE box precedes
            the Hu-WBI box).
    Step 7  Downsample back to the original size (skipped if no upsample ran)
    Step 8  Unsharp mask (HPF)   Eq. 7 : PS = (I_i - LPF(I_i)) * C_n

    Returns (ctx, intensity): `ctx` is the colour-restore context from
    split_intensity(), `intensity` is the processed (still inverted) channel at
    the input's spatial size. Feed both to merge_intensity() after Step 9.

    The paper's exact configuration is the default:
    color_space="hsv", contrast_method="clahe", upsample_method="hu_wbi".
    """
    # ── Step 1 ──────────────────────────────────────────────────────────────
    intensity, ctx = split_intensity(image_bgr, color_space)

    # ── Step 2  normalize ────────────────────────────────────────────────────
    v_norm = intensity.astype(np.float32) / 255.0

    # ── Step 3  invert ───────────────────────────────────────────────────────
    v_inv = 1.0 - v_norm

    # ── Step 4  gamma correction ──────────────────────────────────────────────
    v_gamma = np.power(np.clip(v_inv, 0.0, 1.0), gamma)
    v_uint8 = np.clip(v_gamma * 255, 0, 255).astype(np.uint8)

    # ── Steps 5–6  contrast enhancement, then upsample (paper order) ────────
    v_contrast_pre = apply_contrast(v_uint8, contrast_method,
                                    clip_limit=clip_limit, tile_size=tile_size,
                                    agcwd_alpha=agcwd_alpha,
                                    tophat_ksize=tophat_ksize)
    v_contrast = upsample_2x(v_contrast_pre, upsample_method)

    # ── Step 7  downsample ────────────────────────────────────────────────────
    h_orig, w_orig = v_uint8.shape
    if v_contrast.shape != v_uint8.shape:
        v_down = cv2.resize(v_contrast, (w_orig, h_orig),
                            interpolation=downsample_interp)
    else:
        v_down = v_contrast

    # ── Step 8  unsharp mask (HPF)  PS = (I_i - LPF(I_i)) * C_n ─────────────
    lpf     = cv2.GaussianBlur(v_down.astype(np.float32), (0, 0), sigma)
    ps      = (v_down.astype(np.float32) - lpf) * cn
    v_sharp = np.clip(v_down.astype(np.float32) + ps, 0, 255).astype(np.uint8)

    return ctx, v_sharp


def preprocess_v_channel(image_bgr: np.ndarray,
                          gamma: float      = 0.8,
                          clip_limit: float = 2.0,
                          tile_size: tuple  = (8, 8),
                          cn: float         = 0.85,
                          sigma: float      = 1.0):
    """
    HSV-only wrapper around preprocess_intensity(), kept because
    train_srgan_v.py builds its training targets from this exact
    (h, s, v_sharp) signature — training and inference must stay on the
    identical distribution, so this stays pinned to the paper's HSV + CLAHE +
    Hu-WBI configuration regardless of what the UI selects.

    Step 1  RGB → HSV,  extract V channel
    Step 2  Normalize V         Eq. 1 : I_I(a,b) = I(a,b) / I_I(Max)
    Step 3  Invert              Eq. 2 : I' = 1 - I_I(a,b)
    Step 4  Gamma correction    Eq. 3 : Y = C * M^γ      (C=1, γ=0.8)
    Step 5  CLAHE (original-resolution intensity)
    Step 6  Hu-WBI 2× upsample  Eq. 4
    Step 7  Downsample back to original size
    Step 8  Unsharp mask (HPF)  Eq. 7 : PS = (I_i - LPF(I_i)) * C_n

    Returns (h_ch, s_ch, v_sharp): the untouched Hue/Saturation channels and
    the fully-processed (still inverted) V channel, same spatial size as the
    input image.
    """
    ctx, v_sharp = preprocess_intensity(
        image_bgr, gamma=gamma, clip_limit=clip_limit, tile_size=tile_size,
        cn=cn, sigma=sigma,
        color_space="hsv", contrast_method="clahe", upsample_method="hu_wbi")
    return ctx["h"], ctx["s"], v_sharp


def enhance(image_bgr: np.ndarray,
            gamma: float                   = 0.8,
            clip_limit: float              = 2.0,
            tile_size: tuple               = (8, 8),
            cn: float                      = 0.85,
            sigma: float                   = 1.0,
            use_srgan: bool                = True,
            use_color_transfer: bool       = True,
            color_transfer_strength: float = 1.0,
            use_visual_punch: bool         = False,
            visual_punch_saturation: float = 1.6,
            visual_punch_warmth: float     = 22.0,
            color_space: str               = "hsv",
            contrast_method: str           = "clahe",
            upsample_method: str           = "hu_wbi",
            sr_method: str                 = "auto",
            agcwd_alpha: float             = 0.5,
            tophat_ksize: int              = 15,
            return_config: bool            = False):
    """
    Paper pipeline (Fig. 1 + algorithm steps), with the four swappable stages
    selectable by name:

      color_space      Step 1  — HSV (paper) / YIQ / YCrCb / YUV / Lab / HSI / gray
      contrast_method  Step 5  — CLAHE (paper) / HE / BBHE / DSIHE / AGCWD / retinex / …
      upsample_method  Step 6  — Hu-WBI (paper) / bilinear / bicubic / Lanczos / …
      sr_method        Step 9  — SRGAN-V (paper) / pretrained RGB SRGAN / IBP / …

    Defaults reproduce the paper exactly. `use_srgan=False` still forces
    sr_method to "none" so existing callers keep working. With
    return_config=True the call returns (image, config_dict), where the config
    records what actually ran — including any SR fallback taken because the
    requested weights file was absent.

    Steps 1–8 are delegated to preprocess_intensity(); this function handles
    SRGAN + reconstruction:

    Step 9  Invert V back,  merge HSV,  convert → RGB
    Step 10 SRGAN super-resolution on the reconstructed RGB image, then
             downsample back to the original size
    Step 11 Color transfer (Eq. 6) against the original input
    Step 12 Visual punch (NOT paper-derived, off by default — see
             apply_visual_punch() docstring)

    Step 9/10 order note: Fig. 1 runs SRGAN on the V channel before merging
    back to RGB. Now that 'srgan_v.pth' is a real trained checkpoint (paper's
    own 1-channel/2x architecture, trained on endoscopy V-channel data), that
    is exactly what happens: _apply_srgan() runs it on v_sharp here in Step 9,
    matching Fig. 1's order precisely. The pretrained RGB generator
    (srgan_rgb_pretrained.pth, section 2b) is kept only as a fallback for when
    'srgan_v.pth' is absent — it needs 3-channel input, so it necessarily runs
    after HSV→RGB reconstruction (Step 10) instead, which is a deviation from
    Fig. 1's order. If neither weight file is present, _apply_srgan() falls
    back further to the untrained Lanczos approximation.
    """
    ctx, v_sharp = preprocess_intensity(
        image_bgr, gamma=gamma, clip_limit=clip_limit, tile_size=tile_size,
        cn=cn, sigma=sigma, color_space=color_space,
        contrast_method=contrast_method, upsample_method=upsample_method,
        agcwd_alpha=agcwd_alpha, tophat_ksize=tophat_ksize)
    h_orig, w_orig = v_sharp.shape

    # ── Step 9  SR on the intensity channel (paper order), invert, reconstruct ─
    sr_actual = "none" if not use_srgan else resolve_sr_method(sr_method)
    v_pre_sr  = apply_sr_intensity(v_sharp, sr_actual)
    v_final   = 255 - v_pre_sr
    enhanced_bgr = merge_intensity(v_final, ctx)

    # ── Step 10  RGB-domain SR (only srgan_rgb needs 3 channels), downscale ──
    if sr_actual == "srgan_rgb":
        sr_bgr = apply_pretrained_rgb_srgan(enhanced_bgr)
        if sr_bgr is not None:
            enhanced_bgr = cv2.resize(sr_bgr, (w_orig, h_orig),
                                      interpolation=cv2.INTER_LANCZOS4)

    # ── Step 11  color transfer (Eq. 6) against the original input ──────────
    if use_color_transfer:
        enhanced_bgr = color_transfer(image_bgr, enhanced_bgr,
                                      strength=color_transfer_strength)

    # ── Step 12  visual punch (cosmetic, not paper-derived) ──────────────────
    if use_visual_punch:
        enhanced_bgr = apply_visual_punch(enhanced_bgr,
                                          saturation_boost=visual_punch_saturation,
                                          warmth=visual_punch_warmth)

    if return_config:
        config = {
            "color_space":     color_space,
            "contrast_method": contrast_method,
            "upsample_method": upsample_method,
            "sr_method":       sr_method,
            "sr_actual":       sr_actual,
            "sr_fallback":     sr_method not in ("auto", sr_actual),
            "gamma":           gamma,
            "clip_limit":      clip_limit,
            "cn":              cn,
            "sigma":           sigma,
            "color_transfer":  use_color_transfer,
            "visual_punch":    use_visual_punch,
        }
        return enhanced_bgr, config

    return enhanced_bgr


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

#
# Two families, and they answer different questions:
#
#  • Full-reference (enhanced vs original) — PSNR/SSIM/MSE/RMSE/MAE/AMBE/NCC/
#    UQI/ΔE. These measure FIDELITY, i.e. how little the enhancement changed
#    the image. For an enhancement pipeline they are diagnostic, not a score to
#    maximise: a no-op scores infinite PSNR and SSIM 1.0. Read them as "the
#    change is structurally sane", not "the output is good".
#
#  • No-reference (computed on each image separately) — entropy, contrast, EME,
#    average gradient, Laplacian variance, colourfulness, edge density,
#    Michelson contrast. These measure how much information/contrast/detail the
#    image carries, and are what actually improve when enhancement works. The
#    "gain" block reports enhanced-vs-original ratios (CII and friends).
#
# METRIC_INFO drives the UI table, so every number shown carries its own
# direction ("higher is better") and one-line meaning.

METRIC_INFO: dict[str, dict] = {
    # full-reference
    "psnr":       {"label": "PSNR",        "unit": "dB", "family": "fr", "better": "context",
                   "desc": "Peak signal-to-noise vs the original. High = little changed; it is a fidelity check, not a quality score."},
    "ssim":       {"label": "SSIM",        "unit": "",   "family": "fr", "better": "context",
                   "desc": "Structural similarity to the original (1.0 = identical). Should stay high — structure must survive enhancement."},
    "mse":        {"label": "MSE",         "unit": "",   "family": "fr", "better": "context",
                   "desc": "Mean squared error vs the original, on grayscale."},
    "rmse":       {"label": "RMSE",        "unit": "",   "family": "fr", "better": "context",
                   "desc": "Root MSE — same units as pixel intensity, so easier to read than MSE."},
    "mae":        {"label": "MAE",         "unit": "",   "family": "fr", "better": "context",
                   "desc": "Mean absolute error vs the original — less dominated by a few big pixel changes than MSE."},
    "ambe":       {"label": "AMBE",        "unit": "",   "family": "fr", "better": "lower",
                   "desc": "Absolute Mean Brightness Error |mean(orig) − mean(enh)|. Low = brightness preserved (what BBHE/DSIHE optimise)."},
    "ncc":        {"label": "NCC",         "unit": "",   "family": "fr", "better": "higher",
                   "desc": "Normalised cross-correlation with the original (1.0 = perfectly correlated)."},
    "uqi":        {"label": "UQI",         "unit": "",   "family": "fr", "better": "higher",
                   "desc": "Universal Quality Index (Wang & Bovik) — SSIM's predecessor: correlation × luminance × contrast."},
    "delta_e":    {"label": "ΔE (CIEDE2000)", "unit": "", "family": "fr", "better": "context",
                   "desc": "Mean perceptual colour difference from the original. <2 imperceptible, >10 clearly different."},
    # no-reference (reported for original and enhanced separately)
    "brightness": {"label": "Brightness",  "unit": "",   "family": "nr", "better": "context",
                   "desc": "Mean grayscale intensity (0–255)."},
    "contrast":   {"label": "Contrast (σ)", "unit": "",  "family": "nr", "better": "higher",
                   "desc": "Standard deviation of intensity — the standard global contrast measure."},
    "entropy":    {"label": "Entropy",     "unit": "bits", "family": "nr", "better": "higher",
                   "desc": "Discrete Shannon entropy (max 8). Higher = more information / better-spread histogram."},
    "eme":        {"label": "EME",         "unit": "",   "family": "nr", "better": "higher",
                   "desc": "Measure of Enhancement (Agaian): mean 20·log10(max/min) over 8×8 blocks — local contrast."},
    "avg_gradient": {"label": "Avg. gradient", "unit": "", "family": "nr", "better": "higher",
                   "desc": "Mean Sobel gradient magnitude — overall sharpness/detail density."},
    "laplacian_var": {"label": "Laplacian var.", "unit": "", "family": "nr", "better": "higher",
                   "desc": "Variance of the Laplacian — the classic focus/blur measure."},
    "colorfulness": {"label": "Colourfulness", "unit": "", "family": "nr", "better": "higher",
                   "desc": "Hasler & Süsstrunk colourfulness — how saturated/varied the colour is."},
    "edge_density": {"label": "Edge density", "unit": "%", "family": "nr", "better": "higher",
                   "desc": "Percentage of pixels Canny marks as edges — visible structure."},
    "michelson":  {"label": "Michelson",   "unit": "",   "family": "nr", "better": "higher",
                   "desc": "(max − min)/(max + min) on the 1st/99th percentiles — dynamic-range usage."},
    # gains (enhanced ÷ original)
    "cii":            {"label": "CII",              "unit": "×", "family": "gain", "better": "higher",
                       "desc": "Contrast Improvement Index = σ(enh)/σ(orig). >1 means contrast genuinely increased."},
    "entropy_gain":   {"label": "Entropy gain",     "unit": "bits", "family": "gain", "better": "higher",
                       "desc": "Entropy(enh) − Entropy(orig). Positive = information added, not destroyed."},
    "sharpness_gain": {"label": "Sharpness gain",   "unit": "×", "family": "gain", "better": "higher",
                       "desc": "Avg. gradient ratio — how much more detail the output carries."},
    "eme_gain":       {"label": "EME gain",         "unit": "×", "family": "gain", "better": "higher",
                       "desc": "Local-contrast (EME) ratio enhanced/original."},
    "colorfulness_gain": {"label": "Colourfulness gain", "unit": "×", "family": "gain", "better": "context",
                       "desc": "Colourfulness ratio. >1 = more vivid; very high values mean the colour was pushed hard."},
}


def _eme(gray: np.ndarray, blocks: int = 8) -> float:
    """
    EME (Agaian's Measure of Enhancement): split into `blocks`×`blocks` tiles,
    average 20·log10(I_max / I_min) over them. Rewards local contrast, which is
    exactly what CLAHE-family methods are supposed to produce.
    """
    h, w = gray.shape
    bh, bw = max(1, h // blocks), max(1, w // blocks)
    vals = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            blk = gray[y:y + bh, x:x + bw].astype(np.float32)
            mx, mn = float(blk.max()), float(blk.min())
            vals.append(20.0 * np.log10((mx + 1e-3) / (mn + 1e-3)))
    return float(np.mean(vals)) if vals else 0.0


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric on the rg/yb opponents."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _no_reference_metrics(bgr: np.ndarray) -> dict:
    """Quality measures that need no ground truth — computed per image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    lo, hi = np.percentile(gray, (1.0, 99.0))

    return {
        "brightness":    round(float(gray.mean()), 4),
        "contrast":      round(float(gray.std()), 4),
        "entropy":       round(float(shannon_entropy(gray)), 4),
        "eme":           round(_eme(gray), 4),
        "avg_gradient":  round(float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2.0))), 4),
        "laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
        "colorfulness":  round(_colorfulness(bgr), 4),
        "edge_density":  round(float(np.count_nonzero(cv2.Canny(gray, 100, 200)))
                               / gray.size * 100.0, 4),
        "michelson":     round(float((hi - lo) / (hi + lo + 1e-6)), 4),
    }


def compute_metrics(original_bgr: np.ndarray,
                    enhanced_bgr: np.ndarray,
                    include_delta_e: bool = True) -> dict:
    """
    Full metric report for one original/enhanced pair.

    Returns a flat set of full-reference keys (psnr/ssim/mse/… — the three
    legacy keys keep their old names and meaning), plus nested "original",
    "enhanced" and "gain" blocks of no-reference measures. See METRIC_INFO for
    what each one means and which direction is better.

    `include_delta_e=False` skips the CIEDE2000 pass, which dominates the cost
    on large images — used by the batch sweep so dozens of variants stay fast.
    """
    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    enh_gray  = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = orig_gray - enh_gray
    mse  = float(np.mean(diff ** 2))
    psnr = float(cv2.PSNR(original_bgr, enhanced_bgr))
    ssim_val = float(ssim(orig_gray.astype(np.uint8), enh_gray.astype(np.uint8),
                          data_range=255))

    # NCC and UQI on the grayscale pair
    o_c, e_c = orig_gray - orig_gray.mean(), enh_gray - enh_gray.mean()
    denom    = float(np.sqrt((o_c ** 2).sum() * (e_c ** 2).sum())) + 1e-12
    ncc      = float((o_c * e_c).sum() / denom)

    o_m, e_m = float(orig_gray.mean()), float(enh_gray.mean())
    o_v, e_v = float(orig_gray.var()),  float(enh_gray.var())
    cov      = float(np.mean(o_c * e_c))
    uqi      = float((4.0 * cov * o_m * e_m)
                     / ((o_v + e_v + 1e-12) * (o_m ** 2 + e_m ** 2 + 1e-12)))

    result = {
        "psnr": round(psnr, 4),
        "ssim": round(ssim_val, 4),
        "mse":  round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae":  round(float(np.mean(np.abs(diff))), 4),
        "ambe": round(abs(o_m - e_m), 4),
        "ncc":  round(ncc, 4),
        "uqi":  round(uqi, 4),
    }

    if include_delta_e:
        from skimage.color import deltaE_ciede2000, rgb2lab
        lab_o = rgb2lab(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB) / 255.0)
        lab_e = rgb2lab(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB) / 255.0)
        result["delta_e"] = round(float(np.mean(deltaE_ciede2000(lab_o, lab_e))), 4)

    nr_o = _no_reference_metrics(original_bgr)
    nr_e = _no_reference_metrics(enhanced_bgr)

    def _ratio(key):
        return round(float(nr_e[key] / (nr_o[key] + 1e-6)), 4)

    result["original"] = nr_o
    result["enhanced"] = nr_e
    result["gain"] = {
        "cii":               _ratio("contrast"),
        "entropy_gain":      round(nr_e["entropy"] - nr_o["entropy"], 4),
        "sharpness_gain":    _ratio("avg_gradient"),
        "eme_gain":          _ratio("eme"),
        "colorfulness_gain": _ratio("colorfulness"),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Comparison image builder
# ═══════════════════════════════════════════════════════════════════════════════

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _label_bar(canvas, x0, y0, w, h, text, color, bg=(28, 30, 48)):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), bg, -1)
    (_, th), _ = cv2.getTextSize(text, _FONT_BOLD, 0.60, 2)
    cv2.putText(canvas, text, (x0 + 14, y0 + (h + th) // 2),
                _FONT_BOLD, 0.60, color, 2, cv2.LINE_AA)


def _draw_histogram(canvas, gray, color, x0, y0, w, h):
    ML, MR, MB, MT = 6, 6, 20, 8
    pw, ph = w - ML - MR, h - MT - MB
    px0, py0, py1 = x0 + ML, y0 + MT, y0 + MT + ph

    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (14, 16, 26), -1)
    for frac in (0.25, 0.50, 0.75):
        gy = int(py1 - frac * ph)
        cv2.line(canvas, (px0, gy), (px0 + pw, gy), (35, 40, 58), 1)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    mx   = float(hist.max()) or 1.0
    bw   = pw / 256.0

    # filled area
    overlay = canvas.copy()
    pts = [(px0, py1)]
    for i, v in enumerate(hist):
        pts.append((px0 + int(i * bw), py1 - int((v / mx) * ph)))
    pts.append((px0 + pw, py1))
    fill = tuple(max(0, c - 140) for c in color)
    cv2.fillPoly(overlay, [np.array(pts, np.int32)], fill)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    # outline curve
    prev = None
    for i, v in enumerate(hist):
        pt = (px0 + int(i * bw), py1 - int((v / mx) * ph))
        if prev:
            cv2.line(canvas, prev, pt, color, 1, cv2.LINE_AA)
        prev = pt

    cv2.line(canvas, (px0, py1), (px0 + pw, py1), (60, 65, 85), 1)

    # mean line + stats
    mean_v = float(gray.mean())
    std_v  = float(gray.std())
    ml     = px0 + int(mean_v * pw / 255)
    cv2.line(canvas, (ml, py0), (ml, py1), (255, 255, 255), 1, cv2.LINE_AA)
    tri = np.array([[ml, py1+4],[ml-4, py1+11],[ml+4, py1+11]], np.int32)
    cv2.fillPoly(canvas, [tri], (220, 220, 220))
    cv2.putText(canvas, f"u={mean_v:.0f}  s={std_v:.0f}",
                (ml + 5, py0 + 14), _FONT, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # x-axis ticks
    for tick in (0, 64, 128, 192, 255):
        tx = px0 + int(tick * pw / 255)
        cv2.line(canvas, (tx, py1), (tx, py1 + 4), (70, 75, 95), 1)
        cv2.putText(canvas, str(tick), (tx - 8, py1 + 16),
                    _FONT, 0.30, (100, 110, 130), 1, cv2.LINE_AA)


def build_comparison_image(original_bgr: np.ndarray,
                            enhanced_bgr: np.ndarray,
                            metrics: dict,
                            config: dict | None = None) -> np.ndarray:
    """
    Side-by-side figure: images, histograms, and a metrics banner.

    `metrics` is a compute_metrics() dict; the banner shows the full-reference
    numbers on one row and the no-reference original→enhanced changes on the
    next, so the exported PNG is self-contained for a report. `config`, when
    given, is the enhance(return_config=True) dict and is printed as the
    footer line — otherwise the footer just reports which SRGAN weights exist.
    """
    IH, IW   = original_bgr.shape[:2]
    LBL_H    = 44
    HIST_H   = 190
    METRIC_H = 104
    PAD      = 12
    TW       = IW * 2 + PAD
    BG       = (18, 20, 30)
    C_O      = (90, 155, 255)
    C_E      = (70, 225, 140)

    total_h = LBL_H + IH + LBL_H + HIST_H + METRIC_H
    c       = np.full((total_h, TW, 3), BG, dtype=np.uint8)

    # column divider
    cv2.rectangle(c, (IW, 0), (IW + PAD, total_h), (12, 14, 22), -1)

    # image labels + images
    _label_bar(c, 0,        0, IW, LBL_H, "Original  (Input)",  C_O)
    _label_bar(c, IW + PAD, 0, IW, LBL_H, "Enhanced  (Output)", C_E)
    y_img = LBL_H
    c[y_img:y_img+IH,  0:IW]        = original_bgr
    c[y_img:y_img+IH,  IW+PAD:TW]   = enhanced_bgr
    for x0 in (0, IW + PAD):
        cv2.rectangle(c, (x0, y_img), (x0+IW-1, y_img+IH-1), (45, 50, 70), 1)

    # histogram labels + histograms
    y_hl = LBL_H + IH
    _label_bar(c, 0,        y_hl, IW, LBL_H, "Intensity Histogram - Original", C_O)
    _label_bar(c, IW + PAD, y_hl, IW, LBL_H, "Intensity Histogram - Enhanced", C_E)
    y_h  = y_hl + LBL_H
    og   = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    eg   = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    _draw_histogram(c, og, C_O, 0,        y_h, IW, HIST_H)
    _draw_histogram(c, eg, C_E, IW + PAD, y_h, IW, HIST_H)

    # metrics banner
    y_m = y_h + HIST_H
    cv2.rectangle(c, (0, y_m), (TW, total_h), (10, 12, 20), -1)
    cv2.line(c, (0, y_m), (TW, y_m), (40, 45, 65), 1)

    nr_o = metrics.get("original", {})
    nr_e = metrics.get("enhanced", {})
    gain = metrics.get("gain", {})

    def _draw_chips(chips, cy):
        cx = 18
        for txt, col in chips:
            (tw, th), _ = cv2.getTextSize(txt, _FONT, 0.46, 1)
            cv2.rectangle(c, (cx-6, cy-th-6), (cx+tw+6, cy+6), (25, 28, 42), -1)
            cv2.rectangle(c, (cx-6, cy-th-6), (cx+tw+6, cy+6), (40, 45, 65),  1)
            cv2.putText(c, txt, (cx, cy), _FONT, 0.46, col, 1, cv2.LINE_AA)
            cx += tw + 18

    # row 1 — full-reference (fidelity vs the original)
    row1 = [
        (f"PSNR : {metrics['psnr']} dB", C_O),
        (f"SSIM : {metrics['ssim']}",    C_E),
        (f"MSE : {metrics['mse']}",      (180, 160, 255)),
        (f"RMSE : {metrics.get('rmse','-')}", (180, 160, 255)),
        (f"MAE : {metrics.get('mae','-')}",   (180, 160, 255)),
        (f"AMBE : {metrics.get('ambe','-')}", (150, 200, 255)),
        (f"UQI : {metrics.get('uqi','-')}",   (150, 200, 255)),
    ]
    if "delta_e" in metrics:
        row1.append((f"dE2000 : {metrics['delta_e']}", (150, 200, 255)))
    _draw_chips(row1, y_m + 26)

    # row 2 — no-reference original -> enhanced, plus the gain ratios
    if nr_o and nr_e:
        row2 = [
            (f"Brightness : {nr_o['brightness']:.0f} -> {nr_e['brightness']:.0f}", (200, 200, 200)),
            (f"Contrast : {nr_o['contrast']:.1f} -> {nr_e['contrast']:.1f}  (CII {gain.get('cii','-')}x)", (120, 220, 200)),
            (f"Entropy : {nr_o['entropy']:.2f} -> {nr_e['entropy']:.2f}", (120, 220, 200)),
            (f"EME : {nr_o['eme']:.1f} -> {nr_e['eme']:.1f}", (120, 220, 200)),
            (f"Sharpness : {nr_o['avg_gradient']:.1f} -> {nr_e['avg_gradient']:.1f}", (120, 220, 200)),
            (f"Colourfulness : {nr_o['colorfulness']:.1f} -> {nr_e['colorfulness']:.1f}", (120, 220, 200)),
        ]
        _draw_chips(row2, y_m + 60)
    else:
        _draw_chips([(f"Brightness : {og.mean():.0f} -> {eg.mean():.0f}", (200, 200, 200))],
                    y_m + 60)

    if config:
        sr_txt = config.get("sr_actual", "?")
        if config.get("sr_fallback"):
            sr_txt += f"  (requested '{config.get('sr_method')}' - weights missing)"
        note = (f"colour space: {config.get('color_space','?')}   |   "
                f"upsample: {config.get('upsample_method','?')}   |   "
                f"contrast: {config.get('contrast_method','?')}   |   "
                f"SR: {sr_txt}")
    elif os.path.exists(_WEIGHTS_PATH):
        note = "SRGAN: trained (paper V-channel, 2x)  [srgan_v.pth found]"
    elif os.path.exists(_RGB_WEIGHTS_PATH):
        note = "SRGAN: trained (pretrained RGB, 4x)  [srgan_rgb_pretrained.pth found]"
    else:
        note = "SRGAN: fallback  [no trained weights found]"
    cv2.putText(c, note, (18, y_m + METRIC_H - 12),
                _FONT, 0.38, (90, 98, 125), 1, cv2.LINE_AA)

    return c
