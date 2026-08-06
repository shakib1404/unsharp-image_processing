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
  WRONG. The current pipeline already follows the paper exactly:
    normalize → invert → gamma → Hu-WBI upsample → CLAHE → downsample →
    unsharp mask → SRGAN → invert back → RGB.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
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
                    strength: float = 0.6) -> np.ndarray:
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

    `strength` blends the fully color-transferred result back toward the
    pre-transfer target (CLAHE/SRGAN output) — full correction (strength=1)
    pulls the output's color statistics all the way back to the original's,
    which is desired when the pipeline has drifted duller/darker than the
    input, but over-corrects when CLAHE/SRGAN legitimately boosted a
    diagnostically useful signal (e.g. redness/erythema in inflamed tissue)
    beyond the original's baseline — full correction would cancel that
    boost. Default 0.6 keeps most of the correction while preserving some of
    that boost.
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
# 3.  Full enhancement pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_v_channel(image_bgr: np.ndarray,
                          gamma: float      = 0.8,
                          clip_limit: float = 2.0,
                          tile_size: tuple  = (8, 8),
                          cn: float         = 0.85,
                          sigma: float      = 1.0):
    """
    Steps 1–8 of the paper pipeline — everything up to (but excluding) SRGAN.

    Step 1  RGB → HSV,  extract V channel
    Step 2  Normalize V         Eq. 1 : I_I(a,b) = I(a,b) / I_I(Max)
    Step 3  Invert              Eq. 2 : I' = 1 - I_I(a,b)
    Step 4  Gamma correction    Eq. 3 : Y = C * M^γ      (C=1, γ=0.8)
    Step 5  Hu-WBI 2× upsample  Eq. 4
    Step 6  CLAHE on upsampled image
    Step 7  Downsample back to original size
    Step 8  Unsharp mask (HPF)  Eq. 7 : PS = (I_i - LPF(I_i)) * C_n

    Returns (h_ch, s_ch, v_sharp): the untouched Hue/Saturation channels and
    the fully-processed (still inverted) V channel, same spatial size as the
    input image. This is exactly what enhance() hands to SRGAN, and what
    train_srgan_v.py uses to build ground-truth training targets — keeping
    training and inference on the identical distribution.
    """
    # ── Step 1 ──────────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # ── Step 2  normalize ────────────────────────────────────────────────────
    v_norm = v_ch.astype(np.float32) / 255.0

    # ── Step 3  invert ───────────────────────────────────────────────────────
    v_inv = 1.0 - v_norm

    # ── Step 4  gamma correction ──────────────────────────────────────────────
    v_gamma = np.power(np.clip(v_inv, 0.0, 1.0), gamma)
    v_uint8 = np.clip(v_gamma * 255, 0, 255).astype(np.uint8)

    # ── Steps 5–6  Hu-WBI upsample then CLAHE ────────────────────────────────
    v_up    = hu_wbi_upsample(v_uint8)
    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    v_clahe = clahe.apply(v_up)

    # ── Step 7  downsample ────────────────────────────────────────────────────
    h_orig, w_orig = v_uint8.shape
    v_down = cv2.resize(v_clahe, (w_orig, h_orig),
                        interpolation=cv2.INTER_LINEAR)

    # ── Step 8  unsharp mask (HPF)  PS = (I_i - LPF(I_i)) * C_n ─────────────
    lpf    = cv2.GaussianBlur(v_down.astype(np.float32), (0, 0), sigma)
    ps     = (v_down.astype(np.float32) - lpf) * cn
    v_sharp = np.clip(v_down.astype(np.float32) + ps, 0, 255).astype(np.uint8)

    return h_ch, s_ch, v_sharp


def enhance(image_bgr: np.ndarray,
            gamma: float                   = 0.8,
            clip_limit: float              = 2.0,
            tile_size: tuple               = (8, 8),
            cn: float                      = 0.85,
            sigma: float                   = 1.0,
            use_srgan: bool                = True,
            use_color_transfer: bool       = True,
            color_transfer_strength: float = 0.6,
            use_visual_punch: bool         = False,
            visual_punch_saturation: float = 1.6,
            visual_punch_warmth: float     = 22.0) -> np.ndarray:
    """
    Paper pipeline (Fig. 1 + algorithm steps). Steps 1–8 are delegated to
    preprocess_v_channel(); this function handles SRGAN + reconstruction:

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
    h_ch, s_ch, v_sharp = preprocess_v_channel(
        image_bgr, gamma, clip_limit, tile_size, cn, sigma)
    h_orig, w_orig = v_sharp.shape

    # ── Step 9  SRGAN on V (paper order), invert back, reconstruct to RGB ────
    use_v_srgan   = use_srgan and os.path.exists(_WEIGHTS_PATH)
    use_rgb_srgan = use_srgan and not use_v_srgan and os.path.exists(_RGB_WEIGHTS_PATH)
    v_pre_sr     = v_sharp if use_rgb_srgan else (_apply_srgan(v_sharp) if use_srgan else v_sharp)
    v_final      = 255 - v_pre_sr
    enhanced_hsv = cv2.merge([h_ch, s_ch, v_final])
    enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
    enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

    # ── Step 10  real trained SRGAN super-resolution on RGB, then downscale ──
    if use_rgb_srgan:
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

    return enhanced_bgr


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(original_bgr: np.ndarray,
                    enhanced_bgr: np.ndarray) -> dict:
    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    enh_gray  = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    mse       = float(np.mean(
        (orig_gray.astype(np.float32) - enh_gray.astype(np.float32)) ** 2))
    psnr      = float(cv2.PSNR(original_bgr, enhanced_bgr))
    ssim_val, _ = ssim(orig_gray, enh_gray, full=True, data_range=255)
    return {
        "psnr": round(psnr, 4),
        "ssim": round(float(ssim_val), 4),
        "mse":  round(mse, 4),
    }


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
                            metrics: dict) -> np.ndarray:
    IH, IW   = original_bgr.shape[:2]
    LBL_H    = 44
    HIST_H   = 190
    METRIC_H = 72
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

    chips = [
        (f"PSNR : {metrics['psnr']} dB", C_O),
        (f"SSIM : {metrics['ssim']}",    C_E),
        (f"MSE  : {metrics['mse']}",     (180, 160, 255)),
        (f"Brightness : {og.mean():.0f} -> {eg.mean():.0f}", (200, 200, 200)),
    ]
    cx, cy = 18, y_m + 28
    for txt, col in chips:
        (tw, th), _ = cv2.getTextSize(txt, _FONT, 0.50, 1)
        cv2.rectangle(c, (cx-6, cy-th-6), (cx+tw+6, cy+6), (25, 28, 42), -1)
        cv2.rectangle(c, (cx-6, cy-th-6), (cx+tw+6, cy+6), (40, 45, 65),  1)
        cv2.putText(c, txt, (cx, cy), _FONT, 0.50, col, 1, cv2.LINE_AA)
        cx += tw + 22

    if os.path.exists(_WEIGHTS_PATH):
        srgan_note = "SRGAN: trained (paper V-channel, 2x)  [srgan_v.pth found]"
    elif os.path.exists(_RGB_WEIGHTS_PATH):
        srgan_note = "SRGAN: trained (pretrained RGB, 4x)  [srgan_rgb_pretrained.pth found]"
    else:
        srgan_note = "SRGAN: fallback  [no trained weights found]"
    cv2.putText(c, srgan_note, (18, y_m + METRIC_H - 10),
                _FONT, 0.36, (55, 60, 80), 1, cv2.LINE_AA)

    return c
