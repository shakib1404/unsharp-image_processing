"""
Train the paper's V-channel SRGAN generator (srgan_v.pth) on endoscopy images.

Reference: Jagarajan & Jayaraman (2026), Multimedia Tools and Applications 85:477
Section 3.4, Eq. 5 (degradation model) and Eq. 8 (composite loss).

WHY THIS SCRIPT EXISTS
─────────────────────────────────────────────────────────────────────────────
The paper's own SRGAN weights were never released, and no public pretrained
model both (a) operates on a single V/luminance channel and (b) was trained
on endoscopy images. This script trains one from scratch, using the exact
enhance.py generator architecture (SRGANGenerator, 1-channel in/out, 2x
PixelShuffle upscale, 16 residual blocks) so the resulting checkpoint drops
straight into enhance.py as 'srgan_v.pth' — no code changes needed there.

TRAINING RECIPE (self-supervised, since no paired LR/HR ground truth exists)
─────────────────────────────────────────────────────────────────────────────
1. Load an endoscopy RGB image, run it through preprocess_v_channel() (the
   exact same HSV->normalize->invert->gamma->CLAHE->Hu-WBI->unsharp pipeline
   enhance.py uses) to get v_sharp — this IS the input SRGAN sees at
   inference. v_sharp is treated as ground-truth "HR". Because this calls the
   shared preprocess_intensity()/preprocess_v_channel() functions rather than
   duplicating the steps, training data automatically tracks any pipeline
   fix made in enhance.py — no separate change needed here.
2. Random-crop a patch_size x patch_size patch from v_sharp  ->  HR target.
3. Synthetically degrade it (Eq. 5: blur + additive noise) and downsample by
   2x  ->  LR input. This is the standard self-supervised SR recipe: the
   network learns to invert a known degradation, which generalizes to real
   low-detail regions because CLAHE/Hu-WBI artifacts are already "baked in"
   to both LR and HR through step 1.
4. Train generator + discriminator adversarially with the paper's composite
   loss (Eq. 8): L_overall = w_hpf * L_HPF + w_adv * L_adv + w_content * L_content.

COLAB QUICK START
─────────────────────────────────────────────────────────────────────────────
    # 1. Runtime -> Change runtime type -> GPU (T4 is enough)
    !pip install torch torchvision opencv-python-headless -q

    # 2. Get endoscopy images into /content/data (any folder layout, the
    #    script walks it recursively). Sources cited by the paper:
    #      - Kvasir dataset      : https://datasets.simula.no/kvasir/
    #      - CVC-ClinicDB        : search "CVC-ClinicDB" on Kaggle
    #      - ETIS-Larib          : search "ETIS-Larib Polyp DB"
    #    Upload a zip via the Colab file browser and unzip it, e.g.:
    !unzip -q /content/kvasir-dataset.zip -d /content/data

    # 3. Upload enhance.py alongside this script (Colab: file browser, or
    #    `from google.colab import files; files.upload()`), then run:
    !python train_srgan_v.py --data_dir /content/data --epochs 100 \\
        --batch_size 16 --patch_size 96 --content_loss vgg --out srgan_v.pth

    # 4. Download the result and drop it next to enhance.py in this project:
    from google.colab import files
    files.download('srgan_v.pth')

Training ~100 epochs on a few hundred images takes roughly 1–3 hours on a
free Colab T4. Loss curves print every --log_every steps; sample comparison
PNGs are written to --sample_dir every --save_every epochs so you can watch
quality improve without waiting for the full run.
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from enhance import SRGANGenerator, preprocess_v_channel

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset — self-supervised LR/HR pairs from the paper's own V pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class EndoscopyVDataset(Dataset):
    """
    Yields (lr, hr) single-channel float32 tensors in [0, 1].
    hr: patch_size x patch_size crop of v_sharp (paper's pre-SRGAN V channel).
    lr: hr degraded (blur + noise, Eq. 5) and downsampled 2x.
    """

    def __init__(self, root_dir: str, patch_size: int = 96,
                 crops_per_image: int = 4, augment: bool = True):
        assert patch_size % 2 == 0, "patch_size must be even (2x downscale)"
        self.paths = sorted(
            p for p in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
            if p.lower().endswith(IMG_EXTS)
        )
        if not self.paths:
            raise RuntimeError(f"No images found under {root_dir!r}")
        self.patch_size = patch_size
        self.crops_per_image = crops_per_image
        self.augment = augment
        print(f"[dataset] {len(self.paths)} images found under {root_dir!r} "
              f"-> {len(self)} patches/epoch")

    def __len__(self):
        return len(self.paths) * self.crops_per_image

    def _load_v_sharp(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        ps = self.patch_size
        h, w = img.shape[:2]
        if h < ps or w < ps:
            scale = ps / min(h, w)
            img = cv2.resize(img, (max(ps, int(w * scale)), max(ps, int(h * scale))))
        _, _, v_sharp = preprocess_v_channel(img)
        return v_sharp

    def _random_crop(self, v_sharp: np.ndarray) -> np.ndarray:
        ps = self.patch_size
        h, w = v_sharp.shape
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        patch = v_sharp[y:y + ps, x:x + ps]
        if self.augment:
            if random.random() < 0.5:
                patch = np.fliplr(patch)
            if random.random() < 0.5:
                patch = np.flipud(patch)
            patch = np.rot90(patch, k=random.randint(0, 3))
        return np.ascontiguousarray(patch)

    @staticmethod
    def _degrade(hr_patch: np.ndarray) -> np.ndarray:
        """Eq. 5 style degradation: blur + additive noise, then 2x downsample."""
        blur_sigma = random.uniform(0.3, 1.2)
        blurred = cv2.GaussianBlur(hr_patch.astype(np.float32), (0, 0), blur_sigma)
        h, w = hr_patch.shape
        lr = cv2.resize(blurred, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        noise_sigma = random.uniform(0.0, 4.0)  # 0-255 scale
        lr = lr + np.random.normal(0.0, noise_sigma, lr.shape).astype(np.float32)
        return np.clip(lr, 0, 255)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        v_sharp = self._load_v_sharp(path)
        hr = self._random_crop(v_sharp).astype(np.float32)
        lr = self._degrade(hr)
        hr_t = torch.from_numpy(hr / 255.0).unsqueeze(0)
        lr_t = torch.from_numpy(lr / 255.0).unsqueeze(0)
        return lr_t, hr_t


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Discriminator — paper Section 3.4 / Fig. 4 (CNN, LeakyReLU, 8 conv layers)
# ═══════════════════════════════════════════════════════════════════════════════

def _d_block(in_ch, out_ch, stride, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                        bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    """8 conv layers, batch-norm after the first block, LeakyReLU throughout,
    global-average-pooled so it accepts any input resolution."""

    def __init__(self, in_ch: int = 1):
        super().__init__()
        chans = [64, 64, 128, 128, 256, 256, 512, 512]
        strides = [1, 2, 1, 2, 1, 2, 1, 2]
        layers = []
        c_in = in_ch
        for i, (c_out, s) in enumerate(zip(chans, strides)):
            layers += _d_block(c_in, c_out, s, use_bn=(i != 0))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),   # raw logits; use BCEWithLogitsLoss
        )

    def forward(self, x):
        f = self.features(x)
        f = self.pool(f).flatten(1)
        return self.classifier(f)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Losses — Eq. 8 : L_overall = w1*L_HPF + w2*L_adv + w3*L_content
# ═══════════════════════════════════════════════════════════════════════════════

class HPFLoss(nn.Module):
    """Sharpness/edge loss: L1 distance between high-pass-filtered fake & real,
    where HPF(x) = x - GaussianBlur(x)  (same definition used in enhance.py's
    unsharp-mask step)."""

    def __init__(self, sigma: float = 1.0, kernel_size: int = 7):
        super().__init__()
        ax = torch.arange(kernel_size) - kernel_size // 2
        g1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        g1d /= g1d.sum()
        kernel = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)
        self.pad = kernel_size // 2

    def _hpf(self, x):
        lpf = F.conv2d(x, self.kernel, padding=self.pad)
        return x - lpf

    def forward(self, fake, real):
        return F.l1_loss(self._hpf(fake), self._hpf(real))


class ContentLoss(nn.Module):
    """Either plain pixel L1 (no internet / offline-safe default) or VGG19
    perceptual loss (needs torchvision pretrained weights, better quality)."""

    def __init__(self, mode: str = "l1"):
        super().__init__()
        self.mode = mode
        if mode == "vgg":
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
            for p in vgg.parameters():
                p.requires_grad_(False)
            self.vgg = vgg
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, fake, real):
        if self.mode == "l1":
            return F.l1_loss(fake, real)
        fake3 = (fake.repeat(1, 3, 1, 1) - self.mean) / self.std
        real3 = (real.repeat(1, 3, 1, 1) - self.mean) / self.std
        return F.l1_loss(self.vgg(fake3), self.vgg(real3))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def save_sample(lr, fake, hr, path):
    def to_u8(t):
        return (t.detach().cpu().numpy()[0, 0] * 255).clip(0, 255).astype(np.uint8)
    lr_u8 = cv2.resize(to_u8(lr), (hr.shape[-1], hr.shape[-2]), interpolation=cv2.INTER_NEAREST)
    strip = np.hstack([lr_u8, to_u8(fake), to_u8(hr)])
    cv2.imwrite(path, strip)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU found — this will be very slow. "
              "Use a small --epochs/--patch_size/--limit_images for a smoke test.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = EndoscopyVDataset(args.data_dir, patch_size=args.patch_size,
                                crops_per_image=args.crops_per_image)
    if args.limit_images:
        dataset.paths = dataset.paths[:args.limit_images]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    G = SRGANGenerator(n_res=16, scale=2).to(device)
    D = Discriminator(in_ch=1).to(device)
    if args.resume and os.path.exists(args.resume):
        G.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[resume] loaded generator weights from {args.resume}")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))

    hpf_loss_fn = HPFLoss().to(device)
    content_loss_fn = ContentLoss(mode=args.content_loss).to(device)
    adv_loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(args.sample_dir, exist_ok=True)
    step = 0
    for epoch in range(1, args.epochs + 1):
        running = {"d": 0.0, "g": 0.0, "hpf": 0.0, "adv": 0.0, "content": 0.0}
        for lr_v, hr_v in loader:
            lr_v, hr_v = lr_v.to(device), hr_v.to(device)

            # ── Discriminator step ──────────────────────────────────────────
            with torch.no_grad():
                fake = G(lr_v)
            real_pred = D(hr_v)
            fake_pred = D(fake)
            d_loss = (adv_loss_fn(real_pred, torch.ones_like(real_pred)) +
                     adv_loss_fn(fake_pred, torch.zeros_like(fake_pred)))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # ── Generator step (Eq. 8) ───────────────────────────────────────
            fake = G(lr_v)
            fake_pred_g = D(fake)
            adv_loss = adv_loss_fn(fake_pred_g, torch.ones_like(fake_pred_g))
            hpf_loss = hpf_loss_fn(fake, hr_v)
            content_loss = content_loss_fn(fake, hr_v)
            g_loss = (args.w_hpf * hpf_loss +
                     args.w_adv * adv_loss +
                     args.w_content * content_loss)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            running["d"] += d_loss.item()
            running["g"] += g_loss.item()
            running["hpf"] += hpf_loss.item()
            running["adv"] += adv_loss.item()
            running["content"] += content_loss.item()
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                print(f"epoch {epoch:03d} step {step:06d}  "
                     f"D={running['d']/n:.4f}  G={running['g']/n:.4f}  "
                     f"(hpf={running['hpf']/n:.4f} adv={running['adv']/n:.4f} "
                     f"content={running['content']/n:.4f})")
                running = {k: 0.0 for k in running}

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(G.state_dict(), args.out)
            print(f"[checkpoint] saved generator -> {args.out}")
            G.eval()
            with torch.no_grad():
                sample_lr, sample_hr = dataset[0]
                sample_fake = G(sample_lr.unsqueeze(0).to(device))
            save_sample(sample_lr.unsqueeze(0), sample_fake, sample_hr.unsqueeze(0),
                       os.path.join(args.sample_dir, f"epoch_{epoch:03d}.png"))
            G.train()

    torch.save(G.state_dict(), args.out)
    print(f"[done] final generator saved -> {args.out}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                  help="folder of endoscopy RGB images (searched recursively)")
    p.add_argument("--out", default="srgan_v.pth",
                  help="output checkpoint path (drop next to enhance.py)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=96,
                  help="HR patch size (must be even; LR = patch_size/2)")
    p.add_argument("--crops_per_image", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4, help="paper Table 4 value")
    p.add_argument("--w_hpf", type=float, default=1.0)
    p.add_argument("--w_adv", type=float, default=1e-3,
                  help="kept small; standard practice for SRGAN adversarial loss")
    p.add_argument("--w_content", type=float, default=1.0)
    p.add_argument("--content_loss", choices=["l1", "vgg"], default="l1",
                  help="'vgg' needs internet to download ImageNet VGG19 weights "
                       "(fine on Colab); 'l1' works fully offline")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=5, help="epochs between checkpoints")
    p.add_argument("--sample_dir", default="srgan_train_samples")
    p.add_argument("--resume", default=None, help="path to an existing srgan_v.pth to continue training")
    p.add_argument("--limit_images", type=int, default=None,
                  help="debug: use only the first N images (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
