"""
- Green channel + CLAHE, resize to 512x512
- BCEWithLogitsLoss (stable BCE on logits)
- Adam, lr=1e-4, batch=4, max epochs=100
- Train/Val split = 9:1
- Save checkpoint with the lowest validation loss
- Early stopping: min_delta=1e-4, patience=150

Example usage:
python train_vessel_unet.py ^
  --stare_root "D:\\dataset\\STARE" ^
  --hrf_root   "D:\\dataset\\HRF" ^
  --drive_root "D:\\dataset\\DRIVE" ^
  --chase_root "D:\\dataset\\CHASEDB1" ^
  --out_dir "outputs_vessel_unet" ^
  --seed 42
"""

from __future__ import annotations

import argparse
import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may slightly reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# UNet
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec4 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        xb = self.bottleneck(self.pool(x4))

        y1 = self.up1(xb)
        y1 = self.dec1(torch.cat([x4, y1], dim=1))

        y2 = self.up2(y1)
        y2 = self.dec2(torch.cat([x3, y2], dim=1))

        y3 = self.up3(y2)
        y3 = self.dec3(torch.cat([x2, y3], dim=1))

        y4 = self.up4(y3)
        y4 = self.dec4(torch.cat([x1, y4], dim=1))

        return self.out_conv(y4)  # logits


# -----------------------------
# Preprocess: green + CLAHE + resize(512)
# -----------------------------
def read_image_any(path: str) -> np.ndarray:
    """Read image robustly with PIL then convert to numpy.
    Return BGR if possible for consistency with OpenCV operations."""
    img = Image.open(path)
    # Some .gif are palette images; convert to RGB first
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        # grayscale
        return arr
    # RGB -> BGR
    return arr[:, :, ::-1].copy()


def green_clahe_resize(img_bgr_or_gray: np.ndarray, out_size: int = 512) -> np.ndarray:
    """Return float32 image in [0,1] with shape (1, out_size, out_size)."""
    if img_bgr_or_gray.ndim == 2:
        gray = img_bgr_or_gray
        g = gray
    else:
        # BGR
        g = img_bgr_or_gray[:, :, 1]

    # CLAHE on green channel
    g_u8 = g.astype(np.uint8) if g.dtype != np.uint8 else g
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_eq = clahe.apply(g_u8)

    # Resize
    g_eq = cv2.resize(g_eq, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0,1]
    g_f = g_eq.astype(np.float32) / 255.0
    return g_f[None, ...]  # (1,H,W)


def read_label_binary(path: str, out_size: int = 512) -> np.ndarray:
    """Return float32 binary label in {0,1} with shape (1,out_size,out_size)."""
    lab = Image.open(path)
    lab = lab.convert("L")
    lab = np.array(lab)

    # Binary: vessels=1, background=0
    lab = (lab > 0).astype(np.uint8)

    lab = cv2.resize(lab, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return lab.astype(np.float32)[None, ...]



def collect_stare(stare_root: str) -> List[Tuple[str, str]]:
    # Expected:
    #   stare_root/stare-images/imXXXX.ppm
    #   stare_root/labels-ah/imXXXX.ah.ppm
    img_dir = os.path.join(stare_root, "stare-images")
    lab_dir = os.path.join(stare_root, "labels-ah")
    if not (os.path.isdir(img_dir) and os.path.isdir(lab_dir)):
        return []

    pairs = []
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(".ppm"):
            continue
        m = re.match(r"(im\d{4})\.ppm$", fn, flags=re.IGNORECASE)
        if not m:
            continue
        stem = m.group(1)
        img_path = os.path.join(img_dir, f"{stem}.ppm")
        lab_path = os.path.join(lab_dir, f"{stem}.ah.ppm")
        if os.path.exists(img_path) and os.path.exists(lab_path):
            pairs.append((img_path, lab_path))
    return pairs


def collect_hrf(hrf_root: str) -> List[Tuple[str, str]]:
    # Expected:
    #   hrf_root/diabetic_retinopathy/*.jpg
    #   hrf_root/diabetic_retinopathy_manualsegm/*.tif
    cats = ["diabetic_retinopathy", "glaucoma", "healthy"]
    pairs = []
    for c in cats:
        img_dir = os.path.join(hrf_root, c)
        lab_dir = os.path.join(hrf_root, f"{c}_manualsegm")
        if not (os.path.isdir(img_dir) and os.path.isdir(lab_dir)):
            continue
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue
            stem = os.path.splitext(fn)[0]
            img_path = os.path.join(img_dir, fn)
            # HRF manual seg labels are often .tif with same stem
            lab_path = os.path.join(lab_dir, f"{stem}.tif")
            if os.path.exists(img_path) and os.path.exists(lab_path):
                pairs.append((img_path, lab_path))
    return pairs


def collect_drive(drive_root: str) -> List[Tuple[str, str]]:
    # Canonical DRIVE structure (common):
    #   drive_root/training/images/*.tif
    #   drive_root/training/1st_manual/*_manual1.gif

    candidates = []

    # Variant A
    img_dir_a = os.path.join(drive_root, "training", "images")
    lab_dir_a = os.path.join(drive_root, "training", "1st_manual")
    if os.path.isdir(img_dir_a) and os.path.isdir(lab_dir_a):
        for fn in os.listdir(img_dir_a):
            if fn.lower().endswith(".tif"):
                stem = os.path.splitext(fn)[0]
                # Typical label: <stem>_manual1.gif
                lab_path = os.path.join(lab_dir_a, f"{stem}_manual1.gif")
                img_path = os.path.join(img_dir_a, fn)
                if os.path.exists(lab_path):
                    candidates.append((img_path, lab_path))

    # Variant B (flat images/1st_manual)
    img_dir_b = os.path.join(drive_root, "images")
    lab_dir_b = os.path.join(drive_root, "1st_manual")
    if os.path.isdir(img_dir_b) and os.path.isdir(lab_dir_b):
        for fn in os.listdir(img_dir_b):
            if fn.lower().endswith(".tif"):
                stem = os.path.splitext(fn)[0]
                lab_path = os.path.join(lab_dir_b, f"{stem}_manual1.gif")
                img_path = os.path.join(img_dir_b, fn)
                if os.path.exists(lab_path):
                    candidates.append((img_path, lab_path))

    # Deduplicate
    uniq = []
    seen = set()
    for p in candidates:
        key = (os.path.abspath(p[0]), os.path.abspath(p[1]))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def collect_chase(chase_root: str) -> List[Tuple[str, str]]:
    pairs = []
    img_dirs = [chase_root, os.path.join(chase_root, "images")]
    for img_dir in img_dirs:
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue
            stem = os.path.splitext(fn)[0]
            img_path = os.path.join(img_dir, fn)

            lab_path1 = os.path.join(chase_root, f"{stem}_2ndHO.png")
            lab_path2 = os.path.join(chase_root, "labels", f"{stem}_2ndHO.png")

            lab_path = lab_path1 if os.path.exists(lab_path1) else lab_path2
            if os.path.exists(lab_path):
                pairs.append((img_path, lab_path))
    # Deduplicate
    uniq = []
    seen = set()
    for p in pairs:
        key = (os.path.abspath(p[0]), os.path.abspath(p[1]))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


class VesselSegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], out_size: int = 512):
        self.pairs = pairs
        self.out_size = out_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lab_path = self.pairs[idx]

        img = read_image_any(img_path)
        x = green_clahe_resize(img, out_size=self.out_size)  # (1,H,W)

        y = read_label_binary(lab_path, out_size=self.out_size)  # (1,H,W)

        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float()
        return x_t, y_t


# -----------------------------
# Metrics / losses
# -----------------------------
@torch.no_grad()
def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


@dataclass
class EarlyStopper:
    patience: int
    min_delta: float
    best: float = float("inf")
    bad_epochs: int = 0

    def step(self, val_loss: float) -> bool:
        """Return True if should stop."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# -----------------------------
# Train / Eval loops
# -----------------------------
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_dice += float(dice_from_logits(logits.detach(), y.detach())) * bs
        n += bs

    return total_loss / max(n, 1), total_dice / max(n, 1)


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_loss: float, seed: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "seed": seed,
        },
        path,
    )


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Dataset roots (optional, but at least one should be provided)
    p.add_argument("--stare_root", type=str, default="", help="Root folder of STARE dataset")
    p.add_argument("--hrf_root", type=str, default="", help="Root folder of HRF dataset")
    p.add_argument("--drive_root", type=str, default="", help="Root folder of DRIVE dataset")
    p.add_argument("--chase_root", type=str, default="", help="Root folder of CHASE_DB1 dataset")

    # Hyperparams (aligned to paper)
    p.add_argument("--out_size", type=int, default=512, help="Resize to out_size x out_size (paper: 512)")
    p.add_argument("--batch_size", type=int, default=4, help="Paper: 4")
    p.add_argument("--epochs", type=int, default=100, help="Paper: 100")
    p.add_argument("--lr", type=float, default=1e-4, help="Paper: 1e-4")
    p.add_argument("--val_split", type=float, default=0.1, help="Paper: 9:1 => 0.1 validation")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Early stopping (paper)
    p.add_argument("--es_min_delta", type=float, default=1e-4, help="Paper: 1e-4")
    p.add_argument("--es_patience", type=int, default=150, help="Paper: 150 epochs")

    # Outputs
    p.add_argument("--out_dir", type=str, default="outputs_vessel_unet")
    p.add_argument("--save_name", type=str, default="best_vessel_unet.pth")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # Collect pairs
    all_pairs: List[Tuple[str, str]] = []

    if args.stare_root:
        pairs = collect_stare(args.stare_root)
        print(f"[Info] STARE pairs: {len(pairs)}")
        all_pairs += pairs

    if args.hrf_root:
        pairs = collect_hrf(args.hrf_root)
        print(f"[Info] HRF pairs: {len(pairs)}")
        all_pairs += pairs

    if args.drive_root:
        pairs = collect_drive(args.drive_root)
        print(f"[Info] DRIVE pairs: {len(pairs)}")
        all_pairs += pairs

    if args.chase_root:
        pairs = collect_chase(args.chase_root)
        print(f"[Info] CHASE_DB1 pairs: {len(pairs)}")
        all_pairs += pairs

    if len(all_pairs) == 0:
        raise RuntimeError(
            "No training pairs found. Please provide at least one dataset root "
            "(--stare_root/--hrf_root/--drive_root/--chase_root) with expected structure."
        )

    # Dataset
    dataset = VesselSegDataset(all_pairs, out_size=args.out_size)
    n_total = len(dataset)
    n_val = int(round(args.val_split * n_total))
    n_train = n_total - n_val
    n_val = max(n_val, 1)
    n_train = max(n_train, 1)

    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
    print(f"[Info] Split train/val = {len(train_set)}/{len(val_set)} (seed={args.seed})")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model / loss / optim
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping + best ckpt
    early = EarlyStopper(patience=args.es_patience, min_delta=args.es_min_delta)
    best_val = float("inf")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, args.save_name)

    print(f"[Info] Start training: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = run_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_dice = run_one_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.6f}, dice {tr_dice:.4f} | "
            f"val loss {va_loss:.6f}, dice {va_dice:.4f}"
        )

        # Save best val checkpoint
        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_val, args.seed)
            print(f"  [Save] best checkpoint updated: val_loss={best_val:.6f} -> {best_path}")

        # Early stopping check (paper setting)
        if early.step(va_loss):
            print(
                f"[EarlyStop] Stop at epoch {epoch} (best={early.best:.6f}, "
                f"min_delta={args.es_min_delta}, patience={args.es_patience})"
            )
            break

    print(f"[Done] Best val loss: {best_val:.6f}")
    print(f"[Done] Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
