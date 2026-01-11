"""
- Input: green channel + CLAHE
- Train resize: 512x512
- Heatmap: sigma=1.5, effective support ~3*sigma
- Optimizer: Adam, batch=1, max_epochs=1000
- Split: 8:2 (train:val)
- Early stopping: min_delta=1e-4, patience=50 epochs
- Inference postprocess: T=0.05, Amin=5, weighted centroid per component
"""

from __future__ import annotations
import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def seed_worker(worker_id: int) -> None:
    # make dataloader workers deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------ Config ------------------------------
@dataclass
class Config:
    image_dir: str
    annotation_dir: str
    output_dir: str = "./outputs_bifurcation"

    # training
    train_size: Tuple[int, int] = (512, 512)  # (H, W)
    batch_size: int = 1
    lr: float = 1e-4
    max_epochs: int = 1000
    val_split: float = 0.2  # 8:2 split
    seed: int = 42

    # heatmap
    sigma: float = 1.5

    # early stopping (paper: 1e-4 for 50 epochs)
    early_min_delta: float = 1e-4
    early_patience: int = 50

    # inference post-process (paper default)
    infer_resolution: Tuple[int, int] = (576, 720)  # (H, W) = 720x576 in paper -> careful order
    thresh_T: float = 0.05
    area_Amin: int = 5

    # CLAHE (paper does not fix params; keep explicit for reproducibility)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ Preprocess ------------------------------
def preprocess_green_clahe(bgr: np.ndarray, clip_limit: float, tile_grid: Tuple[int, int]) -> np.ndarray:
    """
    green channel + CLAHE

    """
    g = bgr[:, :, 1]  # green channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    g = clahe.apply(g)
    g = g.astype(np.float32) / 255.0
    return g


# ------------------------------ Heatmap ------------------------------
def generate_gaussian_heatmap(
    img_size_hw: Tuple[int, int],
    points_xy: List[Tuple[float, float]],
    sigma: float
) -> np.ndarray:
    """
    img_size_hw: (H, W)
    points_xy: list of (x, y)
    """
    H, W = img_size_hw
    heatmap = np.zeros((H, W), dtype=np.float32)
    radius = int(3 * sigma)  #  support ~3*sigma

    for x, y in points_xy:
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 0 or x_i >= W or y_i < 0 or y_i >= H:
            continue

        x_min = max(0, x_i - radius)
        x_max = min(W, x_i + radius + 1)
        y_min = max(0, y_i - radius)
        y_max = min(H, y_i + radius + 1)

        xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        gaussian = np.exp(-((xx - x_i) ** 2 + (yy - y_i) ** 2) / (2 * sigma ** 2)).astype(np.float32)

        # sum supervision (paper: "summing per-point responses")
        heatmap[y_min:y_max, x_min:x_max] += gaussian

    return heatmap


# ------------------------------ LabelMe loader ------------------------------
def load_labelme_points(json_path: str, accepted_labels: Optional[List[str]] = None) -> List[Tuple[float, float]]:
    """
    Load labelme points. If accepted_labels is None, accept all point shapes.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts: List[Tuple[float, float]] = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        label = shape.get("label", "")
        if accepted_labels is not None and label not in accepted_labels:
            continue
        x, y = shape["points"][0]
        pts.append((float(x), float(y)))
    return pts


# ------------------------------ Dataset ------------------------------
class BifurcationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        ann_paths: List[str],
        train_size_hw: Tuple[int, int],
        sigma: float,
        clahe_clip: float,
        clahe_grid: Tuple[int, int],
        accepted_labels: Optional[List[str]] = None,
    ):
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.train_size_hw = train_size_hw
        self.sigma = sigma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.accepted_labels = accepted_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        ann_path = self.ann_paths[idx]

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        H0, W0 = bgr.shape[:2]
        g = preprocess_green_clahe(bgr, self.clahe_clip, self.clahe_grid)  # (H0,W0) float32 [0,1]

        # load labelme points (x,y) in original coordinates
        pts = load_labelme_points(ann_path, accepted_labels=self.accepted_labels)

        # resize
        Ht, Wt = self.train_size_hw
        g_rs = cv2.resize(g, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

        # scale points
        sx = Wt / float(W0)
        sy = Ht / float(H0)
        pts_rs = [(x * sx, y * sy) for x, y in pts]

        heatmap = generate_gaussian_heatmap((Ht, Wt), pts_rs, sigma=self.sigma)

        # tensor
        img_t = torch.from_numpy(g_rs[None, ...]).float()          # (1,H,W)
        hm_t = torch.from_numpy(heatmap[None, ...]).float()        # (1,H,W)
        return img_t, hm_t


# ------------------------------ UNet ------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # heatmap regression; paper uses MSE to GT heatmap; keep sigmoid to output [0,1]
        return torch.sigmoid(self.out_conv(d1))


# ------------------------------ Train / Val ------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for img, hm in tqdm(loader, desc="Train", leave=False):
        img = img.to(device)
        hm = hm.to(device)

        pred = model(img)
        loss = criterion(pred, hm)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for img, hm in tqdm(loader, desc="Val", leave=False):
        img = img.to(device)
        hm = hm.to(device)
        pred = model(img)
        loss = criterion(pred, hm)
        total += float(loss.item())
    return total / max(1, len(loader))


# ------------------------------ Inference post-process (paper) ------------------------------
def heatmap_to_keypoints_weighted_cc(
    heatmap: np.ndarray,  # (H,W) float in [0,1]
    T: float,
    Amin: int
) -> List[Tuple[float, float]]:
    """
    Paper: binarize with T, remove components < Amin,
    then place keypoint at heatmap-weighted centroid per component.
    """
    bin_map = (heatmap > T).astype(np.uint8)  # 0/1
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, connectivity=8)

    kpts: List[Tuple[float, float]] = []
    for k in range(1, num_labels):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < Amin:
            continue

        ys, xs = np.where(labels == k)
        w = heatmap[ys, xs].astype(np.float64)
        wsum = float(w.sum())
        if wsum <= 1e-12:
            # fallback: unweighted centroid
            cx = float(xs.mean())
            cy = float(ys.mean())
        else:
            cx = float((xs * w).sum() / wsum)
            cy = float((ys * w).sum() / wsum)
        kpts.append((cx, cy))
    return kpts


@torch.no_grad()
def infer_image(
    model: nn.Module,
    image_path: str,
    cfg: Config
) -> Dict:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    g = preprocess_green_clahe(bgr, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    Ht, Wt = cfg.infer_resolution  # (H,W)
    g_rs = cv2.resize(g, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(g_rs[None, None, ...]).float().to(cfg.device)  # (1,1,H,W)
    pred = model(x).squeeze().detach().cpu().numpy()  # (H,W)

    kpts = heatmap_to_keypoints_weighted_cc(pred, T=cfg.thresh_T, Amin=cfg.area_Amin)
    return {"image": image_path, "heatmap": pred, "keypoints_xy": kpts}


# ------------------------------ Main ------------------------------
def build_pairs(image_dir: str, ann_dir: str) -> Tuple[List[str], List[str]]:
    ann_files = [f for f in os.listdir(ann_dir) if f.lower().endswith(".json")]
    image_paths, ann_paths = [], []

    for ann in ann_files:
        base = os.path.splitext(ann)[0]
        found = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            p = os.path.join(image_dir, base + ext)
            if os.path.exists(p):
                found = p
                break
        if found is not None:
            image_paths.append(found)
            ann_paths.append(os.path.join(ann_dir, ann))

    if len(image_paths) == 0:
        raise RuntimeError(f"No paired (image,json) found under: {image_dir} and {ann_dir}")
    return image_paths, ann_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_bifurcation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accepted_labels", type=str, default="")  # e.g. "bifurcation,thin_point"
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    set_seed(cfg.seed, deterministic=True)
    device = torch.device(cfg.device)

    accepted_labels = None
    if args.accepted_labels.strip():
        accepted_labels = [s.strip() for s in args.accepted_labels.split(",") if s.strip()]

    image_paths, ann_paths = build_pairs(cfg.image_dir, cfg.annotation_dir)

    train_imgs, val_imgs, train_anns, val_anns = train_test_split(
        image_paths,
        ann_paths,
        test_size=cfg.val_split,
        random_state=cfg.seed,
        shuffle=True,
    )

    train_ds = BifurcationDataset(
        train_imgs, train_anns,
        train_size_hw=cfg.train_size,
        sigma=cfg.sigma,
        clahe_clip=cfg.clahe_clip_limit,
        clahe_grid=cfg.clahe_tile_grid,
        accepted_labels=accepted_labels,
    )
    val_ds = BifurcationDataset(
        val_imgs, val_anns,
        train_size_hw=cfg.train_size,
        sigma=cfg.sigma,
        clahe_clip=cfg.clahe_clip_limit,
        clahe_grid=cfg.clahe_tile_grid,
        accepted_labels=accepted_labels,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()  # paper: MSE between predicted heatmap and GT heatmap
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    ckpt_path = os.path.join(cfg.output_dir, "best_bifurcation_unet.pth")

    print(f"[Info] Device: {device}")
    print(f"[Info] Train/Val: {len(train_ds)} / {len(val_ds)} (split=8:2)")
    print(f"[Info] Max epochs: {cfg.max_epochs}, batch={cfg.batch_size}, lr={cfg.lr}")
    print(f"[Info] Early stop: min_delta={cfg.early_min_delta}, patience={cfg.early_patience}")
    print(f"[Info] Heatmap sigma={cfg.sigma} (support~3*sigma)")
    print(f"[Info] Inference: resolution={cfg.infer_resolution[::-1]} (W,H), T={cfg.thresh_T}, Amin={cfg.area_Amin}")

    for epoch in range(cfg.max_epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = eval_one_epoch(model, val_loader, criterion, device)

        improved = (best_val - va) > cfg.early_min_delta
        if improved:
            best_val = va
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va,
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        print(f"Epoch {epoch+1:4d}/{cfg.max_epochs} | train={tr:.6f} | val={va:.6f} | best={best_val:.6f} @ {best_epoch+1}")

        if no_improve >= cfg.early_patience:
            print(f"[EarlyStop] Stop at epoch {epoch+1}. Best val={best_val:.6f} at epoch {best_epoch+1}.")
            break

    print(f"[Done] Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
