#!/usr/bin/env python3
"""tree7：基線 CNN 與遷移學習（ResNet18）訓練管線。

可在 Kaggle GPU Kernel 或本機 CPU 執行：

    python3 train.py --prepare-only          # 只下載/解包 HF 資料集
    python3 train.py                         # 完整訓練（自動準備資料）
    python3 train.py --smoke                 # 小樣本快速煙霧測試

資料來源：https://huggingface.co/datasets/Xieih/tree7
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# --------------------------------------------------------------------------- #
# 常數
# --------------------------------------------------------------------------- #

HF_REPO = "Xieih/tree7"
HF_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
SPLIT_FILES = {
    "train": [f"data/train-0000{i}-of-00004.parquet" for i in range(4)],
    "valid": ["data/valid-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"],
}
EXTRA_FILES = ["splits.json"]

CLASSES = [
    "Ficus_microcarpa",
    "Terminalia_catappa",
    "Araucaria_columnaris",
    "Dypsis_lutescens",
    "Strelitzia_reginae",
    "Drynaria_roosii",
    "Acalypha_wilkesiana",
]
CLASS_ZH = {
    "Ficus_microcarpa": "榕樹",
    "Terminalia_catappa": "大葉欖仁",
    "Araucaria_columnaris": "柱狀南洋杉",
    "Dypsis_lutescens": "散尾葵",
    "Strelitzia_reginae": "鶴望蘭",
    "Drynaria_roosii": "槲蕨",
    "Acalypha_wilkesiana": "紅桑",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IS_KAGGLE = Path("/kaggle/working").is_dir()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 資料下載與解包
# --------------------------------------------------------------------------- #


def _open_url(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "tree7-train/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    t0 = time.time()
    last = t0
    done = 0
    with _open_url(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last > 10:
                speed = done / (time.time() - t0) / 2**20
                pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                log(f"    {dest.name}: {done / 2**20:7.1f} MiB {pct} {speed:6.2f} MiB/s")
                last = time.time()
    if total and done != total:
        raise RuntimeError(f"{dest.name} 下載不完整：{done}/{total}")
    tmp.replace(dest)
    return dest


def fetch_remote_files(parquet_dir: Path) -> Path:
    """依序嘗試 HF 主站與鏡像，回傳可用的 endpoint。"""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    probe = "splits.json"
    errors = []
    for endpoint in HF_ENDPOINTS:
        try:
            log(f"探測資料來源 {endpoint}")
            download_file(f"{endpoint}/datasets/{HF_REPO}/resolve/main/{probe}", parquet_dir / probe)
            log(f"使用 {endpoint}")
            return endpoint
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{endpoint}: {exc!r}")
            log(f"  失敗：{exc!r}")
    raise RuntimeError("所有 HF endpoint 都無法連線：\n" + "\n".join(errors))


def check_class_names(pf, path: Path) -> None:
    """用 parquet 內的 HF feature metadata 驗證 label 順序與 CLASSES 一致。"""
    raw = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if not raw:
        log(f"  {path.name}: 無 HF metadata，略過類別順序驗證")
        return
    try:
        features = json.loads(raw)["info"]["features"]
        label_feature = features["label"]
    except (KeyError, TypeError, json.JSONDecodeError):
        log(f"  {path.name}: metadata 格式不符，略過類別順序驗證")
        return
    names = label_feature.get("names") if isinstance(label_feature, dict) else None
    if names is None:
        return
    if list(names) != CLASSES:
        raise RuntimeError(f"類別順序不一致！parquet={names} vs CLASSES={CLASSES}")
    log(f"  {path.name}: 類別順序驗證通過")


def load_grouped_metadata(manifest_path: Path, splits_path: Path):
    """建立 (label, bytes, w, h) → manifest row 與 page_url → split 的查表。"""
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest：{manifest_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"找不到 grouped split：{splits_path}")

    manifest_by_key: dict[tuple[int, int, int, int], dict] = {}
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["class_index"]), int(row["bytes"]), int(row["w"]), int(row["h"]))
            prev = manifest_by_key.get(key)
            if prev is not None and prev["page_url"] != row["page_url"]:
                raise RuntimeError(f"manifest key 不唯一：{key} → {prev['page_url']} / {row['page_url']}")
            manifest_by_key[key] = row

    split_by_page: dict[str, str] = {}
    with open(splits_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split_by_page[row["page_url"]] = row["split"]

    log(f"grouped metadata：{len(manifest_by_key)} 個影像 key、{len(split_by_page)} 個 page_url")
    return manifest_by_key, split_by_page


def extract_parquet(parquet_dir: Path, images_dir: Path, manifest_by_key=None, split_by_page=None) -> list[dict]:
    """把 parquet 內的 JPEG bytes 寫成 images/<split>/<class>/<sha1>.jpg。

    指定 manifest_by_key 時使用 observation-level split：以
    (label, bytes, width, height) 對回 manifest 的 page_url，再查 split。
    """
    import pyarrow.parquet as pq

    grouped = manifest_by_key is not None
    rows: list[dict] = []
    missing: list[tuple] = []
    for split, files in SPLIT_FILES.items():
        for fname in files:
            path = parquet_dir / Path(fname).name
            if not path.exists():
                raise FileNotFoundError(path)
            pf = pq.ParquetFile(path)
            check_class_names(pf, path)
            n = 0
            for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
                images = batch.column(0).to_pylist()
                labels = batch.column(1).to_pylist()
                for img, label in zip(images, labels):
                    data = img["bytes"] if isinstance(img, dict) else img.as_py()["bytes"]
                    if data is None:
                        continue
                    label = int(label)
                    cls = CLASSES[label]
                    page_url = None
                    if grouped:
                        with Image.open(io.BytesIO(data)) as im:
                            key = (label, len(data), im.width, im.height)
                        meta = manifest_by_key.get(key)
                        if meta is None:
                            missing.append(key)
                            continue
                        page_url = meta["page_url"]
                        split = split_by_page.get(page_url)
                        if split is None:
                            raise RuntimeError(f"page_url 不在 grouped split：{page_url}")
                    digest = hashlib.sha1(data).hexdigest()[:16]
                    rel = Path(split) / cls / f"{digest}.jpg"
                    out = images_dir / rel
                    if not out.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(data)
                    row = {"split": split, "label": label, "class_name": cls, "rel_path": str(rel)}
                    if grouped:
                        row["page_url"] = page_url
                    rows.append(row)
                    n += 1
            log(f"  {Path(fname).name}: {n} 張")
    if missing:
        raise RuntimeError(f"{len(missing)} 張 parquet 影像無法對回 manifest，例如 {missing[:5]}")
    return rows


def prepare_dataset(
    data_root: Path,
    force: bool = False,
    split_mode: str = "parquet",
    manifest_path: Path = Path("manifest.csv"),
    splits_path: Path = Path("splits_grouped.csv"),
) -> Path:
    """確保 data_root/images 與 index.csv 存在，回傳 index.csv 路徑。"""
    index_path = data_root / "index.csv"
    images_dir = data_root / "images"
    mode_path = data_root / "dataset_mode.txt"
    if index_path.exists() and not force and mode_path.exists() and mode_path.read_text().strip() == split_mode:
        log(f"資料集已存在，略過下載：{index_path}")
        return index_path

    manifest_by_key = split_by_page = None
    if split_mode == "grouped":
        manifest_by_key, split_by_page = load_grouped_metadata(manifest_path, splits_path)

    parquet_dir = data_root / "parquet"
    endpoint = fetch_remote_files(parquet_dir)
    for fname in [f for files in SPLIT_FILES.values() for f in files] + EXTRA_FILES:
        dest = parquet_dir / Path(fname).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        log(f"下載 {fname}")
        download_file(f"{endpoint}/datasets/{HF_REPO}/resolve/main/{fname}", dest)

    log("解包影像…")
    rows = extract_parquet(parquet_dir, images_dir, manifest_by_key, split_by_page)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "label", "class_name", "rel_path"]
    if split_mode == "grouped":
        fieldnames.append("page_url")
        group_splits: dict[str, set[str]] = {}
        for row in rows:
            group_splits.setdefault(row["page_url"], set()).add(row["split"])
        leaks = [g for g, s in group_splits.items() if len(s) > 1]
        if leaks:
            raise RuntimeError(f"{len(leaks)} 個 page_url 跨 split，例如 {leaks[:3]}")
        log(f"grouped split 驗證：{len(group_splits)} 組、0 跨組洩漏")
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    mode_path.write_text(split_mode)

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["split"], {}).setdefault(row["class_name"], 0)
        counts[row["split"]][row["class_name"]] += 1
    log("資料集統計：")
    for split in ("train", "valid", "test"):
        total = sum(counts.get(split, {}).values())
        dist = ", ".join(f"{c}={counts.get(split, {}).get(c, 0)}" for c in CLASSES)
        log(f"  {split}: {total} 張（{dist}）")

    with open(data_root / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"split_mode": split_mode, "classes": CLASSES, "counts": counts}, fh, ensure_ascii=False, indent=2)
    return index_path


# --------------------------------------------------------------------------- #
# 資料集
# --------------------------------------------------------------------------- #


class TreeDataset(Dataset):
    def __init__(self, rows: list[dict], images_dir: Path, transform):
        self.rows = rows
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.images_dir / row["rel_path"]) as img:
            image = img.convert("RGB")
        return self.transform(image), row["label"]


class RandomJpeg:
    """模擬手機／通訊軟體壓縮：隨機 JPEG quality。"""

    def __init__(self, quality: tuple[int, int] = (30, 50), p: float = 0.5):
        self.quality = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        with Image.open(buf) as out:
            return out.convert("RGB")


class RandomDownscale:
    """先縮小再放大，模擬遠景或低解析度。"""

    def __init__(self, scale: tuple[float, float] = (0.35, 0.7), p: float = 0.5):
        self.scale = scale
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(*self.scale)
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def build_transforms(img_size: int, degrade: bool = False):
    pre = []
    if degrade:
        pre = [
            RandomDownscale(p=0.5),
            transforms.GaussianBlur(3, sigma=(0.5, 1.5)),
            RandomJpeg(quality=(30, 50), p=0.5),
        ]
    train_tf = transforms.Compose(
        pre
        + [
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def make_loaders(
    data_root: Path,
    index_path: Path,
    img_size: int,
    batch_size: int,
    workers: int,
    degrade: bool = False,
):
    rows_by_split: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    with open(index_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["label"] = int(row["label"])
            rows_by_split[row["split"]].append(row)

    train_tf, eval_tf = build_transforms(img_size, degrade=degrade)
    images_dir = data_root / "images"
    datasets = {
        "train": TreeDataset(rows_by_split["train"], images_dir, train_tf),
        "valid": TreeDataset(rows_by_split["valid"], images_dir, eval_tf),
        "test": TreeDataset(rows_by_split["test"], images_dir, eval_tf),
    }
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    if workers > 0:
        common.update(persistent_workers=True, prefetch_factor=2)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=False, **common),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, **common),
    }
    return loaders, datasets, rows_by_split


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """四層卷積＋GAP 的小型 CNN，從零開始訓練。"""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def make_model(name: str, num_classes: int = 7) -> nn.Module:
    if name == "baseline":
        return BaselineCNN(num_classes)
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        log("載入 ImageNet 預訓練 ResNet18 權重…")
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"未知模型：{name}")


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# 訓練 / 評估
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 32
    img_size: int = 224
    lr: float = 1e-3
    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 6
    freeze_epochs: int = 0
    seed: int = 42
    workers: int = 4
    time_budget_min: float = 600.0
    amp: bool = True


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=True):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * labels.size(0)
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
        y_true.extend(labels.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), acc, macro_f1, y_true, y_pred


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
        probs.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred, probs


def save_checkpoint(path: Path, model, tag: str, cfg: TrainConfig, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": tag,
            "state_dict": model.state_dict(),
            "classes": CLASSES,
            "img_size": cfg.img_size,
            "metrics": metrics,
        },
        path,
    )


def plot_curves(history: list[dict], tag: str, out_dir: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("loss", "Loss", ("train_loss", "valid_loss")),
        ("acc", "Accuracy", ("train_acc", "valid_acc")),
        ("f1", "Macro F1", ("valid_f1",)),
    ]
    for ax, (_, title, keys) in zip(axes, panels):
        for key in keys:
            ax.plot(epochs, [h[key] for h in history], marker="o", ms=3, label=key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{tag} training curves")
    fig.tight_layout()
    fig.savefig(out_dir / f"curves_{tag}.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true: list[int], y_pred: list[int], tag: str, out_dir: Path) -> dict:
    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, [c.replace("_", "\n") for c in CLASSES], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(labels, [c.replace("_", "\n") for c in CLASSES], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_title(f"{tag} confusion matrix (test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{tag}.png", dpi=150)
    plt.close(fig)
    return {"raw": cm.tolist(), "normalized": cm_norm.round(4).tolist()}


def plot_samples(dataset: TreeDataset, y_true, y_pred, probs, tag: str, out_dir: Path, n: int = 12) -> None:
    indices = random.sample(range(len(y_true)), min(n, len(y_true)))
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    for ax, idx in zip(np.ravel(axes), indices):
        image, _ = dataset[idx]
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ok = y_true[idx] == y_pred[idx]
        ax.set_title(f"true: {CLASSES[y_true[idx]]}\npred: {CLASSES[y_pred[idx]]} ({probs[idx][y_pred[idx]]:.0%})",
                     fontsize=8, color="green" if ok else "crimson")
        ax.axis("off")
    fig.suptitle(f"{tag} test predictions")
    fig.tight_layout()
    fig.savefig(out_dir / f"samples_{tag}.png", dpi=150)
    plt.close(fig)


def evaluate_and_save(model, tag, loader, dataset, device, cfg, out_dir: Path) -> dict:
    y_true, y_pred, probs = predict(model, loader, device, cfg.amp)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    cm = plot_confusion(y_true, y_pred, tag, out_dir)
    plot_samples(dataset, y_true, y_pred, probs, tag, out_dir)

    with open(out_dir / f"per_class_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in CLASSES:
            r = report[cls]
            writer.writerow([cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", int(r["support"])])

    with open(out_dir / f"predictions_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "true", "pred", "prob_pred", "prob_true"])
        for row, t, p, pr in zip(dataset.rows, y_true, y_pred, probs):
            writer.writerow([row["rel_path"], CLASSES[t], CLASSES[p], f"{pr[p]:.4f}", f"{pr[t]:.4f}"])

    metrics = {
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
        "confusion_matrix": cm,
    }
    with open(out_dir / f"metrics_test_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    return metrics


def fit(model, tag, loaders, device, cfg: TrainConfig, out_dir: Path) -> dict:
    train_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history: list[dict] = []
    best = {"f1": -1.0, "epoch": -1, "val_loss": float("inf")}
    stale = 0
    started = time.time()
    ckpt_path = out_dir / "models" / f"{tag}_best.pt"

    def budget_left() -> bool:
        return (time.time() - started) / 60 < cfg.time_budget_min

    def train_phase(phase_name: str, epochs: int, optimizer, scheduler) -> bool:
        nonlocal stale
        for _ in range(epochs):
            if not budget_left():
                log(f"[{tag}] 已達時間預算，停止訓練")
                return False
            epoch = len(history) + 1
            t0 = time.time()
            tr_loss, tr_acc, _, _, _ = run_epoch(
                model, loaders["train"], train_criterion, device, optimizer, scaler, cfg.amp
            )
            va_loss, va_acc, va_f1, _, _ = run_epoch(model, loaders["valid"], eval_criterion, device, amp=cfg.amp)
            lr_now = optimizer.param_groups[-1]["lr"]
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "phase": phase_name,
                    "lr": lr_now,
                    "train_loss": round(tr_loss, 4),
                    "train_acc": round(tr_acc, 4),
                    "valid_loss": round(va_loss, 4),
                    "valid_acc": round(va_acc, 4),
                    "valid_f1": round(va_f1, 4),
                    "sec": round(time.time() - t0, 1),
                }
            )
            improved = va_f1 > best["f1"] + 1e-4
            if improved:
                best.update(f1=va_f1, epoch=epoch, val_loss=va_loss)
                save_checkpoint(ckpt_path, model, tag, cfg, {"val_f1": va_f1, "val_acc": va_acc, "epoch": epoch})
                stale = 0
            else:
                stale += 1
            log(
                f"[{tag}] epoch {epoch:02d} ({phase_name}) "
                f"train {tr_loss:.3f}/{tr_acc:.3f} | valid {va_loss:.3f}/{va_acc:.3f} f1 {va_f1:.3f} "
                f"lr {lr_now:.1e} {history[-1]['sec']:.1f}s{' *' if improved else ''}"
            )
            if stale >= cfg.patience:
                log(f"[{tag}] 驗證指標連續 {stale} 個 epoch 未改善，早停")
                return False
        return True

    if cfg.freeze_epochs > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.head_lr,
                                      weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.freeze_epochs, 1))
        log(f"[{tag}] 階段一：凍結骨幹，只訓練分類頭 {cfg.freeze_epochs} epochs")
        if not train_phase("frozen", cfg.freeze_epochs, optimizer, scheduler):
            return {"history": history, "best": best, "train_time_sec": round(time.time() - started, 1)}

        stale = 0
        for param in model.parameters():
            param.requires_grad = True
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": cfg.backbone_lr},
                {"params": head, "lr": cfg.head_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
        remaining = max(cfg.epochs - cfg.freeze_epochs, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
        log(f"[{tag}] 階段二：解凍全部，骨幹 lr={cfg.backbone_lr}、頭 lr={cfg.head_lr}，{remaining} epochs")
        train_phase("finetune", remaining, optimizer, scheduler)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        log(f"[{tag}] 從零訓練 {cfg.epochs} epochs，lr={cfg.lr}")
        train_phase("scratch", cfg.epochs, optimizer, scheduler)

    return {"history": history, "best": best, "train_time_sec": round(time.time() - started, 1)}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def default_data_root() -> Path:
    if not IS_KAGGLE:
        return Path("data/tree7")
    for candidate in (Path("/kaggle/temp/tree7_data"), Path("/kaggle/working/tree7_data")):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return Path("/kaggle/working/tree7_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/tree7_outputs") if IS_KAGGLE else Path("runs"))
    parser.add_argument("--models", default="baseline,resnet18")
    parser.add_argument("--epochs", type=int, default=25, help="基線 CNN 的 epoch 數")
    parser.add_argument("--transfer-epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-budget-min", type=float, default=600.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-data", action="store_true", help="Kernel 結束後保留暫存影像")
    parser.add_argument("--smoke", action="store_true", help="小樣本、低解析度快速測試")
    parser.add_argument("--split-mode", choices=["parquet", "grouped"], default="parquet",
                        help="parquet=原始隨機 split；grouped=依 page_url 的 observation split")
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("splits_grouped.csv"))
    parser.add_argument("--degrade-epochs", type=int, default=0,
                        help="在 resnet18 之後，用退化增強 fine-tune 的 epoch 數")
    args = parser.parse_args()

    if args.smoke:
        args.out = args.out / "smoke"
        args.img_size = 96
        args.batch_size = 16
        args.workers = 0
        args.epochs = 1
        args.transfer_epochs = 1
        args.freeze_epochs = 0
        args.patience = 5
        args.degrade_epochs = 0
        log("煙霧模式：每類別最多 40 張、1 epoch、96px")

    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("使用 CPU 訓練")

    index_path = prepare_dataset(
        args.data_root,
        split_mode=args.split_mode,
        manifest_path=args.manifest,
        splits_path=args.splits,
    )
    if args.prepare_only:
        log("資料準備完成（--prepare-only）")
        return 0

    loaders, datasets, rows_by_split = make_loaders(
        args.data_root, index_path, args.img_size, args.batch_size, args.workers
    )
    if args.smoke:
        rng = random.Random(args.seed)
        for split in ("train", "valid", "test"):
            limited: dict[int, int] = {}
            subset = []
            for row in datasets[split].rows:
                label = row["label"]
                if limited.get(label, 0) < 40:
                    limited[label] = limited.get(label, 0) + 1
                    subset.append(row)
            rng.shuffle(subset)
            datasets[split].rows = subset
        log("煙霧子集：" + ", ".join(f"{s}={len(datasets[s])}" for s in datasets))
        common = dict(num_workers=0)
        loaders = {
            split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=(split == "train"), **common)
            for split in ("train", "valid", "test")
        }

    log(f"train={len(datasets['train'])} valid={len(datasets['valid'])} test={len(datasets['test'])}")
    log(f"類別分佈：{ {c: sum(1 for r in rows_by_split['train'] if r['class_name'] == c) for c in CLASSES} }")

    results: dict = {
        "dataset": {
            "repo": HF_REPO,
            "classes": CLASSES,
            "class_zh": CLASS_ZH,
            "split_mode": args.split_mode,
            "splits": {s: len(datasets[s]) for s in ("train", "valid", "test")},
            "img_size": args.img_size,
        },
        "env": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "seed": args.seed,
        },
        "models": {},
    }
    results_dir = args.out
    results_dir.mkdir(parents=True, exist_ok=True)

    model_tags = [m.strip() for m in args.models.split(",") if m.strip()]
    for tag in model_tags:
        log("=" * 70)
        log(f"訓練模型：{tag}")
        model = make_model(tag).to(device)
        total_params, _ = count_params(model)
        cfg = TrainConfig(
            epochs=args.epochs if tag == "baseline" else args.transfer_epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            head_lr=1e-3,
            backbone_lr=1e-4,
            patience=args.patience,
            freeze_epochs=args.freeze_epochs if tag == "resnet18" else 0,
            seed=args.seed,
            workers=args.workers,
            time_budget_min=args.time_budget_min,
            amp=True,
        )
        set_seed(args.seed)
        fit_result = fit(model, tag, loaders, device, cfg, results_dir)

        ckpt_path = results_dir / "models" / f"{tag}_best.pt"
        if not ckpt_path.exists():
            log(f"[{tag}] 沒有可用的 checkpoint，跳過測試評估")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        metrics = evaluate_and_save(model, tag, loaders["test"], datasets["test"], device, cfg, results_dir)

        with open(results_dir / f"history_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fit_result["history"][0].keys()))
            writer.writeheader()
            writer.writerows(fit_result["history"])
        plot_curves(fit_result["history"], tag, results_dir)

        results["models"][tag] = {
            "params": total_params,
            "best_val_f1": round(fit_result["best"]["f1"], 4),
            "best_epoch": fit_result["best"]["epoch"],
            "train_time_sec": fit_result["train_time_sec"],
            "config": {
                "epochs": cfg.epochs,
                "freeze_epochs": cfg.freeze_epochs,
                "batch_size": cfg.batch_size,
                "img_size": cfg.img_size,
                "label_smoothing": cfg.label_smoothing,
            },
            **metrics,
        }
        log(f"[{tag}] 測試 accuracy={metrics['test_accuracy']:.4f} macro_f1={metrics['test_macro_f1']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.degrade_epochs > 0 and "resnet18" in model_tags:
        base_ckpt = results_dir / "models" / "resnet18_best.pt"
        if not base_ckpt.exists():
            log("[resnet18_degrade] 找不到 resnet18_best.pt，跳過退化 fine-tune")
        else:
            log("=" * 70)
            log(f"退化增強 fine-tune：resnet18_degrade（{args.degrade_epochs} epochs）")
            model = make_model("resnet18").to(device)
            model.load_state_dict(torch.load(base_ckpt, map_location=device)["state_dict"])
            total_params, _ = count_params(model)
            degrade_loaders, _, _ = make_loaders(
                args.data_root, index_path, args.img_size, args.batch_size, args.workers, degrade=True
            )
            cfg = TrainConfig(
                epochs=args.degrade_epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                lr=2e-4,
                head_lr=2e-4,
                backbone_lr=2e-4,
                patience=min(4, args.patience),
                freeze_epochs=0,
                seed=args.seed,
                workers=args.workers,
                time_budget_min=args.time_budget_min,
                amp=True,
            )
            set_seed(args.seed)
            fit_result = fit(model, "resnet18_degrade", degrade_loaders, device, cfg, results_dir)

            ckpt_path = results_dir / "models" / "resnet18_degrade_best.pt"
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
                metrics = evaluate_and_save(
                    model, "resnet18_degrade", loaders["test"], datasets["test"], device, cfg, results_dir
                )
                with open(results_dir / "history_resnet18_degrade.csv", "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(fit_result["history"][0].keys()))
                    writer.writeheader()
                    writer.writerows(fit_result["history"])
                plot_curves(fit_result["history"], "resnet18_degrade", results_dir)
                results["models"]["resnet18_degrade"] = {
                    "params": total_params,
                    "best_val_f1": round(fit_result["best"]["f1"], 4),
                    "best_epoch": fit_result["best"]["epoch"],
                    "train_time_sec": fit_result["train_time_sec"],
                    "init_checkpoint": "resnet18_best.pt",
                    "degrade": {"downscale": [0.35, 0.7], "blur_sigma": [0.5, 1.5], "jpeg_quality": [30, 50]},
                    "config": {
                        "epochs": cfg.epochs,
                        "freeze_epochs": cfg.freeze_epochs,
                        "batch_size": cfg.batch_size,
                        "img_size": cfg.img_size,
                        "label_smoothing": cfg.label_smoothing,
                    },
                    **metrics,
                }
                log(
                    f"[resnet18_degrade] 測試 accuracy={metrics['test_accuracy']:.4f} "
                    f"macro_f1={metrics['test_macro_f1']:.4f}"
                )
            else:
                log("[resnet18_degrade] 沒有可用的 checkpoint，跳過測試評估")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if results["models"]:
        tags = list(results["models"])
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(tags))
        width = 0.36
        accs = [results["models"][t]["test_accuracy"] for t in tags]
        f1s = [results["models"][t]["test_macro_f1"] for t in tags]
        ax.bar(x - width / 2, accs, width, label="accuracy")
        ax.bar(x + width / 2, f1s, width, label="macro F1")
        for i, (a, f) in enumerate(zip(accs, f1s)):
            ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
            ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x, tags)
        ax.set_ylim(0, 1.08)
        ax.set_title("test performance")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "comparison.png", dpi=150)
        plt.close(fig)

    with open(results_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    log(f"完成，輸出目錄：{results_dir}")

    if IS_KAGGLE and not args.keep_data:
        shutil.rmtree(args.data_root, ignore_errors=True)
        log(f"已清除暫存資料：{args.data_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
