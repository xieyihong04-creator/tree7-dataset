#!/usr/bin/env python3
"""對已訓練的 baseline CNN 與 ResNet18 產生 Grad-CAM 熱力圖。

用法：
    python3 gradcam.py
    python3 gradcam.py --data-root data/tree7 --out runs/gradcam
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #


class BaselineCNN(torch.nn.Module):
    """與 train.py 相同的基線 CNN 結構。"""

    def __init__(self, num_classes: int = 7):
        super().__init__()

        def block(cin: int, cout: int):
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(cout),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(cout),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
            )

        self.features = torch.nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "baseline":
        model = BaselineCNN(num_classes)
    elif arch == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(arch)
    return model


def target_layer(model: torch.nn.Module, arch: str):
    if arch == "baseline":
        return model.features[3][3]  # 最後一個 block 的第二個 conv
    return model.layer4[-1]  # BasicBlock 輸出（ReLU 後）


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(self, tensor: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx, float(probs[0, class_idx].detach())


def overlay(orig: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * orig + alpha * heat, 0, 1)


# --------------------------------------------------------------------------- #
# 資料
# --------------------------------------------------------------------------- #


def load_test_rows(index_path: Path) -> list[dict]:
    rows = []
    with open(index_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "test":
                row["label"] = int(row["label"])
                rows.append(row)
    return rows


@torch.no_grad()
def predict_all(model, rows, images_dir, tf, batch_size: int = 32) -> np.ndarray:
    probs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        tensors = []
        for row in batch_rows:
            with Image.open(images_dir / row["rel_path"]) as img:
                tensors.append(tf(img.convert("RGB")))
        logits = model(torch.stack(tensors))
        probs.append(torch.softmax(logits, dim=1))
    return torch.cat(probs).numpy()


def load_tensor(path: Path, tf) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as img:
        tensor = tf(img.convert("RGB"))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    display = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return tensor.unsqueeze(0), display


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tree7"))
    parser.add_argument("--ckpt-dir", type=Path,
                        default=Path("runs/kaggle_output/tree7_outputs/models"))
    parser.add_argument("--out", type=Path, default=Path("runs/gradcam"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "samples").mkdir(exist_ok=True)

    tf = transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.14)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    rows = load_test_rows(args.data_root / "index.csv")
    images_dir = args.data_root / "images"
    print(f"test 影像：{len(rows)} 張", flush=True)

    models, cams, probs_by_arch, classes = {}, {}, {}, None
    for arch in ("baseline", "resnet18"):
        ckpt_path = args.ckpt_dir / f"{arch}_best.pt"
        if not ckpt_path.exists():
            print(f"找不到 {ckpt_path}，略過 {arch}")
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        classes = checkpoint["classes"]
        model = build_model(arch, len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models[arch] = model
        cams[arch] = GradCAM(model, target_layer(model, arch))
        probs_by_arch[arch] = predict_all(model, rows, images_dir, tf)
        acc = (probs_by_arch[arch].argmax(1) == np.array([r["label"] for r in rows])).mean()
        print(f"{arch}: test accuracy {acc:.4f}", flush=True)

    y_true = np.array([r["label"] for r in rows])
    picks = []
    for cls_idx in range(len(classes)):
        candidates = [
            i for i, row in enumerate(rows)
            if row["label"] == cls_idx and all(probs_by_arch[a][i].argmax() == cls_idx for a in models)
        ]
        if not candidates:
            candidates = [i for i, row in enumerate(rows) if row["label"] == cls_idx]
        best = max(candidates, key=lambda i: probs_by_arch["resnet18"][i][cls_idx])
        picks.append(best)

    fig, axes = plt.subplots(len(picks), 3, figsize=(11.5, 3.6 * len(picks)))
    for row_idx, idx in enumerate(picks):
        row = rows[idx]
        tensor, display = load_tensor(images_dir / row["rel_path"], tf)
        axes[row_idx, 0].imshow(display)
        axes[row_idx, 0].set_title(f"input\ntrue: {classes[row['label']]}", fontsize=9)
        axes[row_idx, 0].axis("off")
        for col_idx, arch in enumerate(("baseline", "resnet18"), start=1):
            cam, pred, prob = cams[arch].generate(tensor)
            axes[row_idx, col_idx].imshow(overlay(display, cam))
            axes[row_idx, col_idx].set_title(
                f"{arch}\npred: {classes[pred]} ({prob:.1%})", fontsize=9,
                color="green" if pred == row["label"] else "crimson",
            )
            axes[row_idx, col_idx].axis("off")
            Image.fromarray((overlay(display, cam) * 255).astype(np.uint8)).save(
                args.out / "samples" / f"{classes[row['label']]}_{arch}.png"
            )
    fig.suptitle("Grad-CAM: same test image, baseline CNN vs ResNet18", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(args.out / "gradcam_compare.png", dpi=150)
    plt.close(fig)

    with open(args.out / "gradcam_samples.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "file", "baseline_pred", "baseline_prob", "resnet_pred", "resnet_prob"])
        for idx in picks:
            row = rows[idx]
            writer.writerow([
                classes[row["label"]], row["rel_path"],
                classes[probs_by_arch["baseline"][idx].argmax()], f"{probs_by_arch['baseline'][idx].max():.4f}",
                classes[probs_by_arch["resnet18"][idx].argmax()], f"{probs_by_arch['resnet18'][idx].max():.4f}",
            ])

    for arch in models:
        wrong = np.where(probs_by_arch[arch].argmax(1) != y_true)[0]
        wrong = sorted(wrong, key=lambda i: probs_by_arch[arch][i].max(), reverse=True)[:12]
        if not wrong:
            continue
        ncols = 4
        nrows = (len(wrong) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.9 * nrows))
        for ax, idx in zip(np.ravel(axes), wrong):
            row = rows[idx]
            tensor, display = load_tensor(images_dir / row["rel_path"], tf)
            cam, pred, prob = cams[arch].generate(tensor)
            ax.imshow(overlay(display, cam))
            ax.set_title(f"true: {classes[row['label']]}\npred: {classes[pred]} ({prob:.1%})", fontsize=8)
            ax.axis("off")
        for ax in np.ravel(axes)[len(wrong):]:
            ax.axis("off")
        fig.suptitle(f"{arch}: Grad-CAM on misclassified test images", fontsize=12, y=0.997)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(args.out / f"gradcam_errors_{arch}.png", dpi=150)
        plt.close(fig)
        print(f"{arch}: {len(wrong)} 張錯誤範例已輸出", flush=True)

    print(f"完成，輸出於 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
