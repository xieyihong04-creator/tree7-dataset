#!/usr/bin/env python3
"""Run the deployed grouped INT8 ONNX model on field photos.

Produces, for each photo:
  * center / 5-crop TTA probabilities
  * per-crop top-1
  * per-crop CAM for Ficus_microcarpa and Terminalia_catappa
  * a composite image in results/field_case/

Usage:
    python3 field_eval.py --images /path/a.jpg /path/b.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

IMG = 224
RESIZE = 255


def five_crops(img: Image.Image) -> list[Image.Image]:
    ratio = RESIZE / min(img.size)
    resized = img.resize((round(img.width * ratio), round(img.height * ratio)), Image.BILINEAR)
    mx, my = resized.width - IMG, resized.height - IMG
    boxes = [(mx // 2, my // 2), (0, 0), (mx, 0), (0, my), (mx, my)]
    return [resized.crop((x, y, x + IMG, y + IMG)) for x, y in boxes]


def to_input(img: Image.Image, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return ((arr - mean) / std).transpose(2, 0, 1)[None]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def cam(features: np.ndarray, weights: np.ndarray, size: int = IMG) -> np.ndarray:
    heat = np.maximum((features * weights[:, None, None]).sum(axis=0), 0)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    return np.asarray(Image.fromarray(heat).resize((size, size), Image.BILINEAR))


def overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    color = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip((1 - alpha) * base + alpha * color, 0, 1)


def evaluate(path: Path, session, cfg, out_dir: Path) -> dict:
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    classes = cfg["classes"]
    zh = cfg["class_zh"]
    weights = np.array(cfg["fc_weight"], dtype=np.float32)
    ficus, terminalia = classes.index("Ficus_microcarpa"), classes.index("Terminalia_catappa")

    img = Image.open(path).convert("RGB")
    crops = five_crops(img)
    crop_logits, crop_feats = [], []
    for crop in crops:
        out = session.run(["logits", "features"], {"input": to_input(crop, mean, std)})
        crop_logits.append(out[0][0])
        crop_feats.append(out[1][0])
    crop_probs = [softmax(v) for v in crop_logits]
    tta = softmax(np.mean(crop_logits, axis=0))
    center = crop_probs[0]
    top = int(tta.argmax())

    record = {
        "file": path.name,
        "center": {classes[i]: round(float(center[i]), 4) for i in range(len(classes))},
        "tta": {classes[i]: round(float(tta[i]), 4) for i in range(len(classes))},
        "center_top": {"class": classes[int(center.argmax())], "class_zh": zh[int(center.argmax())],
                       "p": round(float(center.max()), 4)},
        "tta_top": {"class": classes[top], "class_zh": zh[top], "p": round(float(tta.max()), 4)},
        "ficus": round(float(tta[ficus]), 4),
        "terminalia": round(float(tta[terminalia]), 4),
        "per_crop": [],
    }
    for i, p in enumerate(crop_probs):
        idx = int(p.argmax())
        record["per_crop"].append({
            "crop": i,
            "top": classes[idx],
            "p": round(float(p[idx]), 4),
            "ficus": round(float(p[ficus]), 4),
            "terminalia": round(float(p[terminalia]), 4),
        })

    fig, axes = plt.subplots(3, 5, figsize=(10, 7.2), facecolor="white")
    fig.suptitle(
        f"{path.stem}  |  TTA top-1: {classes[top]} {tta[top]*100:.1f}%  |  "
        f"Ficus {tta[ficus]*100:.1f}%  ·  Terminalia {tta[terminalia]*100:.1f}%",
        fontsize=11, y=0.985,
    )
    for i, crop in enumerate(crops):
        ax = axes[0, i]
        ax.imshow(crop)
        item = record["per_crop"][i]
        ax.set_title(f"c{i} {item['top']} {item['p']*100:.0f}%", fontsize=8)
        ax.axis("off")
    for row, (cls, name) in enumerate([(ficus, "Ficus microcarpa"), (terminalia, "Terminalia catappa")], start=1):
        for i, crop in enumerate(crops):
            ax = axes[row, i]
            heat = cam(crop_feats[i], weights[cls])
            ax.imshow(overlay(crop, heat))
            ax.set_title(f"{name.split()[0]} CAM c{i}", fontsize=7)
            ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    dest = out_dir / f"{path.stem}_v2_per_crop_cam.jpg"
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    record["composite"] = dest.name
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, default=Path("demo/model.onnx"))
    parser.add_argument("--config", type=Path, default=Path("demo/config.json"))
    parser.add_argument("--out", type=Path, default=Path("results/field_case"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(args.config, encoding="utf-8"))
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    records = [evaluate(p, session, cfg, args.out) for p in args.images]
    payload = {r["file"].rsplit(".", 1)[0]: r for r in records}
    (args.out / "field_case_predictions_v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for r in records:
        print(r["file"], "center", r["center_top"], "tta", r["tta_top"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
