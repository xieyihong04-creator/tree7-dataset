#!/usr/bin/env python3
"""把訓練好的 ResNet18 匯出成 ONNX（含 layer4 特徵圖），並做 int8 量化。

輸出：
    demo/model_fp32.onnx     原始 fp32 模型（含 logits 與 features 兩個輸出）
    demo/model.onnx          int8 QDQ 量化模型（供瀏覽器使用）
    demo/fc_weights.json     CAM 用的分類頭權重
    demo/config.json         類別、正規化參數、範例圖與驗證指標
    demo/examples/*.jpg      每個類別一張高信心的 test 範例
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
CLASS_ZH = {
    "Ficus_microcarpa": "榕樹",
    "Terminalia_catappa": "大葉欖仁",
    "Araucaria_columnaris": "柱狀南洋杉",
    "Dypsis_lutescens": "散尾葵",
    "Strelitzia_reginae": "鶴望蘭",
    "Drynaria_roosii": "槲蕨",
    "Acalypha_wilkesiana": "紅桑",
}


class ExportModel(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        net = self.net
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        features = net.layer4(x)
        pooled = net.avgpool(features)
        logits = net.fc(torch.flatten(pooled, 1))
        return logits, features


class CalibrationReader(CalibrationDataReader):
    def __init__(self, rows: list[dict], images_dir: Path, tf, limit: int, input_name: str = "input"):
        self.samples = []
        for row in rows[:limit]:
            with Image.open(images_dir / row["rel_path"]) as img:
                tensor = tf(img.convert("RGB")).unsqueeze(0)
            self.samples.append({input_name: tensor.numpy()})
        self._iter = iter(self.samples)

    def get_next(self):
        return next(self._iter, None)


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.14)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_rows(data_root: Path, split: str) -> list[dict]:
    rows = []
    with open(data_root / "index.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == split:
                row["label"] = int(row["label"])
                rows.append(row)
    return rows


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for cls in range(n_classes):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def evaluate_onnx(session: ort.InferenceSession, rows, images_dir, tf) -> dict:
    y_true, y_pred = [], []
    for row in rows:
        with Image.open(images_dir / row["rel_path"]) as img:
            tensor = tf(img.convert("RGB")).unsqueeze(0).numpy()
        logits = session.run(["logits"], {"input": tensor})[0]
        y_true.append(row["label"])
        y_pred.append(int(logits.argmax(1)[0]))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, len(set(y_true))), 4),
    }


def evaluate_torch(model, rows, images_dir, tf) -> dict:
    y_true, y_pred = [], []
    with torch.no_grad():
        for row in rows:
            with Image.open(images_dir / row["rel_path"]) as img:
                tensor = tf(img.convert("RGB")).unsqueeze(0)
            logits = model(tensor)[0]
            y_true.append(row["label"])
            y_pred.append(int(logits.argmax(1).item()))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, 7), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/kaggle_output/tree7_outputs/models/resnet18_best.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tree7"))
    parser.add_argument("--out", type=Path, default=Path("demo"))
    parser.add_argument("--calib", type=int, default=256)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "examples").mkdir(exist_ok=True)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    classes = checkpoint["classes"]
    net = torchvision.models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    model = ExportModel(net).eval()

    tf = make_transform()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    fp32_path = args.out / "model_fp32.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["logits", "features"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"fp32 onnx: {fp32_path.stat().st_size / 2**20:.1f} MiB")

    train_rows = load_rows(args.data_root, "train")
    test_rows = load_rows(args.data_root, "test")
    images_dir = args.data_root / "images"

    quantize_static(
        str(fp32_path),
        str(args.out / "model.onnx"),
        CalibrationReader(train_rows, images_dir, tf, args.calib),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    int8_path = args.out / "model.onnx"
    print(f"int8 onnx: {int8_path.stat().st_size / 2**20:.1f} MiB")

    torch_metrics = evaluate_torch(model, test_rows, images_dir, tf)
    fp32_metrics = evaluate_onnx(ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]),
                                 test_rows, images_dir, tf)
    int8_metrics = evaluate_onnx(ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"]),
                                 test_rows, images_dir, tf)
    print("torch", torch_metrics)
    print("onnx fp32", fp32_metrics)
    print("onnx int8", int8_metrics)

    # 與 PyTorch fp32 的數值一致性
    with Image.open(images_dir / test_rows[0]["rel_path"]) as img:
        sample = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        torch_logits = model(sample)[0].numpy()
    ort_logits = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).run(
        ["logits"], {"input": sample.numpy()})[0]
    print(f"max |torch - onnx fp32| = {np.abs(torch_logits - ort_logits).max():.2e}")

    # CAM 權重：fc.weight [7, 512]、fc.bias [7]
    fc = net.fc
    payload = {
        "classes": classes,
        "class_zh": [CLASS_ZH.get(c, c) for c in classes],
        "img_size": IMG_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "fc_weight": [[round(float(v), 6) for v in row] for row in fc.weight.detach().tolist()],
        "fc_bias": [round(float(v), 6) for v in fc.bias.detach().tolist()],
        "metrics": {"torch": torch_metrics, "onnx_fp32": fp32_metrics, "onnx_int8": int8_metrics},
    }
    with open(args.out / "fc_weights.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    # 每類別挑一張預測正確且信心最高的 test 影像當範例
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    best: dict[int, tuple[float, dict]] = {}
    for row in test_rows:
        with Image.open(images_dir / row["rel_path"]) as img:
            tensor = tf(img.convert("RGB")).unsqueeze(0).numpy()
        logits = session.run(["logits"], {"input": tensor})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pred = int(probs.argmax())
        if pred == row["label"] and probs[pred] > best.get(pred, (0.0, None))[0]:
            best[pred] = (float(probs[pred]), row)

    examples = []
    for label, (prob, row) in sorted(best.items()):
        dest = args.out / "examples" / f"{classes[label]}.jpg"
        shutil.copyfile(images_dir / row["rel_path"], dest)
        examples.append({
            "file": f"examples/{classes[label]}.jpg",
            "label": label,
            "class": classes[label],
            "class_zh": CLASS_ZH.get(classes[label], classes[label]),
            "prob": round(prob, 4),
        })
    payload["examples"] = examples
    with open(args.out / "config.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    print("examples:", [(e["class"], e["prob"]) for e in examples])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
