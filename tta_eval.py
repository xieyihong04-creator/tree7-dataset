#!/usr/bin/env python3
"""Selective-prediction thresholds and multi-crop TTA evaluation.

Reads the HF dataset parquet files directly (images are embedded), runs the
deployed INT8 ONNX model on:

* the single center crop used by the demo, and
* 5 crops (center + 4 corners) of the 255-px resized image, averaging logits.

Thresholds are selected on the validation split only and then applied to test.

Usage:
    python3 tta_eval.py            # evaluate valid + test, write results/
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "data" / "parquet"
RESULTS = ROOT / "results"
MODEL = ROOT / "demo" / "model.onnx"
IMG_SIZE = 224
RESIZE = 255
CLASSES = [
    "Ficus_microcarpa",
    "Terminalia_catappa",
    "Araucaria_columnaris",
    "Dypsis_lutescens",
    "Strelitzia_reginae",
    "Drynaria_roosii",
    "Acalypha_wilkesiana",
]
TARGET_SELECTIVE_ACC = 0.99
SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

to_tensor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)


def load_split(split: str) -> list[dict]:
    path = PARQUET / f"{split}-00000-of-00001.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows:
        row["image"] = row["image"]["bytes"]
    return rows


def five_crops(img: Image.Image) -> list[Image.Image]:
    """Resize short side to 255 (torchvision Resize) then take 5 224 crops."""
    resized = transforms.Resize(RESIZE)(img)
    width, height = resized.size
    max_x = width - IMG_SIZE
    max_y = height - IMG_SIZE
    mid_x = int(round(max_x / 2))
    mid_y = int(round(max_y / 2))
    boxes = [
        (mid_x, mid_y),  # center
        (0, 0),  # top-left
        (max_x, 0),  # top-right
        (0, max_y),  # bottom-left
        (max_x, max_y),  # bottom-right
    ]
    return [resized.crop((x, y, x + IMG_SIZE, y + IMG_SIZE)) for x, y in boxes]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def evaluate_split(session: ort.InferenceSession, split: str) -> list[dict]:
    rows = load_split(split)
    records = []
    for index, row in enumerate(rows):
        with Image.open(io.BytesIO(row["image"])) as img:
            crops = five_crops(img.convert("RGB"))
        crop_logits = []
        for crop in crops:
            tensor = to_tensor(crop).unsqueeze(0).numpy()
            crop_logits.append(session.run(["logits"], {"input": tensor})[0][0].astype(np.float32))
        crop_logits = np.stack(crop_logits)  # [5, 7]
        tta_logits = crop_logits.mean(axis=0)
        records.append(
            {
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "page_url": row["page_url"],
                "source": row["source"],
                "observed_on": row["observed_on"],
                "center_logits": crop_logits[0].tolist(),
                "crop_logits": crop_logits.tolist(),
                "tta_logits": tta_logits.tolist(),
            }
        )
        if (index + 1) % 50 == 0:
            print(f"  {split}: {index + 1}/{len(rows)}", flush=True)
    return records


def metrics(records: list[dict], kind: str) -> dict:
    y_true = np.array([r["label"] for r in records])
    if kind == "center":
        logits = np.array([r["center_logits"] for r in records])
    else:
        logits = np.array([r["tta_logits"] for r in records])
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return {"y_true": y_true, "pred": pred, "conf": conf, "correct": pred == y_true}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 7) -> float:
    scores = []
    for cls in range(n_classes):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def sweep_row(m: dict, t: float) -> dict:
    covered = m["conf"] >= t
    n = int(covered.sum())
    acc = float(m["correct"][covered].mean()) if n else 0.0
    return {
        "threshold": t,
        "coverage": round(n / len(m["correct"]), 4),
        "n_covered": n,
        "n_errors": int((~m["correct"][covered]).sum()),
        "selective_accuracy": round(acc, 4),
    }


def pick_threshold(rows: list[dict], target: float = TARGET_SELECTIVE_ACC) -> float:
    eligible = [r for r in rows if r["selective_accuracy"] >= target and r["n_covered"] > 0]
    if not eligible:
        return max(rows, key=lambda r: r["selective_accuracy"])["threshold"]
    return min(eligible, key=lambda r: r["threshold"])["threshold"]


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    data = {}
    for split in ("valid", "test"):
        print(f"evaluating {split} …", flush=True)
        records = evaluate_split(session, split)
        data[split] = records
        with open(RESULTS / f"{split}_logits.json", "w", encoding="utf-8") as fh:
            json.dump(records, fh)
        print(f"  saved {len(records)} records", flush=True)

    report = {"target_selective_accuracy": TARGET_SELECTIVE_ACC, "sweep": {}, "selected": {}, "overall": {}}
    lines = [
        "# 選擇性預測門檻（valid 選、test 驗證）",
        "",
        f"規則：門檻在 valid 上選「covered accuracy ≥ {TARGET_SELECTIVE_ACC:.0%} 的最低 t」，再原封不動套到 test。",
        "信心 = softmax 最大值；TTA = center + 左上/右上/左下/右下共 5 個 224 crop 的平均 logits。",
        "",
    ]

    for kind in ("center", "tta"):
        per_split = {split: metrics(data[split], kind) for split in ("valid", "test")}
        rows_by_split = {}
        for split in ("valid", "test"):
            rows_by_split[split] = [sweep_row(per_split[split], t) for t in SWEEP]
        chosen = pick_threshold(rows_by_split["valid"])
        report["sweep"][kind] = rows_by_split
        report["selected"][kind] = chosen

        heading = "單一中心裁切（現況）" if kind == "center" else "5-crop TTA（平均 logits）"
        lines += [f"## {heading}", "", "| t | valid coverage | valid acc | test coverage | test acc |", "|---|---|---|---|---|"]
        for idx, t in enumerate(SWEEP):
            v = rows_by_split["valid"][idx]
            e = rows_by_split["test"][idx]
            mark = " ✅" if t == chosen else ""
            lines.append(
                f"| {t:.2f}{mark} | {fmt_pct(v['coverage'])} ({v['n_covered']}) | {fmt_pct(v['selective_accuracy'])} "
                f"| {fmt_pct(e['coverage'])} ({e['n_covered']}) | {fmt_pct(e['selective_accuracy'])} |"
            )
        lines += ["", f"valid 選出 t={chosen:.2f}；test 在該門檻：{fmt_pct(rows_by_split['test'][SWEEP.index(chosen)]['coverage'])} "
                  f"覆蓋率、{fmt_pct(rows_by_split['test'][SWEEP.index(chosen)]['selective_accuracy'])} 準確率。", ""]

        for split in ("valid", "test"):
            m = per_split[split]
            report["overall"][f"{split}_{kind}"] = {
                "accuracy": round(float(m["correct"].mean()), 4),
                "macro_f1": round(macro_f1(m["y_true"], m["pred"]), 4),
                "n": int(len(m["correct"])),
            }
            lines.append(
                f"- {split} 全量 {kind}：accuracy {fmt_pct(float(m['correct'].mean()))}、"
                f"macro F1 {macro_f1(m['y_true'], m['pred']):.4f}（n={len(m['correct'])}）"
            )
        lines.append("")

    # Threshold table CSV (long format)
    with open(RESULTS / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "split", "threshold", "coverage", "n_covered", "n_errors", "selective_accuracy", "selected_on_valid"])
        for kind in ("center", "tta"):
            for split in ("valid", "test"):
                for row in report["sweep"][kind][split]:
                    writer.writerow([
                        kind, split, f"{row['threshold']:.2f}", f"{row['coverage']:.4f}",
                        row["n_covered"], row["n_errors"], f"{row['selective_accuracy']:.4f}",
                        int(row["threshold"] == report["selected"][kind]),
                    ])

    # Test-side error behaviour: center errors, whether TTA fixed, crop disagreement
    test_rows = data["test"]
    center = metrics(test_rows, "center")
    tta = metrics(test_rows, "tta")
    crop_preds = np.array([[int(np.argmax(c)) for c in r["crop_logits"]] for r in test_rows])
    disagree = crop_preds.min(axis=1) != crop_preds.max(axis=1)
    tta_selected = report["selected"]["tta"]
    selected_row = next(r for r in report["sweep"]["tta"]["test"] if r["threshold"] == tta_selected)

    error_records = []
    for i, row in enumerate(test_rows):
        if center["correct"][i]:
            continue
        error_records.append(
            {
                "sample_id": row["sample_id"],
                "page_url": row["page_url"],
                "true": CLASSES[row["label"]],
                "center_pred": CLASSES[center["pred"][i]],
                "center_conf": round(float(center["conf"][i]), 4),
                "tta_pred": CLASSES[tta["pred"][i]],
                "tta_conf": round(float(tta["conf"][i]), 4),
                "tta_fixed": int(tta["correct"][i]),
                "crop_disagree": int(disagree[i]),
                "crop_preds": ";".join(CLASSES[p] for p in crop_preds[i]),
            }
        )
    with open(RESULTS / "tta_error_cases.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(error_records[0].keys()))
        writer.writeheader()
        writer.writerows(error_records)

    mutual = {}
    for kind, m in (("center", center), ("tta", tta)):
        y_true, pred = m["y_true"], m["pred"]
        mask = np.isin(y_true, [0, 1]) & np.isin(pred, [0, 1]) & (y_true != pred)
        mutual[kind] = {
            "n": int(mask.sum()),
            "terminalia_to_ficus": int(((y_true == 1) & (pred == 0)).sum()),
            "ficus_to_terminalia": int(((y_true == 0) & (pred == 1)).sum()),
        }

    warning = {
        "crop_disagreement_rate": round(float(disagree.mean()), 4),
        "errors_flagged_by_crop_disagreement": int(disagree[~tta["correct"]].sum()),
        "errors_total": int((~tta["correct"]).sum()),
        "correct_flagged_by_crop_disagreement": int(disagree[tta["correct"]].sum()),
        "correct_total": int(tta["correct"].sum()),
        "selected_threshold": tta_selected,
        "low_conf_errors_flagged": int((tta["conf"][~tta["correct"]] < tta_selected).sum()),
        "low_conf_correct_flagged": int((tta["conf"][tta["correct"]] < tta_selected).sum()),
        "test_coverage_at_selected": selected_row["coverage"],
        "test_selective_accuracy_at_selected": selected_row["selective_accuracy"],
    }

    summary = {
        "classes": CLASSES,
        "valid_selected_threshold_center": report["selected"]["center"],
        "valid_selected_threshold_tta": report["selected"]["tta"],
        "overall": report["overall"],
        "mutual_confusion_ficus_terminalia": mutual,
        "crop_warning": warning,
        "n_center_errors": len(error_records),
        "n_tta_errors": int((~tta["correct"]).sum()),
        "tta_fixed_errors": int(sum(r["tta_fixed"] for r in error_records)),
        "tta_new_errors": int((~tta["correct"] & center["correct"]).sum()),
    }
    with open(RESULTS / "tta_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    lines += [
        "## 5-crop TTA 與警告",
        "",
        f"- test 全量 accuracy：center {fmt_pct(float(center['correct'].mean()))} → TTA {fmt_pct(float(tta['correct'].mean()))}",
        f"- 15 個 center 錯誤中，TTA 修正 {summary['tta_fixed_errors']} 個；TTA 新增 {summary['tta_new_errors']} 個錯誤。",
        f"- 榕樹↔大葉欖仁互判：center {mutual['center']['n']} 件 → TTA {mutual['tta']['n']} 件"
        f"（欖仁→榕樹 {mutual['tta']['terminalia_to_ficus']}、榕樹→欖仁 {mutual['tta']['ficus_to_terminalia']}）。",
        f"- crop 間 top-1 不一致：test {fmt_pct(disagree.mean())}；TTA 錯案中被警告 {warning['errors_flagged_by_crop_disagreement']}/{warning['errors_total']}，"
        f"正確案被誤警告 {warning['correct_flagged_by_crop_disagreement']}/{warning['correct_total']}。",
        f"- 以 valid 選出的 t={tta_selected:.2f} 當低信心警告：可攔到 {warning['low_conf_errors_flagged']}/{warning['errors_total']} 個 TTA 錯誤，"
        f"同時誤攔 {warning['low_conf_correct_flagged']}/{warning['correct_total']} 個正確案例。",
        "",
    ]
    with open(RESULTS / "threshold_sweep.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
