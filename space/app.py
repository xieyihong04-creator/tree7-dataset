"""tree7 樹種辨識 demo（Hugging Face Spaces / Gradio）。

以 INT8 ONNX Runtime 在 CPU 上推論 ResNet18，並用 layer4 特徵圖與分類頭權重
產生 CAM 熱區。所有推論都在伺服器記憶體完成，不儲存使用者圖片。
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CLASSES: list[str] = CONFIG["classes"]
CLASS_ZH: list[str] = CONFIG["class_zh"]
MEAN = np.array(CONFIG["mean"], dtype=np.float32).reshape(3, 1, 1)
STD = np.array(CONFIG["std"], dtype=np.float32).reshape(3, 1, 1)
IMG_SIZE = int(CONFIG["img_size"])
FC_WEIGHT = np.array(CONFIG["fc_weight"], dtype=np.float32)
METRICS = CONFIG["metrics"]["onnx_int8"]

SESSION = ort.InferenceSession(str(ROOT / "model.onnx"), providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name

EXAMPLES = [[str(ROOT / item["file"])] for item in CONFIG["examples"]]
EXAMPLE_CAPTIONS = [
    f"{item['class_zh']} {item['class']}" for item in CONFIG["examples"]
]


def preprocess(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    image = image.convert("RGB")
    scale = 255.0 / min(image.size)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.BICUBIC,
    )
    left = (resized.width - IMG_SIZE) // 2
    top = (resized.height - IMG_SIZE) // 2
    crop = resized.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    array = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
    array = (array - MEAN) / STD
    return crop, array[None].astype(np.float32)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def cam_overlay(crop: Image.Image, features: np.ndarray, class_idx: int) -> Image.Image:
    weights = FC_WEIGHT[class_idx]
    cam = (weights[:, None, None] * features).sum(axis=0)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    heat = jet_colormap(np.asarray(cam_img, dtype=np.float32) / 255.0)
    base = np.asarray(crop, dtype=np.float32)
    overlay = np.clip(0.58 * base + 0.42 * heat.astype(np.float32), 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def predict(image: Image.Image | None):
    if image is None:
        raise gr.Error("請先上傳或拖入一張樹木照片。")
    crop, tensor = preprocess(image)
    logits, features = SESSION.run(["logits", "features"], {INPUT_NAME: tensor})
    logits = logits[0].astype(np.float64)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    order = np.argsort(-probs)[:3]
    scores = {
        f"{CLASS_ZH[i]} · {CLASSES[i]}": float(probs[i]) for i in order
    }
    best = int(order[0])
    overlay = cam_overlay(crop, features[0], best)

    top_lines = "\n".join(
        f"{rank + 1}. **{CLASS_ZH[i]}**（*{CLASSES[i]}*）— {probs[i]:.1%}"
        for rank, i in enumerate(order)
    )
    if probs[best] < 0.5:
        note = "> 信心偏低：影像可能不是這 7 種樹，或葉片／樹冠特徵不足，建議換一張較清楚的照片。"
    else:
        note = "> 熱區偏紅代表模型判讀時最依賴的位置；若熱區落在背景，請留意模型可能受到場景線索影響。"
    detail = f"### 預測結果\n{top_lines}\n\n{note}"

    return scores, overlay, crop, detail


CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto; }
#hero { padding: 18px 22px; border-radius: 16px;
        background: linear-gradient(135deg, #0f3d2e 0%, #1f6f4a 55%, #7fb069 100%);
        color: #f4fff8; margin-bottom: 8px; }
#hero h1 { margin: 0 0 6px 0; font-size: 1.7rem; letter-spacing: .5px; }
#hero p { margin: 0; opacity: .9; font-size: .95rem; }
#footnote { opacity: .72; font-size: .84rem; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="green"),
    css=CSS,
    title="tree7 樹種辨識",
) as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>tree7 · 校園樹種辨識</h1>
          <p>ResNet18 遷移學習 · 7 類常見樹種 · INT8 ONNX 即時推論 ·
             test accuracy 92.99%</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(
                type="pil",
                sources=["upload", "clipboard"],
                label="上傳樹木照片（葉片、樹冠或全株皆可）",
                height=360,
            )
            run_button = gr.Button("開始辨識", variant="primary", size="lg")
            gr.Examples(
                examples=EXAMPLES,
                inputs=image_input,
                label="或點選 test set 範例",
            )
        with gr.Column(scale=5):
            label_output = gr.Label(num_top_classes=3, label="預測機率", show_label=True)
            detail_output = gr.Markdown()

    with gr.Row():
        cam_output = gr.Image(
            type="pil", label="CAM 熱區（紅＝模型最關注處）", height=300
        )
        crop_output = gr.Image(type="pil", label="模型實際輸入（224×224 中心裁切）", height=300)

    gr.Markdown(
        """
### 模型與資料
- 架構：ResNet18（ImageNet 預訓練，先凍結骨幹 3 epochs，再全模型微調 15 epochs）
- 資料：`Xieih/tree7`，2,143 張、7 類，切分 1,714 / 215 / 214
- 測試集：accuracy **92.99%**、macro F1 **92.96%**
- 部署：INT8 靜態量化 ONNX（10.8 MiB），CPU 推論
- CAM：`layer4` 14×14 特徵圖 × 分類頭權重，上採樣後以 jet colormap 疊圖
        """
    )

    with gr.Accordion("七個類別", open=False):
        rows = "\n".join(
            f"| {i} | {cn} | *{cls}* |" for i, (cn, cls) in enumerate(zip(CLASS_ZH, CLASSES))
        )
        gr.Markdown("| # | 中文名 | 學名 |\n| -: | --- | --- |\n" + rows)

    gr.Markdown(
        "影像僅在本 Space 的記憶體中處理，不會儲存或公開。"
        "此模型為教學展示，預測結果不能取代專業植物鑑定。",
        elem_id="footnote",
    )

    run_button.click(predict, inputs=image_input, outputs=[label_output, cam_output, crop_output, detail_output])
    image_input.change(predict, inputs=image_input, outputs=[label_output, cam_output, crop_output, detail_output])


if __name__ == "__main__":
    demo.launch()
