---
title: tree7 樹種辨識
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: cc-by-nc-4.0
---

# tree7 · 校園樹種辨識

以 ResNet18 遷移學習辨識 7 種常見樹種。上傳照片後，模型在**瀏覽器內**以
ONNX Runtime Web 完成推論，並顯示 Top-3 機率與 CAM 熱區；圖片不會上傳到伺服器。

- 線上 Demo：<https://xieih-tree7-species-classifier.static.hf.space>
- 完整專案（資料集、訓練、Grad-CAM、匯出）：<https://github.com/xieyihong04-creator/tree7-dataset>
- 資料集：`Xieih/tree7`（2,143 張、7 類）。部署模型使用 observation split：1,707 / 219 / 217（依 `page_url` 分組）；隨機 split 1,714 / 215 / 214 僅為歷史。
- 測試集（grouped，217 張，只評一次）：INT8 中心裁切 accuracy 92.63%、macro F1 92.63%；5-crop TTA **93.55%**、macro F1 93.54%
- 低信心門檻：TTA top-1 < **0.85** 時顯示補拍提示
- 模型：grouped ResNet18 → INT8 ONNX（10.8 MiB，`model.onnx?v=2`），第一次推論才下載並存入 `tree7-model-v2` 快取
- 熱區：`layer4` 7×7 特徵圖 × 分類頭權重（CAM）

此模型為教學展示，不能取代專業植物鑑定。
