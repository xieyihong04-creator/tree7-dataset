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

- 資料集：`Xieih/tree7`（2,143 張、7 類，1,714 / 215 / 214 切分）
- 測試集：accuracy 92.99%、macro F1 92.96%
- 模型：ResNet18 → INT8 ONNX（10.8 MiB）
- 熱區：`layer4` 14×14 特徵圖 × 分類頭權重（CAM）

此模型為教學展示，不能取代專業植物鑑定。
