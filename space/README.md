---
title: tree7 樹種辨識
emoji: 🌳
colorFrom: green
colorTo: emerald
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# tree7 · 校園樹種辨識

以 ResNet18 遷移學習辨識 7 種常見樹種，上傳照片即時取得 Top-3 預測與 CAM 熱區。

- 資料集：`Xieih/tree7`（2,143 張、7 類，1,714 / 215 / 214 切分）
- 測試集：accuracy 92.99%、macro F1 92.96%
- 模型：ResNet18 → INT8 ONNX（10.8 MiB），CPU 推論
- 熱區：`layer4` 特徵圖 × 分類頭權重（CAM）

影像僅在執行期間的記憶體中處理，不儲存、不公開。此模型為教學展示，
不能取代專業植物鑑定。
