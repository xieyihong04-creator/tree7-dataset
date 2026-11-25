# 評估結果

以部署版 INT8 模型 [`demo/model.onnx`](../demo/model.onnx) 在 `Xieih/tree7` 的 test split（214 張）重新推論，前處理為 `Resize(255) → CenterCrop(224) → ToTensor → ImageNet Normalize`。

- accuracy **92.99%**、macro F1 **92.96%**，214 張中 15 張誤判
- fp32 ONNX 同為 92.99% / 92.95%，與訓練報告一致
- 模型 SHA-256 與完整數字見 [`metrics_test_onnx.json`](metrics_test_onnx.json)

| 檔案 | 內容 |
|---|---|
| `metrics_test_onnx.json` | fp32／INT8 的 accuracy、macro F1、per-class 指標、混淆矩陣 |
| `per_class_resnet18_int8.csv` | 各類 precision／recall／F1／support |
| `predictions_resnet18_int8.csv` | 每張 test 影像的預測、信心值與是否正確 |
| `confusion_resnet18_int8.png` | 混淆矩陣 |
| `gradcam_errors_resnet18_int8.png` | 15 個誤判案例的 CAM 疊圖 |
| `gradcam_samples_resnet18_int8.png` | 每類高信心正確樣本與 CAM |

CAM 依 Demo 相同算法：`layer4` 的 7×7 特徵圖乘上分類頭權重、ReLU、正規化後以 jet 色階疊回影像。
