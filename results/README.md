# 評估結果

## 隨機 split（214 張，seed 20261124）

以部署版 INT8 模型 [`demo/model.onnx`](../demo/model.onnx) 在 `Xieih/tree7` 的 test split（214 張）重新推論，前處理為 `Resize(255) → CenterCrop(224) → ToTensor → ImageNet Normalize`。

- accuracy **92.99%**、macro F1 **92.96%**，214 張中 15 張誤判
- 5-crop TTA：accuracy **94.39%**、12 張誤判；valid 94.88% → 95.81%
- 目前部署：TTA + 信心門檻 0.8（無 crop 聯集）；完整權衡見 [`guard_consistency.md`](guard_consistency.md)
- fp32 ONNX 同為 92.99% / 92.95%，與訓練報告一致
- 模型 SHA-256 與完整數字見 [`metrics_test_onnx.json`](metrics_test_onnx.json)；TTA 與門檻掃描見 [`tta_metrics.json`](tta_metrics.json)、[`threshold_sweep.md`](threshold_sweep.md)

| 檔案 | 內容 |
|---|---|
| `metrics_test_onnx.json` | fp32／INT8 的 accuracy、macro F1、per-class 指標、混淆矩陣 |
| `tta_metrics.json` | 5-crop TTA 指標、互判件數、crop 不一致警告 |
| `threshold_sweep.md` / `.csv` | 選擇性預測門檻掃描（valid 選、test 驗證） |
| `guard_consistency.md` / `.json` | {center, TTA}×{0.5, 0.8} 的 coverage／selective acc／誤攔率；含部署配置 |
| `grouped_split_stats.json` | observation-level split 統計（1,729 組、1,707／219／217） |
| `field_case/` | 兩張現場原圖的 TTA 預測與 per-crop CAM |
| `tta_error_cases.csv` | TTA 後仍錯誤的案例明細 |
| `per_class_resnet18_int8.csv` | 各類 precision／recall／F1／support |
| `predictions_resnet18_int8.csv` | 每張 test 影像的預測、信心值與是否正確 |
| `confusion_resnet18_int8.png` | 混淆矩陣 |
| `gradcam_errors_resnet18_int8.png` | 15 個誤判案例的 CAM 疊圖 |
| `gradcam_samples_resnet18_int8.png` | 每類高信心正確樣本與 CAM |

CAM 依 Demo 相同算法：`layer4` 的 7×7 特徵圖乘上分類頭權重、ReLU、正規化後以 jet 色階疊回影像。Demo 的 CAM 只反映**中心裁切**；`field_case/` 則對五個裁切分別輸出 CAM。

## observation split（217 張，只評一次）

依 `manifest.csv` 的 `page_url` 分組（1,729 組、0 跨組洩漏），重訓後的新 test 只評估一次：

| 模型 | accuracy | macro F1 | 榕樹↔欖仁互判 |
|---|---|---|---|
| 基線 CNN | 77.42% | 77.18% | 5 |
| ResNet18 | 92.63% | 92.66% | 3 |
| ResNet18 + 退化增強 | 92.17% | 92.21% | 4 |

- `observation_split/results.json` — 三個模型的完整 metrics、混淆矩陣、訓練設定
- `observation_split/metrics_test_*.json`、`per_class_*.csv`、`predictions_*.csv`
- `observation_split/confusion_*.png`、`curves_*.png`
- 退化增強在乾淨 test 上沒有超越原 ResNet18；field-set 分層評估仍待補。
