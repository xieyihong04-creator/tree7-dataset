# 護欄一致性表（grouped test，n=217）

同一份 grouped test logits 重算；`coverage` = 信心 ≥ 門檻的比例；`selective acc` = 被保留樣本的準確率；`誤攔率` = 被擋掉的正確樣本 / 全部正確樣本。

| 模式 | 門檻 | 額外規則 | coverage | selective acc | 誤攔率 | 攔到錯誤 | 狀態 |
|---|---|---|---|---|---|---|---|
| center | 0.50 | 無 | 86.18% | 97.86% | 8.96% | 12 / 16 | 歷史參考 |
| center | 0.80 | 無 | 65.90% | 100.00% | 28.86% | 16 / 16 | 未部署 |
| TTA | 0.50 | 無 | 86.18% | 97.86% | 9.85% | 10 / 14 | 未部署 |
| TTA | 0.50 | crop top-1 一致 | 76.96% | 98.80% | 18.72% | 12 / 14 | 未部署 |
| TTA | 0.80 | 無 | 62.21% | 100.00% | 33.50% | 14 / 14 | 未部署（覆蓋率較高） |
| **TTA** | **0.85** | 無 | **53.00%** | **100.00%** | **43.35%** | **14 / 14** | **目前部署** |

## 判讀

- 部署門檻改為 **TTA + 0.85、沒有 crop 不一致聯集**；這是 grouped valid 以「covered accuracy ≥ 99% 的最低 t」選出的門檻。UI `LOW_CONFIDENCE = 0.85`、`demo/config.json` 的 `deployment.threshold` 與此表三者一致。
- 0.80 在 grouped test 的 coverage 較高（62.21%），且 selective acc 同為 100%；它沒有被選為部署門檻，因為在 grouped valid 上只有 98.6%，未達 99% 目標。這是「覆蓋率 vs 驗證目標」的取捨，不是資料洩漏。
- grouped test 全量 TTA：accuracy **93.55%**、macro F1 **0.9354**（n=217），14 個錯誤；0.85 可攔到 14/14，但會誤攔 43.35% 的正確樣本。
- 榕樹↔大葉欖仁互判：center 5 件、TTA 4 件（**observation split 下**）。不與隨機 split 的 6 件或 5 件直接相減。
- 原始 data：`guard_consistency.json`；門檻細節：`threshold_sweep.md`、`threshold_sweep.csv`；TTA 摘要：`tta_metrics.json`。
