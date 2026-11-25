# tree7 · 七種樹種辨識

從資料集建置、基線 CNN、ResNet18 遷移學習、Grad-CAM，到把 INT8 模型放進瀏覽器端推論的完整範例。

**[線上 Demo](https://xieih-tree7-species-classifier.static.hf.space)** ｜ **[Hugging Face Space](https://huggingface.co/spaces/Xieih/tree7-species-classifier)** ｜ **[資料集](https://huggingface.co/datasets/Xieih/tree7)**

## 成果一覽

測試集 214 張、7 類。checkpoint 依驗證集 macro F1 選取（early stopping，patience 6）。

| 模型 | 測試 accuracy | macro F1 | 參數量 | 部署檔 |
|---|---|---|---|---|
| 基線 CNN（4 卷積塊 + GAP，從零訓練） | 81.78% | 81.76% | 1.18 M | — |
| ResNet18 遷移學習（ImageNet 預訓練） | 92.99% | 92.95% | 11.18 M | 42.6 MiB fp32 ONNX |
| └ INT8 靜態量化（QDQ） | 92.99% | 92.96% | 11.18 M | 10.8 MiB int8 ONNX |

- 切分：train 1,714 / valid 215 / test 214（分層 80/10/10，seed `20261124`）。
- 量化前後幾乎無落差：torch 與 ONNX fp32 的最大 logit 差為 `1.3e-06`，INT8 與 fp32 的 accuracy 同為 92.99%。
- 完整訓練曲線、混淆矩陣、per-class 指標、預測明細會寫到 `runs/`（`curves_*.png`、`confusion_*.png`、`per_class_*.csv`、`predictions_*.csv`、`metrics_test_*.json`）。

## 評估結果

以部署版 INT8 ONNX（`demo/model.onnx`）在官方 test split 上重算，前處理為 `Resize(255) + CenterCrop(224)`，結果與訓練報告一致：accuracy 92.99%、macro F1 92.96%，214 張中 15 張誤判。完整輸出在 [`results/`](results/)：

- [metrics_test_onnx.json](results/metrics_test_onnx.json) — fp32／INT8 ONNX 的 accuracy、macro F1、混淆矩陣
- [per_class_resnet18_int8.csv](results/per_class_resnet18_int8.csv)、[predictions_resnet18_int8.csv](results/predictions_resnet18_int8.csv)
- [confusion_resnet18_int8.png](results/confusion_resnet18_int8.png)
- [gradcam_errors_resnet18_int8.png](results/gradcam_errors_resnet18_int8.png)、[gradcam_samples_resnet18_int8.png](results/gradcam_samples_resnet18_int8.png)

| 類別 | precision | recall | F1（INT8） | F1（fp32） | test 張數 |
|---|---|---|---|---|---|
| 榕樹 | 0.839 | 0.839 | 0.839 | 0.839 | 31 |
| 大葉欖仁 | 0.962 | 0.833 | 0.893 | 0.877 | 30 |
| 柱狀南洋杉 | 1.000 | 0.935 | 0.967 | 0.967 | 31 |
| 散尾葵 | 0.909 | 0.968 | 0.938 | 0.952 | 31 |
| 鶴望蘭 | 0.968 | 1.000 | 0.984 | 0.984 | 30 |
| 槲蕨 | 0.938 | 0.968 | 0.952 | 0.952 | 31 |
| 紅桑 | 0.906 | 0.967 | 0.935 | 0.935 | 30 |
| **macro avg** | 0.9315 | 0.9299 | **0.9296** | 0.9295 | 214 |

![ResNet18 INT8 混淆矩陣](results/confusion_resnet18_int8.png)

主要誤判集中在榕樹與大葉欖仁：大葉欖仁有 5 張被判成榕樹（recall 0.833 最低），榕樹則有 1 張被判成欖仁；榕樹另有 2 張被分到散尾葵。鶴望蘭 30 張全數正確。下方 CAM 誤判圖可對照模型實際關注的位置。

![ResNet18 INT8 誤判案例與 CAM](results/gradcam_errors_resnet18_int8.png)

## 線上 Demo

[https://xieih-tree7-species-classifier.static.hf.space](https://xieih-tree7-species-classifier.static.hf.space)

上傳照片後在**瀏覽器內**完成推論，圖片不會上傳到伺服器：

- Top-3 類別機率與中文名稱
- CAM 熱區（`layer4` 的 7×7 特徵圖 × 分類頭權重，jet 色階疊回原圖）
- 內建每類一張 test 範例，可直接點選試玩
- ONNX Runtime Web（wasm）搭配 `numThreads=1`，不需要 COOP/COEP 標頭，任何靜態主機都能跑

```mermaid
flowchart LR
  A["上傳照片<br/>（不離開瀏覽器）"] --> B["Resize 255<br/>CenterCrop 224"]
  B --> C["ImageNet 正規化<br/>1×3×224×224"]
  C --> D["ONNX Runtime Web<br/>ResNet18 INT8"]
  D --> E["logits → Top-3 機率"]
  D --> F["layer4 特徵<br/>1×512×7×7"]
  F --> G["fc_weight × features<br/>ReLU → jet"]
  G --> H["CAM 熱區疊圖"]
```

部署在 Hugging Face **Static Space**：免費帳號無法建立 Gradio／Docker Space（需 PRO），因此改用純靜態托管；把 `demo/` 連同 `model.onnx` 與 `vendor/ort/` 一起上傳即可。

## 專案結構

```text
train.py              # 下載/解包 HF 資料集、訓練基線 CNN 與 ResNet18、測試集評估
gradcam.py            # 兩模型 Grad-CAM 對照、錯誤案例與單張樣本輸出
export_onnx.py        # 匯出 fp32 ONNX、INT8 靜態量化、驗證、產生 demo 資產
demo/                 # 純前端 Demo（HTML/CSS/JS + ONNX Runtime Web）
  model.onnx          # INT8 權重（10.8 MiB，已內附）
  vendor/ort/         # ONNX Runtime Web 的 wasm 執行檔（已內附）
  config.json         # 類別、正規化參數、分類頭權重、驗證指標、範例圖
  examples/           # 每類一張高信心的 test 範例
results/              # 部署版 INT8 模型在 test split 的混淆矩陣、CAM 與 CSV
collect.py            # 建立 iNaturalist / Wikimedia Commons 候選索引
build.py              # 下載、縮邊、dHash 去重，寫入 images/<Label>/*.jpg
repair_manifest.py    # 以磁碟為準重建 manifest 並稽核
manifest.csv          # 2,143 張的來源、授權、座標、觀察日期、頁面連結
classes.json          # 0–6 類別順序與中英學名
species.json          # 物種設定、學名異名對照
```

`runs/` 與 `demo/model_fp32.onnx` 由腳本產生，不納入版本控制；`data/parquet/` 由 `train.py` 首次執行時自動下載；`demo/model.onnx`（int8）與 `demo/vendor/ort/` 已內附，clone 後可直接跑 Demo。

## 快速開始

```bash
pip install -r requirements.txt
pip install torch torchvision matplotlib scikit-learn pyarrow onnx onnxruntime
```

### 訓練

```bash
python3 train.py --prepare-only                                  # 只下載並解包 HF 資料集
python3 train.py --models baseline --epochs 30                   # 基線 CNN
python3 train.py --models resnet18 --transfer-epochs 15 --freeze-epochs 3
```

`train.py` 的預設值：224 px、batch 32、AdamW（weight decay `1e-4`）、label smoothing `0.1`、cosine schedule、patience 6、seed 42。GPU 上自動啟用 AMP；資料集不存在時會自動從 HF 下載（必要時改用 `hf-mirror.com`）。

遷移學習分兩段：前 3 個 epoch 凍結 backbone、只訓練分類頭（lr `1e-3`），之後解凍、backbone lr `1e-4`、分類頭 lr `1e-3`，共 15 個 epoch。基線 CNN 從零訓練約 30 個 epoch。

資料增強：`RandomResizedCrop(0.65–1.0, ratio 0.8–1.25)`、水平翻轉、旋轉 ±10°、ColorJitter（0.2/0.2/0.2/0.05）、ImageNet 正規化；驗證與測試用 resize 255 + center crop 224。

### Grad-CAM

```bash
python3 gradcam.py     # 預設讀 runs/models/<arch>_best.pt、輸出到 runs/gradcam/
```

目標層為 ResNet18 的 `layer4[-1]`、基線 CNN 的 `features[3][3]`。輸出：

- `gradcam_compare.png` — 兩模型在同一批 test 影像上的熱區對照
- `gradcam_errors_resnet18.png`、`gradcam_errors_baseline.png` — 錯誤案例的熱區，用來檢查模型究竟看了哪裡
- `samples/<class>_<arch>.png` — 每類單張大圖
- `gradcam_samples.csv` — 樣本、預測、機率與是否正確

熱圖只用來做模型行為檢查，不構成專業植物鑑定的依據。

### ONNX 匯出與量化

```bash
python3 export_onnx.py --ckpt runs/models/resnet18_best.pt --data-root data/tree7
```

匯出的 wrapper 同時輸出 `logits`（分類）與 `features`（`layer4` 的 `[1, 512, 7, 7]` 特徵圖），讓前端不需另外跑一次模型即可畫 CAM。腳本會：

1. 匯出 fp32 ONNX（opset 17），驗證 torch／ONNX fp32 的 logit 一致性；
2. 用 256 張 train 影像做 static QDQ INT8 量化；
3. 在 test 集上比對 torch、fp32、INT8 三者的 accuracy / macro F1；
4. 寫出 `demo/model_fp32.onnx`、`demo/model.onnx`、`demo/fc_weights.json`、`demo/config.json` 與 7 張 `demo/examples/*.jpg`。

### 本機跑 Demo

```bash
python3 -m http.server 8000 --directory demo
```

開啟 `http://localhost:8000`。`demo/model.onnx` 與 ORT wasm 已內附，不需額外下載；要重新產生模型與資產時再跑 `python3 export_onnx.py --ckpt runs/models/resnet18_best.pt`。

## 資料集

影像放在 Hugging Face：[`datasets/Xieih/tree7`](https://huggingface.co/datasets/Xieih/tree7) — 2,143 張打包成 parquet（影像嵌在欄位內，附 train/valid/test 切分），不用先抓 500 MB 就能串流訓練。本倉庫以程式、manifest、文件與可直接執行的 Demo 為主；parquet 由 `train.py` 首次執行時自動下載，CC BY-NC 影像公開散布需連署名一併處理。照下面的指令也能在本機重新抓出一份相同結構的資料集；`manifest.csv` 收錄全部 2,143 張的來源連結與授權，可逐張回溯。

| class | 資料夾 | 中文 | 學名 | 張數 |
|---|---|---|---|---|
| 0 | `Ficus_microcarpa` | 榕樹 | *Ficus microcarpa* | 308 |
| 1 | `Terminalia_catappa` | 大葉欖仁（欖仁樹） | *Terminalia catappa* | 307 |
| 2 | `Araucaria_columnaris` | 柱狀南洋杉 | *Araucaria columnaris* | 308 |
| 3 | `Dypsis_lutescens` | 散尾葵（黃椰子） | *Dypsis lutescens* | 303 |
| 4 | `Strelitzia_reginae` | 鶴望蘭（天堂鳥） | *Strelitzia reginae* | 302 |
| 5 | `Drynaria_roosii` | 槲蕨 | *Drynaria roosii* | 309 |
| 6 | `Acalypha_wilkesiana` | 紅桑（威氏鐵莧） | *Acalypha wilkesiana* | 306 |

共 2,143 張、511 MB、最長邊 1024 px、JPEG q88。

### A. 直接載入（推薦）

```python
from datasets import load_dataset

ds = load_dataset("Xieih/tree7")   # train 1714 / valid 215 / test 214
ds["train"][0]["image"]                                 # PIL.Image，存取時才解碼
ds["train"].features["label"].names                    # 0..6 對應上方類別表
```

兩邊倉庫都是 **public**，不需要 token。`label` 是 `ClassLabel`、`image` 是 `Image()` 型別，兩者都由 parquet 內的 Hugging Face feature 簽名決定，不需要載入腳本。切分為分層 80/10/10（seed `20261124`），每個 split 七類皆均衡；`splits.json` 記錄於 HF 倉庫。

### B. 在本機重建資料集

```bash
pip install -r requirements.txt
python3 collect.py --per-source 900   # 建候選索引（只存 metadata，不下載）
python3 build.py --per-label 300      # 下載、縮邊、去重，寫 images/<Label>/*.jpg
python3 repair_manifest.py            # 以磁盤為準重建清單並稽核
```

`build.py` 是增量的：把 `--per-label` 調高只會補抓差額，已存在的影像與 dHash 會被重用。跑完後 `build_parquet.py`（存放在 HF 倉庫根目錄）可把 `images/` ＋ `manifest.jsonl` 打成與 A 相同的 parquet 結構。

### 資料來源與篩選條件

- **iNaturalist**（2,016 張，94.1%）：只取 `quality_grade=research`，即至少兩位專家同意鑑定，標籤可靠度高。抓取順序為「收藏數高→最新」，先取得較完整的株型照。原圖尺寸用 API 回傳的 `original_dimensions` 預篩（長邊 <900 px 排除），不會浪費頻寬下載縮圖。
- **Wikimedia Commons**（127 張，5.9%）：依物種分類頁抓取，標題黑名單排除地圖、插圖、標本掃描、書頁（BHL／CHS）、海報春聯等；另有「標題屬名 ≠ 目標屬名即排除」的檢查（含學名異名對照，見 `species.json` 的 `syn_genera`），用來攔住混入別種分類頁的檔案。
- **GBIF 未納入**：其 media 多為標本掃描與文獻縮圖，對活體植株辨識價值低。

去重採 64-bit dHash、漢明距離 ≤6；跨類比對目前為 0 組近似重複，代表沒有同一張照片被掛到兩個物種。

### manifest.csv 欄位

`file`, `label`, `class_index`, `zh_name`, `sci_name`, `source`, `image_id`, `w`, `h`, `orig_w`, `orig_h`, `bytes`, `license`, `commercial_ok`, `attribution`, `grade`, `faves`, `latlng`, `observed_on`, `page_url`, `image_url`

訓練時較有用的幾個：

- `page_url` — 原始觀察／檔案頁，可回溯驗證標籤
- `grade`／`faves` — 鑑定等級與人氣，可做加權或再過濾
- `latlng`／`observed_on` — 要按地理或季節切 train/val 時用得到
- `commercial_ok` — 見下方授權說明

資料夾即標籤，`torchvision.datasets.ImageFolder`、`tf.data.Dataset.list_files` 或 HuggingFace `datasets` 都能直接載入；`classes.json` 提供 0–6 的類別順序。

## 授權

影像授權逐張不同，`manifest.csv` 已如實記錄：

| 授權 | 張數 |
|---|---|
| CC BY-NC | 1,879 |
| CC BY-SA（含各版本） | 82 |
| CC0 | 81 |
| CC BY（含各版本） | 76 |
| CC BY-NC-SA | 10 |
| CC BY-NC-ND | 6 |
| CC BY-ND | 4 |
| Public domain | 4 |
| FAL | 1 |

本倉庫的程式與文件採 MIT 授權；**影像的授權屬於原上傳者，不在本倉庫的授權範圍內**。CC BY-NC 不可用於商業目的、CC BY／BY-SA 需要署名（`attribution` 欄位即為所需字串）。若之後要商用，用 `commercial_ok == "yes"` 過濾（246 張），並保留 `attribution`。模型權重由這批 CC BY-NC 為主的影像訓練而來，Demo 與權重同樣僅供非商業使用（CC BY-NC 4.0）。

## 已知限制

1. **只有 7 類**：不在類別內（含同屬其他種）的植物，模型仍會給出其中一類的高機率，Demo 不具備「未知」拒絕能力。
2. **取景混雜**：整株、樹冠、葉／果特寫都有，榕樹類約 4 成是葉近拍。做「通用」辨識是好事，做整株專用模型時需再過濾一輪。
3. **情境偏城市**：多為公園、路樹、校園、庭園，森林原生境與遠景小樹覆蓋率低。
4. **地理偏斜**：熱帶與南半球觀察量大，台灣本地占比有限。要強化台灣街景，可用 iNaturalist 的 `place_id` 限定台灣再抓一輪。
5. **解析度上限**：iNaturalist `large` 檔為長邊 1024 px；要原圖請加 `--inat-size original`（檔案量約 5 倍）。
6. **僅供教學展示**：不能取代專業植物鑑定。

## 擴大抓取

各物種 iNaturalist 的 research 級池量：榕樹 11,404、大葉欖仁 14,714、柱狀南洋杉 3,048、散尾葵 1,718、槲蕨 1,864、紅桑 1,731、鶴望蘭 769。

每類 500 張：`collect.py --per-source 1600` 再 `build.py --per-label 500`。鶴望蘭會先到頂，要更多就必須放寬到 `needs_id`，標籤品質隨之下降，需人工複核。
