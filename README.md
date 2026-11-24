# tree7-dataset

七個樹種／園藝植物的影像資料集**建置管線**與**對照表（manifest）**。

**影像放在 Hugging Face：**[`datasets/Xieih/tree7`](https://huggingface.co/datasets/Xieih/tree7) — 2,143 張打包成 parquet（影像嵌在欄位內，附 train/valid/test 切分），不用先抓 500 MB 就能串流訓練。本倉庫只放程式、manifest 與文件：GitHub 單檔 100 MB、倉庫建議 <1 GB，而 CC BY-NC 影像公開散布需連署名一併處理。照下面的指令也能在本機重新抓出一份相同結構的資料集；`manifest.csv` 收錄全部 2,143 張的來源連結與授權，可逐張回溯。

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

## 快速開始

### A. 直接載入（推薦）

```python
from datasets import load_dataset

ds = load_dataset("Xieih/tree7")   # train 1714 / valid 215 / test 214
ds["train"][0]["image"]                                 # PIL.Image，存取時才解碼
ds["train"].features["label"].names                    # 0..6 對應下方類別表
```

兩邊倉庫都是 **public**，不需要 token。`label` 是 `ClassLabel`、
`image` 是 `Image()` 型別，兩者都由 parquet 內的 Hugging Face feature 簽名決定，不需要載入腳本。
切分為分層 80/10/10（seed `20261124`），每個 split 七類皆均衡；`splits.json` 記錄於 HF 倉庫。

### B. 在本機重建資料集

```bash
pip install -r requirements.txt
python3 collect.py --per-source 900   # 建候選索引（只存 metadata，不下載）
python3 build.py --per-label 300      # 下載、縮邊、去重，寫 images/<Label>/*.jpg
python3 repair_manifest.py            # 以磁盤為準重建清單並稽核
```

`build.py` 是增量的：把 `--per-label` 調高只會補抓差額，已存在的影像與 dHash 會被重用。
跑完後 `build_parquet.py`（存放在 HF 倉庫根目錄）可把 `images/` ＋ `manifest.jsonl` 打成與 A 相同的
parquet 結構。

## 資料來源與篩選條件

- **iNaturalist**（2,016 張，94.3%）：只取 `quality_grade=research`，即至少兩位專家同意鑑定，標籤可靠度高。抓取順序為「收藏數高→最新」，先取得較完整的株型照。原圖尺寸用 API 回傳的 `original_dimensions` 預篩（長邊 <900 px 排除），不會浪費頻寬下載縮圖。
- **Wikimedia Commons**（127 張，5.9%）：依物種分類頁抓取，標題黑名單排除地圖、插圖、標本掃描、書頁（BHL／CHS）、海報春聯等；另有「標題屬名 ≠ 目標屬名即排除」的檢查（含學名異名對照，見 `species.json` 的 `syn_genera`），用來攔住混入別種分類頁的檔案。
- **GBIF 未納入**：其 media 多為標本掃描與文獻縮圖，對活體植株辨識價值低。

去重採 64-bit dHash、漢明距離 ≤6；跨類比對目前為 0 組近似重複，代表沒有同一張照片被掛到兩個物種。

## manifest.csv 欄位

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
| CC BY-NC（含各版本） | 1,879 |
| CC0 | 81 |
| CC BY | 60 |
| CC BY-SA 4.0 / 3.0 | 64 |
| 其他（CC BY-NC-SA 等） | 39 |

本倉庫的程式與文件採 MIT 授權；**影像的授權屬於原上傳者，不在本倉庫的授權範圍內**。CC BY-NC 不可用於商業目的、CC BY／BY-SA 需要署名（`attribution` 欄位即為所需字串）。若之後要商用，用 `commercial_ok == "yes"` 過濾（目前 246 張），並保留 `attribution`。

## 已知取捨

1. **取景混雜**：整株、樹冠、葉／果特寫都有，榕樹類約 4 成是葉近拍。做「通用」辨識是好事，做整株專用模型時需再過濾一輪。
2. **情境偏城市**：多為公園、路樹、校園、庭園，森林原生境與遠景小樹覆蓋率低。
3. **地理偏斜**：熱帶與南半球觀察量大，台灣本地占比有限。要強化台灣街景，可用 iNaturalist 的 `place_id` 限定台灣再抓一輪。
4. **解析度上限**：iNaturalist `large` 檔為長邊 1024 px；要原圖請加 `--inat-size original`（檔案量約 5 倍）。

## 擴大抓取

各物種 iNaturalist 的 research 級池量：榕樹 11,404、大葉欖仁 14,714、柱狀南洋杉 3,048、散尾葵 1,718、槲蕨 1,864、紅桑 1,731、鶴望蘭 769。
每類 500 張：`collect.py --per-source 1600` 再 `build.py --per-label 500`。鶴望蘭會先到頂，要更多就必須放寬到 `needs_id`，標籤品質隨之下降，需人工複核。
