# Probe-Mark

Probe-Mark 是探針痕跡影像辨識工具。現在的設計改成類似 `MP_PCB-tester` 的三段式流程：資料清洗、模型訓練、模型預測。差異是 Probe-Mark 處理的是影像，所以影像本體維持存成 PNG，SQLite 只負責索引、快取狀態、split、training run、model 與 prediction run。

## 快速啟動

```bash
pip install -r requirements.txt
python main.py
```

## 程式碼結構（main.py 執行路徑）

```text
main.py                                      # 執行入口（Tkinter UI，三個主要按鈕）
│
├── [資料清洗 按鈕]
│   └── load_clean_module/
│       ├── __init__.py                      # 匯出 LoadCleanService
│       └── loader.py                        # LoadCleanService：Excel → PNG + DB cache
│           ├── extract_from_sheet()         # 從指定工作表與 TARGET_COL 擷取內嵌圖片
│           ├── imwrite_unicode()            # Windows 中文路徑安全寫圖
│           └── database/db_manager.py       # 登記 source_excels / datasets / samples
│
├── [模型訓練 按鈕]
│   └── train_module/
│       ├── __init__.py                      # 匯出 Trainer
│       └── trainer.py                       # Trainer：DB samples → split → PyTorch 訓練
│           ├── database/db_manager.py       # 讀 datasets/samples/splits，寫 training_runs/models
│           └── probe_mark/
│               ├── datasets/                # SampleListSegmentationDataset + transforms
│               ├── trainer.py               # epoch 訓練、驗證、snapshot、best_model
│               └── logger.py                # opt.txt + TensorBoard
│
└── [模型預測 按鈕]
    └── predict_module/
        ├── __init__.py                      # 匯出 Predictor
        └── predictor.py                     # Predictor：載入 DB model，單張/資料夾批次預測
            ├── probe_mark/predictor.py      # 單張影像推論與 mask/compare 輸出
            └── database/db_manager.py       # 寫 prediction_runs / prediction_outputs
```

## 資料結構

### 清洗輸入

GUI 可選一個或多個 Excel。每個任務需要一個 `dataset_name`，預設使用 Excel 檔名。

Excel 工作表與欄位：

| 設定 | 值 | 說明 |
|---|---|---|
| `SHEET_IMAGE` | `主要` | 原始探針影像來源 |
| `SHEET_LABEL` | `體積面積量測` | 標記或量測影像來源 |
| `TARGET_COL` | `2` | 只取雷射 + 彩色欄位，避開高度/3D/調色板縮圖 |

### 清洗輸出

```text
data/processed/
  {dataset_name}/
    {sample_name}/
      image.png          # 原始影像
      label.png          # Excel 中擷取出的標記影像，保留給人工檢查
      ground_truth.png   # 訓練用 0/1 二元 mask，Dataset 會讀這個檔案
      mask_view.png      # 0/255 mask，方便人眼檢查
```

### SQLite DB

DB 位置：

```text
database/probe_mark.db
```

此 DB 是本機快取產物，已由 `.gitignore` 排除。schema 由 `database/db_manager.py` 的 `DBManager.init_db()` 建立。

核心 tables：

| table | 用途 |
|---|---|
| `source_excels` | 記錄 Excel 原始路徑、檔名、mtime、size、output_name |
| `datasets` | 記錄一次清洗結果，包含 dataset name、processed dir、sample 數量、狀態 |
| `samples` | 記錄每個 sample 的 `image.png`、`ground_truth.png`、`mask_view.png` 路徑 |
| `dataset_splits` | 記錄固定 split 設定，例如 `default`、seed、train/val/test ratio |
| `sample_splits` | 記錄每個 sample 屬於 train / val / test |
| `training_runs` | 記錄一次訓練的參數與狀態 |
| `models` | 記錄訓練輸出的 model dir、`best_model.pt`、`snapshot.pt`、`opt.txt` |
| `prediction_runs` | 記錄一次預測任務 |
| `prediction_outputs` | 記錄每張影像的 mask 與 compare 圖輸出 |

## 按鈕流程

### 資料清洗

1. 使用者按「資料清洗」。
2. 選擇 Excel，確認 `dataset_name`。
3. `LoadCleanService` 先查 DB cache。
4. 若相同 Excel path + output name 已清洗且 PNG 檔案仍存在，預設沿用快取。
5. 若勾選「強制重建快取」，重新抽圖、產生 `ground_truth.png`，並更新 DB。

### 模型訓練

1. 使用者按「模型訓練」。
2. GUI 從 DB 讀取已完成的 dataset。
3. `Trainer` 建立或覆寫 `default` split。
4. 訓練資料由 DB 的 `samples + sample_splits` 取得，不再只靠掃資料夾。
5. 訓練完成後寫入 `training_runs` 與 `models`。

訓練輸出：

```text
runs/{run_name}/run_{run_id}_{YYYY-MM-DD-HH-MM-SS}/
  opt.txt
  snapshot.pt
  best_model.pt
  events.out.tfevents...
```

### 模型預測

1. 使用者按「模型預測」。
2. GUI 從 DB 讀取已完成模型。
3. 使用者選單張影像或資料夾。
4. `Predictor` 載入 `best_model.pt`，逐張輸出 mask 與 compare 圖。
5. 預測任務與每張輸出寫入 DB。

預測輸出：

```text
data/processed/predictions/model_{model_id}_{YYYY-MM-DD-HH-MM-SS}/
  {image_stem}_mask.png
  {image_stem}_compare.png
```

## 維護重點

- 影像本體存 PNG，不存進 SQLite，也不放進 parquet。
- SQLite 是資料索引與流程狀態，讓清洗結果、訓練 split、模型與預測輸出可追蹤。
- `ground_truth.png` 是正式訓練標籤；`mask.png` 不再作為 Dataset 讀取規格。
- 三個 GUI 按鈕都只呼叫 service 類別，不直接寫資料庫細節。
- 若未來資料量大到 DB 查詢不足，再新增 `dataset_manifest.parquet`；目前 v1 不加入 parquet。

## 測試

```bash
python -m py_compile main.py clean_data/clean_up.py database/db_manager.py load_clean_module/loader.py train_module/trainer.py predict_module/predictor.py
pytest tests/test_db_manager.py
```
