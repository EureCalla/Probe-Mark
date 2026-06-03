# Probe-Mark

Probe-Mark 是探針痕跡影像辨識工具。現在的設計改成類似 `MP_PCB-tester` 的三段式流程：資料清洗、模型訓練、模型預測。差異是 Probe-Mark 處理的是影像，所以影像本體維持存成 PNG，SQLite 只負責索引、快取狀態、split、training run、model 與 prediction run。

## 快速啟動

```bash
pip install -r requirements.txt
python main.py
```

## 程式碼結構（main.py 執行路徑）

```text
main.py                                      # 執行入口：Tkinter GUI、按鈕流程、狀態列、即時圖表
│
├── mpivr20_cms.py                           # 共用輸出/DB 路徑設定
│   ├── get_output_root_dir()                 # 共享根目錄：\\mpi-file01\...\Probe-Mark
│   ├── get_clean_output_dir()                # 清洗資料輸出根目錄：...\Probe-Mark\data
│   └── get_database_path()                   # SQLite DB：...\Probe-Mark\probe_mark.db
│
├── main.py 內部 GUI/helper 函式
│   ├── run_background()                      # 背景 thread 執行耗時任務並更新 status/messagebox
│   ├── choose_file() / choose_files()        # 選單張或多張檔案
│   ├── choose_dir()                          # 選資料夾
│   ├── scan_excel_folder()                   # 掃描資料夾內的 .xlsx/.xlsm
│   ├── default_dataset_name()                # Excel 檔名轉 dataset_name
│   ├── format_clean_result()                 # 清洗完成摘要
│   ├── safe_folder_name()                    # Windows-safe run name
│   ├── model_output_path()                   # 模型輸出預覽路徑
│   ├── dataset_options()                     # DBManager.list_datasets/count_samples_by_status
│   └── model_options()                       # DBManager.list_models
│
├── gui/
│   ├── __init__.py                           # 匯出 LogPanel / TkLogHandler / install_log_panel
│   └── log_panel.py                          # main() 呼叫 install_log_panel()
│       ├── LogPanel                          # Tkinter ScrolledText 執行紀錄面板
│       ├── TkLogHandler                      # logging record 寫入 GUI
│       ├── StreamToLogger                    # stdout/stderr 導到 logger
│       └── install_log_panel()               # 建立 log panel 並掛到 root logger
│
├── [資料清洗 按鈕]
│   ├── open_clean_dialog()                   # 建立清洗設定視窗
│   ├── choose_files()                        # 可逐檔加入 Excel
│   ├── choose_excel_folder()                 # 可整個資料夾掃描 Excel
│   └── load_clean_module/
│       ├── __init__.py                       # 匯出 LoadCleanService
│       └── loader.py                         # LoadCleanService：Excel -> PNG + DB cache
│           ├── GroundTruthAreaError          # ground_truth 全部不合格時拋出
│           ├── find_excel()                  # 掃 data/raw 產品資料夾時尋找 Excel
│           ├── decode_image()                # Excel 圖片 bytes -> OpenCV image
│           ├── extract_from_sheet()          # 從 SHEET_IMAGE/SHEET_LABEL + TARGET_COL 擷取圖片
│           ├── ground_truth_ratio()          # 計算 mask 面積比例
│           ├── ground_truth_violation()      # 判斷最大面積/最小 pixels 門檻
│           ├── violation_message()           # 格式化清洗失敗訊息
│           ├── imwrite_unicode()             # Windows 中文路徑安全寫圖
│           ├── LoadCleanService.scan_tasks() # 無 GUI tasks 時掃 data/raw
│           ├── LoadCleanService.run()        # 初始化 DB、逐個 Excel 任務清洗
│           ├── LoadCleanService.process_task()
│           ├── LoadCleanService.validate_cached_ground_truth()
│           ├── LoadCleanService._process_excel()
│           ├── database/db_manager.py        # upsert source/dataset、replace samples、cache_valid
│           └── mpivr20_cms.py                # get_clean_output_dir()
│
├── [模型訓練 按鈕]
│   ├── open_train_dialog()                   # 建立訓練設定視窗與即時 loss/IoU 圖表
│   ├── dataset_options()                     # 讀取可訓練 datasets 與 active/failed sample 數
│   ├── matplotlib.figure.Figure              # GUI 內嵌訓練曲線
│   ├── matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
│   ├── database/db_manager.py                # 讀 dataset、poll training_runs/epoch_metrics
│   └── train_module/
│       ├── __init__.py                       # 匯出 Trainer
│       └── trainer.py                        # Trainer：DB samples -> split -> PyTorch 訓練
│           ├── Trainer._ensure_split()       # train dataset 切 train/test，外部 dataset 當 validation
│           ├── Trainer.list_samples_for_datasets()
│           ├── Trainer._make_opt()           # 建立 core trainer 所需 SimpleNamespace
│           ├── Trainer.run()                 # 建 model/dataloader/logger，訓練、測試、寫 DB
│           ├── Trainer.training_params()     # 保存本次訓練參數
│           ├── database/db_manager.py        # get/list samples、create_split、training_runs、models、metrics
│           ├── mpivr20_cms.py                # get_output_root_dir()
│           └── probe_mark/                   # train_module 會把此資料夾加入 sys.path
│               ├── datasets/
│               │   ├── __init__.py           # 匯出 Dataset 與 transforms
│               │   ├── seg_binary_dataset.py
│               │   │   ├── SampleListSegmentationDataset  # DB sample list -> image/ground_truth
│               │   │   └── SegmentationBinaryDataset      # 舊資料夾掃描 Dataset
│               │   └── transforms.py
│               │       ├── AUGMENTATION_TRANSFORMS()      # train augmentation
│               │       ├── BASIC_TRANSFORMS()             # val/test transform
│               │       └── PREDICT_TRANSFORMS()           # predict transform，預測流程也會用
│               ├── trainer.py                 # CoreTrainer：epoch train/validate/test
│               │   ├── train_one_epoch()
│               │   ├── validate()
│               │   ├── evaluate_loader()
│               │   ├── _save_snapshot()
│               │   └── run()
│               ├── logger.py                  # TensorBoard SummaryWriter、opt.txt、profiler
│               └── utils.py                   # seed_everything()
│
└── [模型預測 按鈕]
    ├── open_predict_dialog()                 # 建立預測視窗與模型訓練 metrics 圖表
    ├── model_options()                       # 讀取可用 models
    ├── matplotlib.figure.Figure              # GUI 內嵌模型 loss/IoU/Dice 圖
    ├── matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
    ├── database/db_manager.py                # get_model/list_epoch_metrics
    └── predict_module/
        ├── __init__.py                       # 匯出 Predictor
        └── predictor.py                      # Predictor：載入 DB model，單張/資料夾批次預測
            ├── Predictor._iter_images()      # 收集單張或資料夾內影像
            ├── Predictor._make_opt()         # 建立 core predictor 所需 SimpleNamespace
            ├── Predictor.run()               # 寫 prediction_run/output 狀態
            ├── database/db_manager.py        # get_model、prediction_runs、prediction_outputs
            └── probe_mark/                   # predict_module 會把此資料夾加入 sys.path
                ├── predictor.py              # CorePredictor：模型載入與單張影像推論
                │   ├── _preprocess()
                │   ├── _postprocess()
                │   ├── _save_mask()
                │   ├── _save_overlay()
                │   ├── _save_compare()
                │   └── predict()
                └── datasets/transforms.py    # PREDICT_TRANSFORMS()
```

`main.py` 執行路徑會共用的 DB 模組：

```text
database/
├── __init__.py                               # 匯出 DBManager
└── db_manager.py                             # SQLite schema 與所有流程狀態
    ├── init_db()                             # 建立/補齊 tables 與欄位
    ├── source/dataset/sample                 # upsert_source_excel、upsert_dataset、replace_samples
    ├── cache/split                           # cache_valid、create_split、get_split_samples
    ├── training/model/metrics                # training_runs、models、epoch/test metrics
    └── prediction                            # prediction_runs、prediction_outputs
```

`main.py` 間接使用的主要第三方套件：

```text
tkinter                                      # GUI
matplotlib                                   # GUI 內嵌訓練/預測 metrics 圖與 compare 圖
openpyxl + opencv-python + numpy             # Excel 圖片抽取、mask 產生、PNG 輸出
Pillow                                       # Dataset 與預測影像讀寫
torch + torchvision + torchmetrics           # Dataset transforms、訓練、metrics
segmentation_models_pytorch                  # FPN/Unet/DeepLabV3Plus 等模型架構
scikit-learn                                 # train_test_split
tqdm                                         # core trainer progress
tensorboard                                  # SummaryWriter
```

不在 `python main.py` 主要執行路徑、但保留在專案中的檔案：

```text
clean_data/clean_up.py                       # 舊/CLI 清洗入口
probe_mark/main.py                           # 舊 probe_mark CLI 入口
probe_mark/train.py                          # 舊訓練 CLI
probe_mark/opts.py                           # 舊 CLI argparse options
probe_mark/datasets/processing.py            # 舊資料處理工具
probe_mark/dataset_Della/get_img.py          # Della 資料抽圖工具
tests/test_db_manager.py                     # DBManager 測試
setup.py / setup.cfg / Makefile              # 專案包裝與工具設定
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
\\mpi-file01\VPCRD2-AI\08_Training\Probe-Mark\probe_mark.db
```

清洗輸出的圖片資料放在同一個共享根目錄的 `data\{dataset_name}` 底下。schema 由 `database/db_manager.py` 的 `DBManager.init_db()` 建立。

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
3. 可在「進階清洗規則」調整 `ground_truth` 品質門檻，預設最大面積 50%、最小 30 pixels。
4. `LoadCleanService` 先查 DB cache。
5. 若相同 Excel path + output name 已清洗且 PNG 檔案仍存在，預設沿用快取。
6. 若勾選「強制重建快取」，重新抽圖、產生 `ground_truth.png`，並更新 DB。
7. `ground_truth` 面積超過最大比例或低於最小 pixels 的 sample 會標記為 `failed`，不會進入訓練。

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
