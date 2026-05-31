# Probe-Mark

Probe-Mark 是探針痕跡影像辨識工具，使用 `segmentation_models.pytorch` 建立二元語意分割模型。根目錄的 `main.py` 是給使用者操作的 Tkinter GUI 入口，預計把流程拆成三個按鈕：

1. 資料清洗
2. 模型訓練
3. 模型預測

目前 `v2_Callami01_code-refactor` 分支中，「資料清洗」已經接上實際流程；「模型訓練」與「模型預測」按鈕已放在 GUI 上，但仍是 `disabled`，後續要把參數視窗和 subprocess 呼叫補上。

## 快速啟動

```bash
pip install -r requirements.txt
python main.py
```

## 程式碼結構（main.py 執行路徑）

```text
main.py                                  # 執行入口（Tkinter UI，三個主要按鈕）
│
├── [資料清洗 按鈕] 已啟用
│   ├── open_clean_dialog()              # 開啟 Excel 選取視窗，建立清洗任務
│   ├── run_clean_up_tasks()             # 將任務寫成暫存 JSON，使用 subprocess 執行清洗腳本
│   └── clean_data/
│       └── clean_up.py                  # Excel 轉 PNG、產生 mask、輸出到 data/processed
│           ├── run_tasks()              # 讀取 main.py 傳入的 tasks-json
│           ├── process_excel()          # 處理單一 Excel，配對 image / label
│           ├── extract_from_sheet()     # 從指定工作表與欄位擷取內嵌圖片
│           ├── imwrite_unicode()        # Windows 中文路徑安全寫圖
│           └── run_scan()               # 不透過 GUI 時，掃描 data/raw 批次處理
│
├── [模型訓練 按鈕] 尚未啟用
│   └── 預期呼叫 probe_mark/main.py
│       ├── opts.py                      # 解析訓練參數、設定 root_dir / device / log_dir
│       ├── logger.py                    # 建立 opt.txt 與 TensorBoard log
│       └── train.py                     # 建立 Dataset / DataLoader / model，開始訓練
│           ├── datasets/
│           │   ├── seg_binary_dataset.py # 掃描 data/processed，建立 image + ground_truth sample
│           │   └── transforms.py         # train / val 影像前處理與資料增強
│           ├── trainer.py               # 訓練、驗證、metrics、snapshot、best_model
│           └── utils.py                 # seed 等共用工具
│
└── [模型預測 按鈕] 尚未啟用
    └── 預期呼叫 probe_mark/main.py --test
        ├── opts.py                      # 解析 load_model_path / image_path / output_dir
        └── predictor.py                 # 載入模型，對單張影像輸出 mask 與 compare 圖
            └── datasets/
                └── transforms.py         # PREDICT_TRANSFORMS，預測用影像前處理
```

維護重點：

- 使用者執行的是根目錄 `main.py`，不是 `probe_mark/main.py`。
- 根目錄 `main.py` 負責 GUI 與 subprocess；模型訓練/預測邏輯集中在 `probe_mark/`。
- 目前只有「資料清洗」按鈕已真正串到 `clean_data/clean_up.py`。
- 「模型訓練」與「模型預測」按鈕目前仍是 `disabled`，README 先寫出建議串接路徑，後續實作時照這個路徑維護。

## GUI 按鈕與呼叫檔案

| GUI 按鈕 | 目前狀態 | 呼叫檔案 | 呼叫方式 | 用途 |
|---|---:|---|---|---|
| 資料清洗 | 已啟用 | `clean_data/clean_up.py` | `subprocess.run([python, clean_up.py, "--tasks-json", tmp_json])` | 選取一個或多個 Excel，轉成訓練用 PNG 資料夾 |
| 模型訓練 | 尚未啟用 | `probe_mark/main.py` -> `probe_mark/train.py` | 預期用 subprocess 傳入訓練參數，不帶 `--test` | 讀取 `data/processed`，訓練 segmentation model |
| 模型預測 | 尚未啟用 | `probe_mark/main.py` -> `probe_mark/predictor.py` | 預期用 subprocess 傳入 `--test --load_model_path --image_path --output_dir` | 載入 `.pt` 模型，輸出 mask 與比對圖 |

## 執行流程

### 1. 根目錄 `main.py`

`main.py` 負責建立 Tkinter 視窗與三個主要按鈕。

目前會用到的關鍵常數：

| 名稱 | 值 | 說明 |
|---|---|---|
| `REPO_ROOT` | repo 根目錄 | subprocess 的工作目錄 |
| `CLEAN_UP_SCRIPT` | `clean_data/clean_up.py` | 資料清洗腳本位置 |
| `OUTPUT_BASE` | `data/processed` | GUI 顯示的清洗輸出根目錄 |

資料清洗按鈕流程：

1. 使用者按下「資料清洗」。
2. `open_clean_dialog()` 開啟 Excel 選取視窗。
3. 使用者選取一個或多個 `.xlsx` / `.xlsm`。
4. GUI 產生暫存 JSON，格式是：

```json
[
  {
    "excel": "D:/path/to/source.xlsx",
    "output_name": "product_or_dataset_name"
  }
]
```

5. `run_clean_up_tasks()` 用 subprocess 執行：

```bash
python clean_data/clean_up.py --tasks-json <tmp_json>
```

6. `clean_up.py` 將結果寫到 `data/processed/{output_name}/`。

### 2. `clean_data/clean_up.py`

`clean_up.py` 負責把 Excel 內嵌圖片整理成模型可讀的影像資料。它支援兩種模式：

| 模式 | 指令 | 用途 |
|---|---|---|
| GUI 任務模式 | `python clean_data/clean_up.py --tasks-json tasks.json` | 由 `main.py` 傳入多個 Excel 與輸出名稱 |
| 掃描模式 | `python clean_data/clean_up.py` | 掃描 `data/raw/{product}/`，尋找 Excel 後批次轉檔 |

清洗邏輯：

1. 讀取 Excel。
2. 從指定工作表抓圖。
3. 只取 `TARGET_COL = 2` 的圖片，避免抓到彩色、高度、3D 或調色板縮圖。
4. 依 sample name 配對 `image` 與 `label`。
5. 若尺寸不同，將 label resize 到 image 尺寸。
6. 用 `cv2.absdiff(image, label)` 計算差異。
7. 產生 binary mask。
8. 用 `imwrite_unicode()` 寫圖，避免 Windows 中文路徑下 `cv2.imwrite()` 靜默失敗。

單一 sample 的目前輸出：

```text
data/processed/{output_name}/{sample_name}/
  image.png      # 原始探針影像
  label.png      # Excel 中取出的標記影像
  mask.png       # 0/1 binary mask
  mask_view.png  # 0/255 mask，方便人眼檢查
```

### 3. `probe_mark/main.py`

`probe_mark/main.py` 是模型流程的 CLI 分流入口：

```python
if opt.test:
    Predictor(opt).predict(opt.image_path)
else:
    train_main(opt, logger)
```

也就是：

| 條件 | 會呼叫 | 用途 |
|---|---|---|
| 不帶 `--test` | `probe_mark/train.py` | 模型訓練 |
| 帶 `--test` | `probe_mark/predictor.py` | 單張影像預測 |

後續 GUI 的「模型訓練」與「模型預測」按鈕建議都呼叫這個檔案，讓 CLI 與 GUI 共用同一套參數解析與流程。

### 4. `probe_mark/train.py`

`train.py` 負責建立 dataset、model、trainer 並開始訓練。

主要流程：

1. `SegmentationBinaryDataset(root=opt.data_dir, is_train=True)` 建立訓練集。
2. `SegmentationBinaryDataset(root=opt.data_dir, is_train=False)` 建立驗證集。
3. 使用 `DataLoader` 載入資料。
4. 從 `segmentation_models_pytorch` 依 `opt.decoder_name` 建立模型。
5. 使用 `Trainer` 執行訓練與驗證。
6. 使用 `Logger` 寫入 TensorBoard 與參數紀錄。

常用訓練指令：

```bash
python probe_mark/main.py --data_dir data/processed --exp_id default
```

常用參數在 `probe_mark/opts.py`：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--data_dir` | `data/processed` | 訓練資料根目錄 |
| `--exp_id` | `default` | 實驗名稱，影響 `runs/{exp_id}/` |
| `--gpu_id` | `0` | GPU id，`-1` 表示 CPU |
| `--encoder_name` | `resnet18` | encoder 名稱 |
| `--encoder_weights` | `imagenet` | encoder 預訓練權重 |
| `--decoder_name` | `FPN` | segmentation decoder |
| `--max_epochs` | `10` | epoch 數 |
| `--batch_size` | `16` | batch size |
| `--num_workers` | `8` | DataLoader workers |
| `--lr` | `1e-4` | learning rate |
| `--eta_min` | `1e-5` | CosineAnnealingLR 最低 learning rate |

訓練輸出：

```text
runs/{exp_id}/{YYYY-MM-DD-HH-MM-SS}/
  opt.txt                  # 本次訓練參數與 torch/cudnn 版本
  snapshot.pt              # 每個 epoch 更新的訓練狀態
  best_model.pt            # 驗證指標最佳的 model.state_dict()
  events.out.tfevents...   # TensorBoard log
```

### 5. `probe_mark/predictor.py`

`predictor.py` 負責載入訓練好的 `.pt` 模型，對單張影像做預測。

常用預測指令：

```bash
python probe_mark/main.py --test ^
  --load_model_path runs/default/2026-01-01-120000/best_model.pt ^
  --image_path data/processed/example/sample_001/image.png ^
  --output_dir outputs
```

預測流程：

1. 用 `PREDICT_TRANSFORMS()` 將影像轉成模型輸入。
2. 載入 `--load_model_path` 指定的 `.pt` 權重。
3. 執行 model inference。
4. `sigmoid(logits) > 0.5` 產生 binary mask。
5. 輸出 mask 與原圖/遮罩比對圖。

預測輸出：

```text
outputs/
  {image_stem}_mask.png
  {image_stem}_compare.png
```

## 資料結構

### 原始資料

GUI 模式可以選任意 Excel 檔，不要求一定放在 repo 內。

掃描模式則預期原始資料放在：

```text
data/raw/
  {product_name}/
    *.xlsx
```

Excel 內目前使用兩個工作表：

| 工作表 | 用途 |
|---|---|
| `主要` | 原始影像來源 |
| `體積面積量測` | 標記或量測影像來源 |

程式只讀取第 2 欄圖片，也就是 `clean_data/clean_up.py` 的 `TARGET_COL = 2`。

### 清洗後資料

`clean_data/clean_up.py` 目前輸出：

```text
data/processed/
  {output_name}/
    {sample_name}/
      image.png
      label.png
      mask.png
      mask_view.png
```

### 訓練資料讀取規格

`probe_mark/datasets/seg_binary_dataset.py` 目前會遞迴搜尋同一個資料夾內同時存在下列檔案的 sample：

```text
image.png
ground_truth.png
```

也就是 Dataset 目前期待：

```text
data/processed/
  {dataset_or_product}/
    {sample_name}/
      image.png
      ground_truth.png
```

維護注意：目前 `clean_up.py` 寫出的 mask 檔名是 `mask.png`，但 `SegmentationBinaryDataset` 期待 label 檔名是 `ground_truth.png`。在把「資料清洗」直接接到「模型訓練」前，需要二選一：

1. 將 `clean_up.py` 輸出的 `mask.png` 另存或改名為 `ground_truth.png`。
2. 將 `SegmentationBinaryDataset.file_mapping` 從 `ground_truth.png` 改成 `mask.png`。

建議優先採用第 1 種，因為 `ground_truth.png` 比較清楚表示這是訓練標籤。

## 檔案職責總表

| 檔案 | 職責 | 主要被誰呼叫 |
|---|---|---|
| `main.py` | Tkinter GUI 入口，管理三個按鈕與 subprocess | 使用者直接執行 |
| `clean_data/clean_up.py` | Excel 轉 PNG、產生 mask | `main.py` 的資料清洗按鈕 |
| `probe_mark/main.py` | 模型 CLI 分流入口，依 `--test` 決定 train/predict | 未來 GUI 訓練/預測按鈕、CLI |
| `probe_mark/opts.py` | argparse 參數定義與路徑檢查 | `probe_mark/main.py`、`train.py` |
| `probe_mark/train.py` | 建立 dataset、model、trainer，開始訓練 | `probe_mark/main.py` |
| `probe_mark/trainer.py` | epoch 訓練、驗證、metrics、snapshot、best model | `train.py` |
| `probe_mark/logger.py` | 寫入 `opt.txt` 與 TensorBoard events | `train.py` |
| `probe_mark/predictor.py` | 單張影像預測，輸出 mask 與 compare 圖 | `probe_mark/main.py` |
| `probe_mark/datasets/seg_binary_dataset.py` | 遞迴建立 image/label sample list，切 train/val | `train.py` |
| `probe_mark/datasets/transforms.py` | 訓練、驗證、預測的影像 transform | dataset、predictor |
| `probe_mark/utils.py` | seed 等共用工具 | `train.py` |

## 後續接 GUI 時的維護建議

模型訓練按鈕建議收集這些欄位後呼叫 `probe_mark/main.py`：

```bash
python probe_mark/main.py ^
  --data_dir data/processed ^
  --exp_id default ^
  --gpu_id 0 ^
  --encoder_name resnet18 ^
  --encoder_weights imagenet ^
  --decoder_name FPN ^
  --max_epochs 10 ^
  --batch_size 16 ^
  --lr 0.0001
```

模型預測按鈕建議收集這些欄位後呼叫 `probe_mark/main.py`：

```bash
python probe_mark/main.py --test ^
  --load_model_path runs/default/<timestamp>/best_model.pt ^
  --image_path path/to/image.png ^
  --output_dir outputs
```

如果要支援資料夾批次預測，需要先修改 `probe_mark/predictor.py` 或 `probe_mark/main.py`，因為目前 `Predictor.predict()` 只處理單張影像。
