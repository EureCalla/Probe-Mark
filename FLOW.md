# Probe-Mark 使用與運作流程

> 版本：v2.0 | 探針痕跡影像辨識系統（二元語義分割）

---

## 專案概覽

Probe-Mark 是一個針對**探針痕跡（probe mark）**的影像辨識系統。  
輸入原始探針影像，訓練模型預測哪些像素區域有探針痕跡（二元分割：有痕 = 1，無痕 = 0）。  
模型骨幹採用 `segmentation_models_pytorch`，支援多種 Encoder/Decoder 組合。

---

## 目錄結構

```
Probe-Mark/
├── data/
│   ├── raw/                  # 原始資料（Excel 含嵌入圖片）
│   │   └── {product_name}/
│   │       └── picture.xlsx
│   └── processed/            # 預處理後的訓練資料
│       └── {product_name}/
│           └── {sample_name}/
│               ├── image.png         # 原始影像
│               └── ground_truth.png  # 二元分割遮罩（0/1）
├── probe_mark/
│   ├── main.py               # 主程式入口（目前為 stub，opts 解析用）
│   ├── train.py              # 訓練主流程（實際執行點）
│   ├── trainer.py            # Trainer class（訓練/驗證迴圈）
│   ├── opts.py               # 參數定義（argparse）
│   ├── logger.py             # TensorBoard Logger
│   ├── utils.py              # 工具函式（seed、DDP 設定）
│   ├── datasets/
│   │   ├── seg_binary_dataset.py  # Dataset class
│   │   ├── transforms.py          # 影像前處理 / 資料增強
│   │   └── processing.py          # 從 Excel 提取影像（新版）
│   └── dataset_Della/
│       └── get_img.py        # 從 Excel 提取影像（舊版 Della 格式）
├── notebook/
│   └── PM.ipynb              # 探索性實驗筆記（VGG16、UNet 自訂架構）
├── runs/                     # 訓練輸出（log、模型權重）
│   └── {exp_id}/
│       └── {timestamp}/
│           ├── opt.txt           # 本次訓練參數紀錄
│           ├── snapshot.pt       # 每 epoch 更新的訓練快照
│           └── best_model.pt     # 驗證指標最佳的模型權重
├── requirements.txt
└── introduce.py
```

---

## 完整運作流程

### 第一步：環境安裝

```bash
pip install -r requirements.txt
```

主要相依套件：

| 套件 | 用途 |
|------|------|
| `pytorch` | 深度學習框架 |
| `segmentation-models-pytorch` | Encoder-Decoder 分割模型 |
| `torchmetrics` | 評估指標（IoU、Dice、AP） |
| `openpyxl` | 讀取 Excel 嵌入圖片 |
| `opencv-python` | 影像處理（差異遮罩生成） |

---

### 第二步：資料準備（Excel → PNG）

原始資料為 Excel 檔，每個產品一個資料夾，包含一個 `picture.xlsx`：

- **`主要` 工作表**：嵌入原始探針影像
- **`體積面積量測` 工作表**：嵌入標記紅色痕跡的影像

執行 `datasets/processing.py` 提取並生成訓練資料：

```bash
python probe_mark/datasets/processing.py
# 預設讀取 data/raw/，輸出至 data/processed/
```

**處理流程（ImageExtractor）**：

1. 讀取 `主要` 工作表 → 取得原始影像（`image`）
2. 讀取 `體積面積量測` 工作表 → 取得紅痕標記影像（`label`）
3. 過濾不完整資料（缺 image 或 label 的樣本）
4. 移除已知問題資料（hardcoded：`Artificial_VM_D80_MW120F-HS` 的特定樣本）
5. 用 `cv2.absdiff` 比對 image vs label → 生成二元遮罩（有差異 = 1）
6. 輸出三個 PNG 至 `data/processed/{product}/{sample}/`：
   - `image.png`：原始影像
   - `label.png`：二元遮罩（0/1 灰階）
   - `mask_view.png`：可視化遮罩（0/255 灰階）

> **Della 格式**：舊版資料請改用 `dataset_Della/get_img.py`（邏輯相近，函式介面略不同）

---

### 第三步：Dataset 與資料前處理

`SegmentationBinaryDataset` 會自動：

1. 遞迴掃描 `data/processed/`，找出所有同時有 `image.png` + `ground_truth.png` 的資料夾
2. 用 sklearn `train_test_split` 以 **8:2** 切分訓練 / 驗證集（`seed=42`）
3. 讀取 PIL 影像：原圖轉 RGB，遮罩轉 L（灰階）

**影像前處理（`BASIC_TRANSFORMS`）**：

```
ToImage → Pad(上下各 129px) → Resize(224×224) → ToDtype(float32/int64) → Normalize(ImageNet 均值/標準差)
```

**資料增強（`AUGMENTATION_TRANSFORMS`，訓練集專用）**：

```
ToImage → Pad → Resize → RandomHorizontalFlip(p=0.5) → RandomVerticalFlip(p=0.5)
       → RandomAdjustSharpness(4, p=0.5) → RandomAutocontrast(p=0.5) → ToDtype → Normalize
```

> Pad 目的：原圖寬高比非正方形，上下各補 129px 黑邊使其接近正方形，再縮放至 224×224。

---

### 第四步：訓練

```bash
cd probe_mark
python train.py [options]
```

**常用參數**：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--exp_id` | `default` | 實驗名稱（決定 `runs/` 子目錄） |
| `--data_dir` | `data/processed` | 資料集路徑 |
| `--encoder_name` | `resnet18` | Encoder 名稱（參考 smp 文件） |
| `--encoder_weights` | `imagenet` | 預訓練權重 |
| `--decoder_name` | `FPN` | Decoder 架構 |
| `--max_epochs` | `10` | 訓練 epoch 數 |
| `--batch_size` | `16` | Batch size |
| `--lr` | `1e-4` | 初始學習率 |
| `--eta_min` | `1e-5` | Cosine Annealing 最低學習率 |
| `--gpu_id` | `0` | GPU 編號（`-1` = CPU） |
| `--seed` | `42` | 隨機種子 |

**訓練細節（Trainer）**：

- **Loss**：`BCEWithLogitsLoss`
- **Optimizer**：`Adam`
- **LR Scheduler**：`CosineAnnealingLR`（T_max = max_epochs）
- **評估指標（每 epoch）**：
  - Binary Average Precision
  - Mean IoU
  - Generalized Dice Score
- **儲存邏輯**：每 epoch 覆寫 `snapshot.pt`；若驗證 AP 超過歷史最佳，額外儲存 `best_model.pt`
- **Log 路徑**：`runs/{exp_id}/{YYYY-MM-DD-HH-MM-SS}/`

**自訂模型範例**：

```bash
# ResNet50 + Unet，使用 ImageNet 預訓練
python train.py --exp_id my_exp --encoder_name resnet50 --decoder_name Unet --max_epochs 50
```

**查看訓練曲線（TensorBoard）**：

```bash
tensorboard --logdir runs/
```

---

### 第五步：恢復訓練（Resume）

```bash
python train.py --resume --snapshot_path runs/{exp_id}/{timestamp}/snapshot.pt
```

- 從 `snapshot.pt` 還原 epoch 數、模型權重、optimizer 狀態、lr_scheduler 狀態
- Log 繼續寫入原本的 `log_dir`

---

### 第六步：推論 / 測試

```bash
python main.py --test --load_model_path runs/{exp_id}/{timestamp}/best_model.pt [--image_path path/to/image]
```

> 注意：`main.py` 的 `main()` 目前為空實作（stub），`--test` 模式的推論邏輯尚未完成，需自行擴充或改用 notebook。

---

## 訓練輸出說明

```
runs/{exp_id}/{timestamp}/
├── opt.txt         # 本次訓練所有參數（含 torch/cudnn 版本）
├── snapshot.pt     # 每 epoch 覆寫：儲存 epochs_run / model / optimizer / lr_scheduler
└── best_model.pt   # 驗證 Binary AP 最高時儲存（僅 model.state_dict()）
```

---

## 模型架構（segmentation_models_pytorch）

```
Input (B, 3, 224, 224)
     ↓
Encoder（resnet18 / resnet50 / vgg16 / efficientnet...）
     ↓ 多尺度特徵圖
Decoder（FPN / Unet / DeepLabV3Plus...）
     ↓
Output (B, 1, 224, 224)  ← 未經 sigmoid 的 logits
     ↓
BCEWithLogitsLoss（訓練）or sigmoid → > 0.5（推論）
```

---

## 目前已知狀態（v2 branch）

| 項目 | 狀態 |
|------|------|
| 資料預處理（Excel → PNG） | ✅ 完成（`processing.py`） |
| Dataset / DataLoader | ✅ 完成 |
| 影像前處理 / 增強 | ✅ 完成 |
| 訓練主迴圈（`train.py`） | ✅ 完成 |
| TensorBoard logging | ✅ 完成 |
| Resume 訓練 | ✅ 完成 |
| `main.py` 推論模式 | ⚠️ stub，`main()` 未實作 |
| 單張影像推論 | ⚠️ 未完成 |
| 測試集評估報告 | ⚠️ 僅在 notebook 有示範 |
| DDP 多 GPU 訓練 | ⚠️ `utils.py` 有骨架，未整合進 `train.py` |

---

## 快速上手指令

```bash
# 1. 安裝環境
pip install -r requirements.txt

# 2. 準備資料（從 Excel 提取）
python probe_mark/datasets/processing.py

# 3. 開始訓練（預設參數）
cd probe_mark && python train.py --exp_id probe_v2

# 4. 查看訓練曲線
tensorboard --logdir ../runs/

# 5. 恢復訓練
python train.py --resume --snapshot_path ../runs/probe_v2/{timestamp}/snapshot.pt
```
