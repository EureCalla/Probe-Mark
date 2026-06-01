import os
from pathlib import Path

import cv2
import numpy as np
import openpyxl

from database import DBManager
from mpivr20_cms import get_clean_output_dir

RAW_DIR = "data/raw"
PROCESSED_DIR = get_clean_output_dir()
SHEET_IMAGE = "主要"
SHEET_LABEL = "體積面積量測"
TARGET_COL = 2


def imwrite_unicode(path: str, img: np.ndarray):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cv2.imencode 失敗：{path}")
    buf.tofile(path)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_excel(product_dir: str):
    for filename in os.listdir(product_dir):
        if filename.startswith("~$"):
            continue
        if filename.lower().endswith((".xlsx", ".xlsm")):
            return os.path.join(product_dir, filename)
    return None


def extract_from_sheet(sheet, key: str, data: dict):
    for img in sheet._images:
        row = img.anchor._from.row + 1
        col = img.anchor._from.col + 1
        if col != TARGET_COL:
            continue
        name = sheet.cell(row, 1).value
        if name is None:
            continue
        data.setdefault(str(name), {})[key] = decode_image(img._data())


class LoadCleanService:
    def __init__(self, tasks=None, save_dir=PROCESSED_DIR, force=False, db=None):
        self.tasks = tasks or []
        self.save_dir = save_dir
        self.force = force
        self.db = db or DBManager()

    @staticmethod
    def scan_tasks(raw_dir=RAW_DIR):
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"找不到 {raw_dir}")
        tasks = []
        for product in sorted(os.listdir(raw_dir)):
            product_dir = os.path.join(raw_dir, product)
            if not os.path.isdir(product_dir):
                continue
            excel_path = find_excel(product_dir)
            if excel_path:
                tasks.append({"excel": excel_path, "output_name": product})
        return tasks

    def run(self):
        self.db.init_db()
        if not self.tasks:
            self.tasks = self.scan_tasks()
        if not self.tasks:
            print("沒有可處理的 Excel 任務")
            return []

        dataset_ids = []
        total_samples = 0
        for task in self.tasks:
            excel_path = task["excel"]
            output_name = task["output_name"]
            dataset_id, n_samples, reused = self.process_task(excel_path, output_name)
            dataset_ids.append(dataset_id)
            total_samples += n_samples
            action = "沿用快取" if reused else "完成清洗"
            print(f"[{output_name}] {action}：{n_samples} samples, dataset_id={dataset_id}")

        print(f"全部完成：{len(dataset_ids)} dataset，{total_samples} samples")
        return dataset_ids

    def process_task(self, excel_path: str, output_name: str):
        if not os.path.isfile(excel_path):
            raise FileNotFoundError(f"檔案不存在：{excel_path}")

        source_id = self.db.upsert_source_excel(excel_path, output_name)
        out_dir = os.path.abspath(os.path.join(self.save_dir, output_name))
        existing = self.db.get_dataset_for_excel_path(excel_path)
        if not self.force and self.db.cache_valid(existing):
            return existing["id"], existing["n_samples"], True

        dataset_id = self.db.upsert_dataset(
            dataset_name=output_name,
            source_excel_id=source_id,
            processed_dir=out_dir,
            status="running",
        )
        try:
            samples = self._process_excel(excel_path, out_dir)
            self.db.replace_samples(dataset_id, samples)
        except Exception:
            self.db.mark_dataset_failed(dataset_id)
            raise
        return dataset_id, len(samples), False

    def _process_excel(self, excel_path: str, out_dir: str):
        print(f"載入 {excel_path}")
        wb = openpyxl.load_workbook(excel_path, data_only=True)
        missing = [s for s in (SHEET_IMAGE, SHEET_LABEL) if s not in wb.sheetnames]
        if missing:
            raise ValueError(f"缺少工作表 {missing}，現有：{wb.sheetnames}")

        data = {}
        extract_from_sheet(wb[SHEET_IMAGE], "image", data)
        extract_from_sheet(wb[SHEET_LABEL], "label", data)

        samples = []
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for sample_name, items in sorted(data.items()):
            if "image" not in items or "label" not in items:
                print(f"  缺資料跳過：{sample_name}")
                continue

            image = items["image"]
            label = items["label"]
            if image.shape != label.shape:
                height, width = image.shape[:2]
                label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)

            diff = cv2.absdiff(image, label)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ground_truth = cv2.threshold(gray, 1, 1, cv2.THRESH_BINARY)[1]
            mask_view = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

            sample_dir = os.path.join(out_dir, str(sample_name))
            os.makedirs(sample_dir, exist_ok=True)
            image_path = os.path.abspath(os.path.join(sample_dir, "image.png"))
            ground_truth_path = os.path.abspath(os.path.join(sample_dir, "ground_truth.png"))
            mask_view_path = os.path.abspath(os.path.join(sample_dir, "mask_view.png"))
            label_path = os.path.abspath(os.path.join(sample_dir, "label.png"))

            imwrite_unicode(image_path, image)
            imwrite_unicode(label_path, label)
            imwrite_unicode(ground_truth_path, ground_truth)
            imwrite_unicode(mask_view_path, mask_view)

            height, width = image.shape[:2]
            samples.append(
                {
                    "sample_name": str(sample_name),
                    "image_path": image_path,
                    "ground_truth_path": ground_truth_path,
                    "mask_view_path": mask_view_path,
                    "width": int(width),
                    "height": int(height),
                }
            )

        if not samples:
            raise ValueError("沒有完整樣本可輸出")
        return samples
