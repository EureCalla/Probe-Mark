import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_MARK_DIR = REPO_ROOT / "probe_mark"
if str(PROBE_MARK_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_MARK_DIR))

from database import DBManager

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _core_predictor_class():
    from predictor import Predictor as CorePredictor

    return CorePredictor


def compute_parameter_count_for_model(model, gpu_id=-1):
    model_path = model.get("best_model_path")
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型權重：{model_path}")
    opt = SimpleNamespace(
        device=f"cuda:{gpu_id}" if int(gpu_id) >= 0 else "cpu",
        output_dir=os.getcwd(),
        decoder_name=model["decoder_name"],
        encoder_name=model["encoder_name"],
        load_model_path=os.path.abspath(model_path),
    )
    core = _core_predictor_class()(opt)
    return int(sum(param.numel() for param in core.model.parameters()))


class Predictor:
    def __init__(
        self,
        model_id,
        input_path=None,
        output_dir=None,
        gpu_id=-1,
        input_mode="path",
        db=None,
    ):
        self.model_id = int(model_id)
        self.input_path = input_path
        self.output_dir = output_dir
        self.gpu_id = int(gpu_id)
        self.input_mode = input_mode
        self.db = db or DBManager()

    def _iter_images(self):
        if not self.input_path:
            raise ValueError("請先選擇 input")
        path = Path(self.input_path)
        if path.is_file():
            return [str(path)]
        if path.is_dir():
            return [
                str(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
        raise FileNotFoundError(f"找不到可預測的 input：{self.input_path}")

    def _iter_db_samples(self):
        samples = self.db.list_db_import_samples_for_model(self.model_id)
        if not samples:
            raise ValueError("資料庫匯入沒有可預測的 active samples")
        return samples

    def _make_opt(self, model):
        model_path = model.get("best_model_path")
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到模型權重：{model_path}")

        output_dir = self.output_dir
        if output_dir is None:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            output_dir = (
                REPO_ROOT
                / "data"
                / "processed"
                / "predictions"
                / f"model_{self.model_id}_{timestamp}"
            )
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        return SimpleNamespace(
            device=f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu",
            output_dir=output_dir,
            decoder_name=model["decoder_name"],
            encoder_name=model["encoder_name"],
            load_model_path=os.path.abspath(model_path),
        )

    @staticmethod
    def _read_grayscale_image(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _compute_iou(mask_path, ground_truth_path):
        pred = Predictor._read_grayscale_image(mask_path)
        target = Predictor._read_grayscale_image(ground_truth_path)
        if pred is None:
            raise FileNotFoundError(f"讀取預測 mask 失敗：{mask_path}")
        if target is None:
            raise FileNotFoundError(f"讀取 ground truth 失敗：{ground_truth_path}")
        if pred.shape != target.shape:
            target = cv2.resize(
                target,
                (pred.shape[1], pred.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        pred_bin = pred > 0
        target_bin = target > 0
        union = np.logical_or(pred_bin, target_bin).sum()
        if union == 0:
            return 1.0
        intersection = np.logical_and(pred_bin, target_bin).sum()
        return float(intersection / union)

    @staticmethod
    def _count_parameters(model):
        return int(sum(param.numel() for param in model.parameters()))

    @staticmethod
    def _excel_value(value):
        if value is None:
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _short_error(exc):
        message = str(exc).strip()
        if not message:
            message = type(exc).__name__
        return f"{type(exc).__name__}: {message}"

    @staticmethod
    def _expected_output_paths(output_dir, image_path):
        stem = Path(image_path).stem
        paths = {
            "mask_path": os.path.join(output_dir, f"{stem}_mask.png"),
            "overlay_path": os.path.join(output_dir, f"{stem}_overlay.png"),
            "compare_path": os.path.join(output_dir, f"{stem}_compare.png"),
        }
        return {
            key: value if os.path.exists(value) else ""
            for key, value in paths.items()
        }

    @classmethod
    def _write_db_import_summary(cls, output_dir, prediction_run_id, db_samples, outputs):
        summary_path = os.path.join(output_dir, "test_summary.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "test_summary"
        headers = [
            "row_no",
            "sample_id",
            "sample_name",
            "dataset_id",
            "dataset_name",
            "image_filename",
            "status",
            "error_message",
            "iou",
            "area_px",
            "h_feret_px",
            "v_feret_px",
            "area_um2",
            "h_feret_um",
            "v_feret_um",
            "image_path",
            "ground_truth_path",
            "mask_path",
            "overlay_path",
            "compare_path",
        ]
        ws.append(headers)

        for row_no, (sample, output) in enumerate(zip(db_samples, outputs), start=1):
            measure = output.get("measure") or {}
            ws.append(
                [
                    row_no,
                    sample.get("id"),
                    sample.get("sample_name"),
                    sample.get("dataset_id"),
                    sample.get("dataset_name"),
                    os.path.basename(output.get("image_path") or sample.get("image_path") or ""),
                    output.get("status", "done"),
                    output.get("error_message"),
                    cls._excel_value(output.get("iou")),
                    cls._excel_value(measure.get("area_px")),
                    cls._excel_value(measure.get("h_feret_px")),
                    cls._excel_value(measure.get("v_feret_px")),
                    cls._excel_value(measure.get("area_um2")),
                    cls._excel_value(measure.get("h_feret_um")),
                    cls._excel_value(measure.get("v_feret_um")),
                    output.get("image_path") or sample.get("image_path"),
                    sample.get("ground_truth_path"),
                    output.get("mask_path"),
                    output.get("overlay_path"),
                    output.get("compare_path"),
                ]
            )

        for column_cells in ws.columns:
            max_length = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 10), 60)
        ws.freeze_panes = "A2"

        meta = wb.create_sheet("metadata")
        meta.append(["prediction_run_id", prediction_run_id])
        meta.append(["n_samples", len(outputs)])
        meta.append(["n_done", sum(1 for output in outputs if output.get("status") == "done")])
        meta.append(["n_failed", sum(1 for output in outputs if output.get("status") == "failed")])
        ious = [
            output["iou"]
            for output in outputs
            if output.get("status") == "done" and output.get("iou") is not None
        ]
        meta.append(["mean_iou", float(np.mean(ious)) if ious else None])

        wb.save(summary_path)
        return summary_path

    def run(self):
        self.db.init_db()
        model = self.db.get_model(self.model_id)
        if model is None:
            raise ValueError(f"找不到 model：id={self.model_id}")

        db_samples = []
        if self.input_mode == "database":
            db_samples = self._iter_db_samples()
            images = [sample["image_path"] for sample in db_samples]
        else:
            images = self._iter_images()
            if not images:
                raise ValueError("預測資料夾內沒有可用影像")

        opt = self._make_opt(model)
        CorePredictor = _core_predictor_class()
        mode = (
            "database"
            if self.input_mode == "database"
            else ("folder" if os.path.isdir(self.input_path) else "single")
        )
        run_input_path = f"db:model:{self.model_id}" if mode == "database" else self.input_path
        prediction_run_id = self.db.insert_prediction_run(
            model_id=self.model_id,
            input_path=run_input_path,
            output_dir=opt.output_dir,
            mode=mode,
        )

        try:
            core = CorePredictor(opt)
            parameter_count = self._count_parameters(core.model)
            if model.get("parameter_count") != parameter_count:
                self.db.update_model_parameter_count(self.model_id, parameter_count)

            outputs = []
            for index, image_path in enumerate(images):
                if mode == "database":
                    sample = db_samples[index]
                    try:
                        result = core.predict(image_path)
                        iou = self._compute_iou(
                            result["mask_path"],
                            sample["ground_truth_path"],
                        )
                        output = dict(
                            result,
                            image_path=image_path,
                            iou=iou,
                            status="done",
                            error_message=None,
                        )
                    except Exception as exc:
                        error_message = self._short_error(exc)
                        paths = self._expected_output_paths(opt.output_dir, image_path)
                        output = dict(
                            image_path=image_path,
                            iou=None,
                            measure=None,
                            status="failed",
                            error_message=error_message,
                            **paths,
                        )
                        print(
                            "[Predict][SKIP] "
                            f"{index + 1}/{len(images)} sample_id={sample.get('id')} "
                            f"sample={sample.get('sample_name')} error={error_message}"
                        )
                    self.db.insert_prediction_output(
                        prediction_run_id=prediction_run_id,
                        image_path=image_path,
                        mask_path=output.get("mask_path") or "",
                        compare_path=output.get("compare_path") or "",
                        overlay_path=output.get("overlay_path") or None,
                        iou=output.get("iou"),
                        status=output.get("status", "done"),
                        error_message=output.get("error_message"),
                    )
                    outputs.append(output)
                    continue

                result = core.predict(image_path)
                self.db.insert_prediction_output(
                    prediction_run_id=prediction_run_id,
                    image_path=image_path,
                    mask_path=result["mask_path"],
                    compare_path=result["compare_path"],
                    overlay_path=result["overlay_path"],
                    iou=None,
                )
                outputs.append(
                    dict(result, image_path=image_path, iou=None, status="done", error_message=None)
                )
            self.db.update_prediction_run(prediction_run_id, "done")

            summary_path = None
            if mode == "database":
                ious = [
                    output["iou"]
                    for output in outputs
                    if output.get("status") == "done" and output.get("iou") is not None
                ]
                mean_iou = float(np.mean(ious)) if ious else None
                self.db.update_model_db_import_metrics(
                    self.model_id,
                    mean_iou,
                    len(ious),
                    prediction_run_id,
                )
                summary_path = self._write_db_import_summary(
                    opt.output_dir,
                    prediction_run_id,
                    db_samples,
                    outputs,
                )
        except Exception:
            self.db.update_prediction_run(prediction_run_id, "failed")
            raise

        print(f"[Predict] prediction_run_id={prediction_run_id}, outputs={len(outputs)}")
        return {
            "prediction_run_id": prediction_run_id,
            "output_dir": opt.output_dir,
            "outputs": outputs,
            "summary_path": summary_path,
        }
