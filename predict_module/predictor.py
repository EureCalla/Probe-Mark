import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_MARK_DIR = REPO_ROOT / "probe_mark"
if str(PROBE_MARK_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_MARK_DIR))

from database import DBManager
from predictor import Predictor as CorePredictor

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Predictor:
    def __init__(
        self,
        model_id,
        input_path,
        output_dir=None,
        gpu_id=-1,
        db=None,
    ):
        self.model_id = int(model_id)
        self.input_path = input_path
        self.output_dir = output_dir
        self.gpu_id = int(gpu_id)
        self.db = db or DBManager()

    def _iter_images(self):
        path = Path(self.input_path)
        if path.is_file():
            return [str(path)]
        if path.is_dir():
            return [
                str(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
        raise FileNotFoundError(f"找不到預測輸入：{self.input_path}")

    def _make_opt(self, model):
        model_path = model.get("best_model_path")
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到模型權重：{model_path}")

        output_dir = self.output_dir
        if output_dir is None:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            output_dir = REPO_ROOT / "data" / "processed" / "predictions" / f"model_{self.model_id}_{timestamp}"
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        return SimpleNamespace(
            device=f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu",
            output_dir=output_dir,
            decoder_name=model["decoder_name"],
            encoder_name=model["encoder_name"],
            load_model_path=os.path.abspath(model_path),
        )

    def run(self):
        self.db.init_db()
        model = self.db.get_model(self.model_id)
        if model is None:
            raise ValueError(f"找不到 model：id={self.model_id}")

        images = self._iter_images()
        if not images:
            raise ValueError("預測資料夾內沒有可用影像")

        opt = self._make_opt(model)
        mode = "folder" if os.path.isdir(self.input_path) else "single"
        prediction_run_id = self.db.insert_prediction_run(
            model_id=self.model_id,
            input_path=self.input_path,
            output_dir=opt.output_dir,
            mode=mode,
        )

        try:
            core = CorePredictor(opt)
            outputs = []
            for image_path in images:
                result = core.predict(image_path)
                self.db.insert_prediction_output(
                    prediction_run_id=prediction_run_id,
                    image_path=image_path,
                    mask_path=result["mask_path"],
                    compare_path=result["compare_path"],
                    overlay_path=result["overlay_path"],
                )
                outputs.append(dict(result, image_path=image_path))
            self.db.update_prediction_run(prediction_run_id, "done")
        except Exception:
            self.db.update_prediction_run(prediction_run_id, "failed")
            raise

        print(f"[Predict] 完成 prediction_run_id={prediction_run_id}, outputs={len(outputs)}")
        return {
            "prediction_run_id": prediction_run_id,
            "output_dir": opt.output_dir,
            "outputs": outputs,
        }
