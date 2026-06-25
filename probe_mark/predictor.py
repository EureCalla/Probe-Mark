import argparse
import os
from pathlib import Path

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import PIL.Image as Image
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from datasets.transforms import PREDICT_TRANSFORMS

# 前處理在原圖上下各補 129px（見 datasets/transforms.py），推論還原時需對齊。
PAD_TOP = 129
PAD_BOTTOM = 129

# 像素->µm 參考比例：很多機台為 1 px = 0.093555 µm，但「並非每一台都相同」，
# 實際換算比例必須依影像來源機台個別確認。預設 None（只標示 pixel），
# 確認後才透過 opt.scale_um_per_px 傳入啟用 µm 換算。此常數僅供參考、不自動套用。
DEFAULT_UM_PER_PX = 0.093555


class Predictor:
    def __init__(self, opt: argparse.Namespace):
        self.device = opt.device
        self.output_dir = opt.output_dir
        self.transform = PREDICT_TRANSFORMS()
        # 預設 None：只標示 pixel。確認該機台比例後再傳入 µm/px 啟用換算。
        self.scale_um_per_px = getattr(opt, "scale_um_per_px", None)

        Model = getattr(smp, opt.decoder_name)
        self.model: nn.Module = Model(
            encoder_name=opt.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        state_dict = torch.load(opt.load_model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image_path: str) -> tuple:
        pil_image = Image.open(image_path).convert("RGB")
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor, pil_image

    def _logits_to_prob(self, logits: torch.Tensor) -> np.ndarray:
        """Return the continuous sigmoid probability map (224x224, float32, 0~1).

        不在低解析度上做 threshold，保留 soft edge 供放大後抗鋸齒與 sub-pixel 量測。
        """
        prob = torch.sigmoid(logits).squeeze()
        return prob.detach().cpu().numpy().astype(np.float32)

    def _prob_to_original_size(self, prob_small: np.ndarray, original_pil: Image.Image) -> np.ndarray:
        """Bilinear-upscale the probability map back to the original image size.

        先用 bilinear 把機率圖放大回 padding 後尺寸 (W, H+258)，再裁掉上下 padding，
        幾何上與前處理的 Pad+Resize 互為反向，輸出原圖座標系的連續機率圖。
        """
        w, h = original_pil.width, original_pil.height
        padded = cv2.resize(prob_small, (w, h + PAD_TOP + PAD_BOTTOM), interpolation=cv2.INTER_LINEAR)
        return padded[PAD_TOP:PAD_TOP + h, 0:w]

    def _binarize(self, prob_full: np.ndarray) -> np.ndarray:
        """Threshold on the full-resolution prob map, then light morphology smoothing.

        回傳 0/1 uint8 遮罩。open->close 去除毛邊與小孔，邊緣比 224-NEAREST 平滑。
        """
        binary = (prob_full > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _measure(self, binary: np.ndarray) -> dict | None:
        """Measure the largest contour on the full-resolution mask.

        對應 Excel「Volume & area」欄位：截面積、水平/垂直弗里特直徑（bounding box 寬/高）。
        一律先以 pixel 標示；若 self.scale_um_per_px 有值（已確認該機台比例）才附帶 µm。
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        measure = {
            "area_px": float(cv2.contourArea(cnt)),
            "h_feret_px": float(bw),  # 水平弗里特直徑
            "v_feret_px": float(bh),  # 垂直弗里特直徑
            "bbox": (int(x), int(y), int(bw), int(bh)),
        }
        if len(cnt) >= 5:
            (_, _), (axis1, axis2), _ = cv2.fitEllipse(cnt)
            measure["ellipse_major_px"] = float(max(axis1, axis2))
            measure["ellipse_minor_px"] = float(min(axis1, axis2))

        # µm 換算：僅在比例尺已確認時啟用（面積按 scale²，長度按 scale）。
        scale = self.scale_um_per_px
        if scale:
            measure["um_per_px"] = float(scale)
            measure["area_um2"] = measure["area_px"] * scale * scale
            measure["h_feret_um"] = measure["h_feret_px"] * scale
            measure["v_feret_um"] = measure["v_feret_px"] * scale
            if "ellipse_major_px" in measure:
                measure["ellipse_major_um"] = measure["ellipse_major_px"] * scale
                measure["ellipse_minor_um"] = measure["ellipse_minor_px"] * scale
        return measure

    def _measure_caption(self, measure: dict) -> str:
        """Build a human-readable label for the measurement (pixel, + µm if scaled)."""
        if not measure:
            return "no contour found"
        scale = measure.get("um_per_px")
        if scale:
            return (
                f"area: {measure['area_px']:.0f} px^2 = {measure['area_um2']:.2f} um^2\n"
                f"H feret: {measure['h_feret_px']:.0f} px = {measure['h_feret_um']:.2f} um\n"
                f"V feret: {measure['v_feret_px']:.0f} px = {measure['v_feret_um']:.2f} um\n"
                f"(scale {scale:g} um/px)"
            )
        return (
            f"area: {measure['area_px']:.0f} px^2\n"
            f"H feret: {measure['h_feret_px']:.0f} px\n"
            f"V feret: {measure['v_feret_px']:.0f} px\n"
            f"(pixel only - um scale not set)"
        )

    def _save_mask(self, mask_np: np.ndarray, stem: str) -> str:
        out_path = os.path.join(self.output_dir, f"{stem}_mask.png")
        Image.fromarray(mask_np, mode="L").save(out_path)
        return out_path

    def _save_compare(self, original_pil: Image.Image, mask_np: np.ndarray, stem: str, measure: dict | None = None) -> str:
        fig = Figure(figsize=(10, 5.6))
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)

        axes[0].imshow(original_pil)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        # 在遮罩上畫出水平/垂直弗里特直徑線，並於圖下標示量測值。
        if measure and "bbox" in measure:
            x, y, bw, bh = measure["bbox"]
            axes[1].plot([x, x + bw], [y + bh / 2, y + bh / 2], color="red", lw=1.2)
            axes[1].plot([x + bw / 2, x + bw / 2], [y, y + bh], color="deepskyblue", lw=1.2)
            axes[1].text(
                0.5, -0.04, self._measure_caption(measure),
                transform=axes[1].transAxes, ha="center", va="top", fontsize=9,
            )

        fig.suptitle(stem, fontsize=12)
        fig.tight_layout()

        out_path = os.path.join(self.output_dir, f"{stem}_compare.png")
        canvas.print_figure(out_path, dpi=150, bbox_inches="tight")
        fig.clear()
        return out_path

    def _save_overlay(self, original_pil: Image.Image, mask_np: np.ndarray, stem: str) -> str:
        """Overlay the (already full-resolution, smoothed) mask onto the original image."""
        out_path = os.path.join(self.output_dir, f"{stem}_overlay.png")
        original_rgba = original_pil.convert("RGBA")
        mask = Image.fromarray(mask_np, mode="L")
        alpha = mask.point(lambda value: 96 if value > 0 else 0)
        overlay = Image.new("RGBA", original_rgba.size, (0, 255, 0, 0))
        overlay.putalpha(alpha)
        Image.alpha_composite(original_rgba, overlay).convert("RGB").save(out_path)
        return out_path

    def predict(self, image_path: str) -> dict:
        """Run inference on a single image, save mask PNG, overlay and comparison figure."""
        stem = Path(image_path).stem

        tensor, original_pil = self._preprocess(image_path)

        with torch.no_grad():
            logits = self.model(tensor)

        # 先還原解析度（bilinear），最後才二值化 —— 這是消除鋸齒的關鍵順序。
        prob_small = self._logits_to_prob(logits)
        prob_full = self._prob_to_original_size(prob_small, original_pil)
        binary = self._binarize(prob_full)
        mask_np = (binary * 255).astype(np.uint8)

        measure = self._measure(binary)

        mask_path = self._save_mask(mask_np, stem)
        overlay_path = self._save_overlay(original_pil, mask_np, stem)
        compare_path = self._save_compare(original_pil, mask_np, stem, measure)

        print(f"Mask saved:    {mask_path}")
        print(f"Overlay saved: {overlay_path}")
        print(f"Compare saved: {compare_path}")
        if measure:
            print("Measure:       " + self._measure_caption(measure).replace("\n", " | "))

        return {
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "compare_path": compare_path,
            "measure": measure,
        }
