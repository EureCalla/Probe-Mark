import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from datasets.transforms import PREDICT_TRANSFORMS


class Predictor:
    def __init__(self, opt: argparse.Namespace):
        self.device = opt.device
        self.output_dir = opt.output_dir
        self.transform = PREDICT_TRANSFORMS()

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

    def _postprocess(self, logits: torch.Tensor) -> np.ndarray:
        binary = (torch.sigmoid(logits) > 0.5).squeeze()
        return binary.cpu().numpy().astype(np.uint8) * 255

    def _save_mask(self, mask_np: np.ndarray, stem: str) -> str:
        out_path = os.path.join(self.output_dir, f"{stem}_mask.png")
        Image.fromarray(mask_np, mode="L").save(out_path)
        return out_path

    def _save_compare(self, original_pil: Image.Image, mask_np: np.ndarray, stem: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(original_pil)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        fig.suptitle(stem, fontsize=12)
        fig.tight_layout()

        out_path = os.path.join(self.output_dir, f"{stem}_compare.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def predict(self, image_path: str) -> dict:
        """Run inference on a single image, save mask PNG and comparison figure."""
        stem = Path(image_path).stem

        tensor, original_pil = self._preprocess(image_path)

        with torch.no_grad():
            logits = self.model(tensor)

        mask_np = self._postprocess(logits)

        mask_path = self._save_mask(mask_np, stem)
        compare_path = self._save_compare(original_pil, mask_np, stem)

        print(f"Mask saved:    {mask_path}")
        print(f"Compare saved: {compare_path}")

        return {"mask_path": mask_path, "compare_path": compare_path}
