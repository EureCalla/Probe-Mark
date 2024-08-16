import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask


class SegmentationBinaryDataset(Dataset):
    def __init__(self, root: Path, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        # TODO(hcchen): Add file mapping for different dataset
        self.file_mapping = {"image": "image.png", "label": "mask.png"}

        samples = self._make_dataset(directory=self.root)

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _make_dataset(self, directory: Path) -> List[Tuple[str, str]]:
        directory = os.path.expanduser(directory)

        instances = []
        for root, _, fnames in os.walk(directory):
            if all(file in fnames for file in self.file_mapping.values()):
                paths = {
                    key: os.path.join(root, fname)
                    for key, fname in self.file_mapping.items()
                }
                instances.append((paths["image"], paths["label"]))
        return instances

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.transform:
            label = Mask(label)
            image, label = self.transform(image, label)

        return image, label
