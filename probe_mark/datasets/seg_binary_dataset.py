import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image as Image
import torchvision.tv_tensors as tv_tensors
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SegmentationBinaryDataset(Dataset):
    def __init__(
        self,
        root: Path,
        is_train: bool = True,
        seed: int = 42,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.transform = transform
        # TODO(hcchen): Add file mapping for different dataset
        self.file_mapping = {"image": "image.png", "label": "ground_truth.png"}

        samples = self._make_dataset(directory=self.root)

        # TODO(hcchen): Add stratified sampling, default to 8:2 split
        train_samples, test_samples = train_test_split(
            samples, test_size=0.2, random_state=seed
        )
        samples = train_samples if is_train else test_samples

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
            label = tv_tensors.Mask(label)
            image, label = self.transform(image, label)

        return image, label
