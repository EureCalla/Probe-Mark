import torch
from torchvision.transforms import v2

BASIC_TRANSFORMS = v2.Compose(
    [
        v2.ToImage(),
        v2.Pad((0, 129, 0, 129), fill=0, padding_mode="constant"),
        v2.Resize(size=(256, 256)),
        v2.ToDtype(torch.float32),
        # v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

AUGMENTATION_TRANSFORMS = v2.Compose(
    [
        v2.ToImage(),
        v2.Pad((0, 129, 0, 129), fill=0, padding_mode="constant"),
        v2.Resize(size=(256, 256)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAdjustSharpness(sharpness_factor=4, p=0.5),
        v2.RandomAutocontrast(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
