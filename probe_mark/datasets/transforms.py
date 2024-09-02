import torch
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors


def BASIC_TRANSFORMS():
    """
    Basic image transformations: resize, convert to tensor, and normalize.
    """
    img_size: list = [224, 224]
    mean: list = [0.485, 0.456, 0.406]
    std: list = [0.229, 0.224, 0.225]

    return T.Compose(
        [
            T.ToImage(),
            T.Pad(padding=(0, 129, 0, 129), fill=0, padding_mode="constant"),
            T.Resize(size=img_size),
            T.ToDtype(
                dtype={
                    tv_tensors.Image: torch.float32,
                    tv_tensors.Mask: torch.int64,
                    "others": None,
                },
                scale=True,
            ),
            T.Normalize(mean=mean, std=std),
        ]
    )


def AUGMENTATION_TRANSFORMS():
    """
    Augmentation transformations: resize, random rotations/flips, convert to tensor, and normalize.
    """
    img_size: list = [224, 224]
    mean: list = [0.485, 0.456, 0.406]
    std: list = [0.229, 0.224, 0.225]

    return T.Compose(
        [
            T.ToImage(),
            T.Pad(padding=(0, 129, 0, 129), fill=0, padding_mode="constant"),
            T.Resize(size=img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=4, p=0.5),
            T.RandomAutocontrast(p=0.5),
            T.ToDtype(
                dtype={
                    tv_tensors.Image: torch.float32,
                    tv_tensors.Mask: torch.int64,
                    "others": None,
                },
                scale=True,
            ),
            T.Normalize(mean=mean, std=std),
        ]
    )
