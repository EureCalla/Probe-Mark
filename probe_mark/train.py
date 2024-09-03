import argparse

import segmentation_models_pytorch as smp
from datasets import (
    AUGMENTATION_TRANSFORMS,
    BASIC_TRANSFORMS,
    SegmentationBinaryDataset,
)
from logger import Logger
from opts import opts
from torch.utils.data import DataLoader
from trainer import Trainer
from utils import seed_everything


def main(opt: argparse.Namespace, logger: Logger):
    seed_everything(opt.seed)  # Set random seed
    # Load training dataset with augmentations
    train_dataset = SegmentationBinaryDataset(
        root=opt.data_dir,
        is_train=True,
        seed=opt.seed,
        transform=AUGMENTATION_TRANSFORMS(),
    )
    # Load validation dataset without augmentations
    val_dataset = SegmentationBinaryDataset(
        root=opt.data_dir,
        is_train=False,
        seed=opt.seed,
        transform=BASIC_TRANSFORMS(),
    )
    # Training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    # Validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    # Create model
    Model = getattr(smp, opt.decoder_name)
    model = Model(
        encoder_name=opt.encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=opt.encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    # Initialize trainer
    trainer = Trainer(opt, model, train_loader, logger, val_loader)
    trainer.run()  # Start training and validation
    logger.close()  # Close logger


if __name__ == "__main__":
    opt = opts().parse(
        [
            # "--resume",
            # "--snapshot_path",
            # "runs/.../best_model.pt",
        ]
    )
    logger = Logger(opt)
    main(opt, logger)
