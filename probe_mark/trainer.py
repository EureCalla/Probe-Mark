import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from logger import Logger
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        opt: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        logger: Logger,
        val_loader: DataLoader = None,
    ):
        """Initialize Trainer with model, data loaders, and logger."""
        self.epochs_run = 0
        self.lr = opt.lr
        self.device = opt.device
        self.eta_min = opt.eta_min
        self.max_epochs = opt.max_epochs
        self.log_dir = opt.log_dir

        self.model = model.to(self.device)  # Send model to device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)  # Loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Optimizer
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=self.eta_min
        )  # LR scheduler
        # metrics
        self.binary_AP = BinaryAveragePrecision().to(self.device)
        self.mIoU = MeanIoU(num_classes=1).to(self.device)

        self.logger = logger  # Logger instance
        # self.prof = logger.prof  # Profiler

        if opt.resume:
            self._load_snapshot(opt.snapshot_path)  # Load training snapshot

    def metrics(self, prefix: str, epoch: int):
        self.logger.add_scalar(
            f"{prefix}/Binary AP",
            self.binary_AP.compute().item(),
            epoch,
        )
        self.logger.add_scalar(
            f"{prefix}/Mean IoU",
            self.mIoU.compute().item(),
            epoch,
        )

        self.binary_AP.reset()
        self.mIoU.reset()

    def _load_snapshot(self, path: str):
        """Load training state from snapshot."""
        snapshot = torch.load(path, map_location=self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.lr_scheduler.load_state_dict(snapshot["LR_SCHEDULER"])

    def _save_snapshot(self, epoch: int):
        """Save current training state as snapshot."""
        snapshot = {
            "EPOCHS_RUN": epoch,
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict(),
            "LR_SCHEDULER": self.lr_scheduler.state_dict(),
        }
        torch.save(snapshot, os.path.join(self.log_dir, "snapshot.pt"))

    def train_one_epoch(self, epoch: int):
        """Train model for one epoch."""
        self.model.train()
        loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch:2d}")
        for source, targets in pbar:
            source, targets = source.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(source)
            loss_batch = self.criterion(output, targets.float())
            loss_batch.backward()
            self.optimizer.step()

            self.binary_AP.update(output, targets)
            output = torch.sigmoid(output)
            self.mIoU((output > 0.5).long(), targets.long())
            loss += loss_batch.item()
            pbar.set_postfix(
                Loss=f"{loss_batch.item():.4f}",
                Accuracy=f"{100 * self.binary_AP.compute().item():.2f}%",
            )

        self.logger.add_scalar("Train/Loss", loss / len(self.train_loader), epoch)
        self.logger.add_scalar(
            "Learning Rate", self.optimizer.param_groups[0]["lr"], epoch
        )
        self.metrics("Train", epoch)

    def validate(self, epoch: int):
        """Validate model on validation set."""
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validate Epoch {epoch:2d}")
            for source, targets in pbar:
                source, targets = source.to(self.device), targets.to(self.device)
                output = self.model(source)
                loss = self.criterion(output, targets.float())

                self.binary_AP.update(output, targets)
                output = torch.sigmoid(output)
                self.mIoU((output > 0.5).long(), targets.long())
                pbar.set_description(
                    f"Accuracy {100 * self.binary_AP.compute().item():.2f}%"
                )

        val_acc = 100 * self.binary_AP.compute().item()
        self.logger.add_scalar("Validate/Loss", loss, epoch)
        self.metrics("Validate", epoch)
        return val_acc

    def run(self):
        """Run training and validation over epochs."""
        best_val_acc = 0.0
        # self.prof.start()
        for epoch in range(self.epochs_run, self.max_epochs):
            # self.prof.step()
            self.train_one_epoch(epoch)
            self.lr_scheduler.step()  # Update learning rate
            if self.val_loader is not None:
                val_acc = self.validate(epoch)
            self._save_snapshot(epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), os.path.join(self.log_dir, "best_model.pt")
                )

        # self.prof.stop()
