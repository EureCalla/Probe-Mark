import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import segmentation_models_pytorch as smp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_MARK_DIR = REPO_ROOT / "probe_mark"
if str(PROBE_MARK_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_MARK_DIR))

from database import DBManager
from datasets import (
    AUGMENTATION_TRANSFORMS,
    BASIC_TRANSFORMS,
    SampleListSegmentationDataset,
)
from logger import Logger
from trainer import Trainer as CoreTrainer
from utils import seed_everything


class Trainer:
    def __init__(
        self,
        dataset_id=None,
        dataset_ids=None,
        validation_dataset_ids=None,
        split_id=None,
        split_name="default",
        run_name=None,
        seed=42,
        val_ratio=0.2,
        test_ratio=0.2,
        encoder_name="resnet18",
        encoder_weights="imagenet",
        decoder_name="FPN",
        max_epochs=10,
        batch_size=16,
        num_workers=0,
        lr=1e-4,
        eta_min=1e-5,
        gpu_id=0,
        db=None,
    ):
        if dataset_ids is None:
            dataset_ids = [dataset_id]
        self.train_dataset_ids = [int(item) for item in dataset_ids if item is not None]
        self.validation_dataset_ids = [
            int(item) for item in (validation_dataset_ids or []) if item is not None
        ]
        self.dataset_ids = self.train_dataset_ids + self.validation_dataset_ids
        if not self.dataset_ids:
            raise ValueError("至少需要選擇一個 dataset")
        self.dataset_id = self.train_dataset_ids[0] if self.train_dataset_ids else self.dataset_ids[0]
        self.split_id = int(split_id) if split_id else None
        self.split_name = split_name
        self.run_name = run_name or self.default_run_name()
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.decoder_name = decoder_name
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.lr = float(lr)
        self.eta_min = float(eta_min)
        self.gpu_id = int(gpu_id)
        self.db = db or DBManager()

    def _ensure_split(self):
        if self.split_id:
            return self.split_id

        train_pool = self.list_samples_for_datasets(self.train_dataset_ids)
        val_samples = self.list_samples_for_datasets(self.validation_dataset_ids)
        if len(train_pool) < 2:
            raise ValueError("train pool 至少需要 2 個有效 sample 才能切 train/test")
        if not val_samples:
            raise ValueError("validation datasets 至少需要 1 個有效 sample")

        train_samples, test_samples = train_test_split(
            train_pool,
            test_size=self.test_ratio,
            random_state=self.seed,
            shuffle=True,
        )
        if not train_samples or not test_samples:
            raise ValueError("train/test split 必須同時包含 train 與 test samples")

        total_samples = len(train_samples) + len(val_samples) + len(test_samples)
        train_ratio = len(train_samples) / total_samples
        val_ratio = len(val_samples) / total_samples
        test_ratio = len(test_samples) / total_samples
        train_ids = [s["id"] for s in train_samples]
        val_ids = [s["id"] for s in val_samples]
        test_ids = [s["id"] for s in test_samples]
        return self.db.create_split(
            dataset_id=self.dataset_id,
            split_name=self.split_name,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            seed=self.seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    def default_run_name(self):
        if self.validation_dataset_ids:
            return f"train_{len(self.train_dataset_ids)}_val_{len(self.validation_dataset_ids)}"
        if len(self.train_dataset_ids) == 1:
            return f"dataset_{self.dataset_id}"
        return f"datasets_{len(self.train_dataset_ids)}"

    def list_samples_for_datasets(self, dataset_ids):
        samples = []
        for dataset_id in dataset_ids:
            samples.extend(self.db.list_samples(dataset_id))
        return samples

    def _make_opt(self, run_id):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        exp_id = self.run_name or f"run_{run_id}"
        log_dir = REPO_ROOT / "runs" / exp_id / f"run_{run_id}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            root_dir=str(REPO_ROOT),
            data_dir=None,
            seed=self.seed,
            gpu_id=self.gpu_id,
            device=f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu",
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            decoder_name=self.decoder_name,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            lr=self.lr,
            eta_min=self.eta_min,
            resume=False,
            snapshot_path=None,
            test=False,
            load_model_path=None,
            image_path=None,
            output_dir=None,
            log_dir=str(log_dir),
            exp_id=exp_id,
        )

    def run(self):
        self.db.init_db()
        for dataset_id in self.dataset_ids:
            dataset = self.db.get_dataset(dataset_id)
            if dataset is None or dataset["status"] != "done":
                raise ValueError(f"找不到可訓練 dataset：id={dataset_id}")

        split_id = self._ensure_split()
        train_samples = self.db.get_split_samples(split_id, "train")
        val_samples = self.db.get_split_samples(split_id, "val")
        test_samples = self.db.get_split_samples(split_id, "test")
        if not train_samples or not val_samples or not test_samples:
            raise ValueError("split 必須同時包含 train、val 與 test samples")

        run_id = self.db.insert_training_run(
            run_name=self.run_name,
            dataset_id=self.dataset_id,
            split_id=split_id,
            encoder_name=self.encoder_name,
            decoder_name=self.decoder_name,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        self.db.replace_training_run_params(run_id, self.training_params())
        opt = self._make_opt(run_id)

        try:
            seed_everything(opt.seed)
            train_dataset = SampleListSegmentationDataset(
                train_samples,
                transform=AUGMENTATION_TRANSFORMS(),
            )
            val_dataset = SampleListSegmentationDataset(
                val_samples,
                transform=BASIC_TRANSFORMS(),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            test_dataset = SampleListSegmentationDataset(
                test_samples,
                transform=BASIC_TRANSFORMS(),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                pin_memory=True,
                shuffle=False,
            )

            Model = getattr(smp, opt.decoder_name)
            model = Model(
                encoder_name=opt.encoder_name,
                encoder_weights=opt.encoder_weights,
                in_channels=3,
                classes=1,
            )
            logger = Logger(opt)
            core = CoreTrainer(opt, model, train_loader, logger, val_loader)
            epoch_metrics = core.run()
            self.db.replace_epoch_metrics(run_id, epoch_metrics)
            test_metrics = core.evaluate_loader(test_loader, "Test", epoch=self.max_epochs)
            self.db.insert_test_metrics(
                run_id,
                split_id,
                {
                    "n_samples": len(test_samples),
                    "test_loss": test_metrics["loss"],
                    "test_iou": test_metrics["iou"],
                    "test_dice": test_metrics["dice"],
                    "test_ap": test_metrics["ap"],
                },
            )
            logger.close()

            best_model_path = os.path.join(opt.log_dir, "best_model.pt")
            snapshot_path = os.path.join(opt.log_dir, "snapshot.pt")
            opt_path = os.path.join(opt.log_dir, "opt.txt")
            if not os.path.exists(best_model_path):
                torch.save(core.model.state_dict(), best_model_path)
            self.db.update_training_run(run_id, status="done")
            model_id = self.db.insert_model(
                run_id=run_id,
                model_dir=opt.log_dir,
                best_model_path=best_model_path if os.path.exists(best_model_path) else None,
                snapshot_path=snapshot_path if os.path.exists(snapshot_path) else None,
                opt_path=opt_path if os.path.exists(opt_path) else None,
            )
        except Exception:
            self.db.update_training_run(run_id, status="failed")
            raise

        print(f"[Train] 完成 run_id={run_id}, model_id={model_id}")
        return run_id

    def training_params(self):
        return {
            "train_dataset_ids": self.train_dataset_ids,
            "validation_dataset_ids": self.validation_dataset_ids,
            "split_name": self.split_name,
            "seed": self.seed,
            "test_ratio": self.test_ratio,
            "run_name": self.run_name,
            "encoder_name": self.encoder_name,
            "encoder_weights": self.encoder_weights,
            "decoder_name": self.decoder_name,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "lr": self.lr,
            "eta_min": self.eta_min,
            "gpu_id": self.gpu_id,
        }
