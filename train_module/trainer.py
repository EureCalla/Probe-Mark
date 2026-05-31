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
        dataset_id,
        split_id=None,
        split_name="default",
        run_name=None,
        seed=42,
        val_ratio=0.2,
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
        self.dataset_id = int(dataset_id)
        self.split_id = int(split_id) if split_id else None
        self.split_name = split_name
        self.run_name = run_name or f"dataset_{dataset_id}"
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
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

        samples = self.db.list_samples(self.dataset_id)
        if len(samples) < 2:
            raise ValueError("至少需要 2 個 sample 才能建立 train/val split")

        train_samples, val_samples = train_test_split(
            samples,
            test_size=self.val_ratio,
            random_state=self.seed,
            shuffle=True,
        )
        train_ids = [s["id"] for s in train_samples]
        val_ids = [s["id"] for s in val_samples]
        return self.db.create_split(
            dataset_id=self.dataset_id,
            split_name=self.split_name,
            train_ids=train_ids,
            val_ids=val_ids,
            seed=self.seed,
            train_ratio=1.0 - self.val_ratio,
            val_ratio=self.val_ratio,
            test_ratio=0.0,
        )

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
        dataset = self.db.get_dataset(self.dataset_id)
        if dataset is None or dataset["status"] != "done":
            raise ValueError(f"找不到可訓練 dataset：id={self.dataset_id}")

        split_id = self._ensure_split()
        train_samples = self.db.get_split_samples(split_id, "train")
        val_samples = self.db.get_split_samples(split_id, "val")
        if not train_samples or not val_samples:
            raise ValueError("split 必須同時包含 train 與 val samples")

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

            Model = getattr(smp, opt.decoder_name)
            model = Model(
                encoder_name=opt.encoder_name,
                encoder_weights=opt.encoder_weights,
                in_channels=3,
                classes=1,
            )
            logger = Logger(opt)
            core = CoreTrainer(opt, model, train_loader, logger, val_loader)
            core.run()
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
