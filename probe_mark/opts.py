import argparse
import os
import time


class opts(object):
    def __init__(self):
        # Initialize an ArgumentParser to handle command-line arguments
        self.parser = argparse.ArgumentParser(
            description="Script for binary classification tasks"
        )

        """Basic experiment settings"""
        self.parser.add_argument("--exp_id", default="default")  # Experiment name
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="data/processed",
            help="dataset directory",  # Dataset directory
        )
        self.parser.add_argument(
            "--seed", type=int, default=42, help="random seed"  # Random seed
        )

        """System settings"""
        self.parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,  # Default GPU is 0
            help="-1 for CPU, use comma for multiple GPUs, e.g., 0,1",  # Specify GPU IDs, -1 for CPU
        )

        """Model settings"""
        self.parser.add_argument(
            "--encoder_name",
            type=str,
            default="resnet18",
            help="Name of the encoder model. See https://github.com/qubvel-org/segmentation_models.pytorch/tree/main?tab=readme-ov-file#encoders for more options.",
        )
        self.parser.add_argument(
            "--encoder_weights",
            type=str,
            default="imagenet",
            choices=["imagenet", "ssl", "swsl", None],
            help="Pretrained weights for the encoder.",
        )
        self.parser.add_argument(
            "--decoder_name",
            type=str,
            default="FPN",
            # fmt: off
            choices=[
                'Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 
                'PSPNet', 'DeepLabV3', 'DeepLabV3Plus', 'PAN'
            ],
            # fmt: on
            help="Name of the decoder (model architecture).",
        )

        """Training hyperparameters"""
        self.parser.add_argument(
            "--max_epochs",
            type=int,
            default=10,
            help="max epochs to train",  # Maximum number of training epochs
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=16, help="batch size"  # Batch size
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="number of workers",  # Number of workers for data loading
        )
        self.parser.add_argument(
            "--lr", type=float, default=1e-4, help="learning rate"  # Learning rate
        )
        self.parser.add_argument(
            "--eta_min",
            type=float,
            default=1e-5,
            help="minimum learning rate",  # Minimum learning rate
        )

        """Resume training settings"""
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="resume an experiment",  # Resume training from a checkpoint
        )
        self.parser.add_argument(
            "--snapshot_path",
            type=str,
            default=None,
            help="path to the snapshot to resume",  # Path to the checkpoint for resuming
        )

        """Testing settings"""
        self.parser.add_argument("--test", action="store_true")  # Run in test mode
        self.parser.add_argument(
            "--load_model_path",
            type=str,
            default=None,
            help="path to trained model for testing",  # Path to the model for testing
        )
        self.parser.add_argument(
            "--image_path",
            type=str,
            default=None,
            help="path to image for testing",  # Path to the image for testing
        )

    def parse(self, args=""):
        # Parse command-line arguments
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # Set root directory
        opt.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Set device based on gpu_id, use CPU if gpu_id is -1
        opt.device = f"cuda:{opt.gpu_id}" if opt.gpu_id >= 0 else "cpu"

        # Check if the dataset directory exists, raise error if not
        opt.data_dir = os.path.join(opt.root_dir, opt.data_dir)
        if not os.path.exists(opt.data_dir):
            raise ValueError(f"Data directory {opt.data_dir} not found")

        # Set logging directory
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        opt.log_dir = os.path.join(opt.root_dir, "runs", opt.exp_id, f"{time_str}")

        # Create logging directory if not in test or resume mode
        if not opt.test and not opt.resume:
            if not os.path.exists(opt.log_dir):
                os.makedirs(opt.log_dir, exist_ok=True)
                print(f"Created directory {opt.log_dir}")

        # Handle resume training case
        if opt.resume:
            snapshot_path = os.path.join(opt.root_dir, opt.snapshot_path)
            if not os.path.isfile(snapshot_path) or not snapshot_path.endswith(
                "snapshot.pt"
            ):  # Check if snapshot file exists and is valid
                raise ValueError(
                    f"Snapshot file {snapshot_path} is not a valid .pt file or does not exist"
                )
            opt.snapshot_path = snapshot_path
            # Use the existing logging directory
            opt.log_dir = os.path.dirname(snapshot_path)

        # Handle test case
        if opt.test:
            load_model_path = os.path.join(opt.root_dir, opt.load_model_path)
            if not os.path.isfile(load_model_path) or not load_model_path.endswith(
                ".pt"
            ):  # Check if model file exists and is valid
                raise FileNotFoundError(
                    f"Model not found at {opt.load_model_path} for testing"
                )
            opt.load_model_path = load_model_path

        return opt
