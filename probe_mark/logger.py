import os
import sys

import torch
import torch.profiler as profiler
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, opt):
        """Logs training information and initializes TensorBoard."""
        file_name = os.path.join(
            opt.log_dir, "opt.txt"
        )  # opt.txt: Logs training parameters

        # Parse opt parameters
        args = dict(
            (name, getattr(opt, name)) for name in dir(opt) if not name.startswith("_")
        )
        # Write parameters to opt.txt
        with open(file_name, "wt") as opt_file:
            opt_file.write("==> torch version: {}\n".format(torch.__version__))
            opt_file.write(
                "==> cudnn version: {}\n".format(torch.backends.cudnn.version())
            )
            opt_file.write("==> Cmd:\n")
            opt_file.write(str(sys.argv))
            opt_file.write("\n==> Opt:\n")
            for k, v in sorted(args.items()):
                opt_file.write("  %s: %s\n" % (str(k), str(v)))

        """Initialize TensorBoard"""
        self.writer = SummaryWriter(log_dir=opt.log_dir)
        self.writer.add_hparams(args, {})  # Write parameters to TensorBoard

        # Initialize profiler to analyze bottlenecks
        # self.prof = profiler.profile(
        #     # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #     on_trace_ready=profiler.tensorboard_trace_handler(self.writer.log_dir),
        #     profile_memory=True,
        #     record_shapes=True,
        #     with_stack=True,
        # )

    def add_scalar(self, tag, value, step):
        """
        Writes scalar data to TensorBoard.

        Args:
            tag (str): Name of the tag
            value (float): Value of the scalar data
            step (int): Step count
        """
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Closes TensorBoard."""
        self.writer.close()
