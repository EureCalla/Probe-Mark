import argparse
import os


class Opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        opt.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return opt
