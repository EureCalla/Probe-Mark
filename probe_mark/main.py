from opts import opts


def main(opt):
    if opt.test:
        from predictor import Predictor
        predictor = Predictor(opt)
        predictor.predict(opt.image_path)
    else:
        from train import main as train_main
        from logger import Logger
        logger = Logger(opt)
        train_main(opt, logger)


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
