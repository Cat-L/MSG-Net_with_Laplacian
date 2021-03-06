import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for PyTorch-Style-Transfer")
        subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        # training args
        train_arg = subparsers.add_parser("train",
                                    help="parser for training arguments")
        train_arg.add_argument("--ngf", type=int, default=128,
                                help="number of generator filter channels, default 128")
        train_arg.add_argument("--epochs", type=int, default=2,
                                help="number of training epochs, default is 2")
        train_arg.add_argument("--batch-size", type=int, default=4,
                                help="batch size for training, default is 4")
        train_arg.add_argument("--dataset", type=str, default="dataset/",
                                help="path to training dataset, the path should point to a folder "
                                "containing another folder with all the training images")
        train_arg.add_argument("--style-folder", type=str, default="images/9styles/",
                                help="path to style-folder")
        train_arg.add_argument("--vgg-model-dir", type=str, default="models/",
                                help="directory for vgg, if model is not present in the directory it is downloaded")
        train_arg.add_argument("--save-model-dir", type=str, default="models/",
                                help="path to folder where trained model will be saved.")
        train_arg.add_argument("--image-size", type=int, default=256,
                                help="size of training images, default is 256 X 256")
        train_arg.add_argument("--style-size", type=int, default=512,
                                help="size of style-image, default is the original size of style image")
        train_arg.add_argument("--cuda", type=int, default=1, 
                                help="set it to 1 for running on GPU, 0 for CPU")
        train_arg.add_argument("--seed", type=int, default=42, 
                                help="random seed for training")
        train_arg.add_argument("--content-weight", type=float, default=1.0,
                                help="weight for content-loss, default is 1.0")
        train_arg.add_argument("--style-weight", type=float, default=5.0,
                                help="weight for style-loss, default is 5.0")
        train_arg.add_argument("--lr", type=float, default=1e-3,
                                help="learning rate, default is 0.001")
        train_arg.add_argument("--log-interval", type=int, default=500,
                                help="number of images after which the training loss is logged, default is 500")
        train_arg.add_argument("--resume", type=str, default=None,
                                help="resume if needed")


    def parse(self):
        return self.parser.parse_args()
