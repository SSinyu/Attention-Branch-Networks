import argparse
from os import path, makedirs

from solver import Solver
from utils.loader import DataLoader


def main(args):
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)
        print("Create {}".format(args.save_dir))

    if args.mode == "train":
        train_loader = DataLoader("train", args.batch_size, args.pad_size, args.crop_size)
        valid_loader = DataLoader("valid", args.batch_size*2)

        solver = Solver(args, train_loader, valid_loader)
        solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pad_size", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=32)

    parser.add_argument("--mixed_training", type=bool, default=False)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--save_dir", type=str, default="./logs")

    args = parser.parse_args()
    main(args)
