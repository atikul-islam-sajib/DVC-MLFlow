import os
import argparse
from tester import TestModel
from trainer import Trainer
from dataloader import Loader
from utils import config


def train_test_model(**kwargs):
    if kwargs["train"]:
        loader = Loader(
            dataset=kwargs["dataset"],
            batch_size=kwargs["batch_size"],
            split_size=kwargs["split_size"],
        )
        loader.create_dataloader()

        trainer = Trainer(
            epochs=kwargs["epochs"],
            lr=kwargs["lr"],
            beta1=kwargs["beta1"],
            beta2=kwargs["beta2"],
            adam=kwargs["adam"],
            SGD=kwargs["SGD"],
            is_display=kwargs["display"],
        )

        trainer.train()

        trainer.plot_history()

    elif kwargs["test"]:
        test = TestModel()
        test.test()


def cli():
    parser = argparse.ArgumentParser(description="CLI for the project".capitalize())

    parser = argparse.ArgumentParser(
        description="Data Loader for the Cancer".capitalize()
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(config()["path"]["RAW_PATH"], "breast-cancer.csv"),
        help="Defin the dataset".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        default=config()["data"]["split_size"],
        type=float,
        help="Defin the split size".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        default=config()["data"]["batch_size"],
        type=int,
        help="Defin the batch size".capitalize(),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["trainer"]["epochs"],
        help="Number of epochs".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["trainer"]["lr"],
        help="Learning rate".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["trainer"]["beta1"],
        help="Beta1".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["trainer"]["beta2"],
        help="Beta2".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Adam".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="SGD".capitalize(),
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=config()["trainer"]["display"],
        help="Display".capitalize(),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file to train and test the model".capitalize(),
    )

    parser.add_argument("--train", action="store_true", help="Train Model".capitalize())
    parser.add_argument("--test", action="store_true", help="Test Model".capitalize())

    args = parser.parse_args()

    if args.config is not None:
        train_test_model(
            train=args.train,
            test=args.test,
            dataset=os.path.join(config()["path"]["RAW_PATH"], "breast-cancer.csv"),
            batch_size=config()["data"]["batch_size"],
            split_size=config()["data"]["split_size"],
            epochs=config()["trainer"]["epochs"],
            lr=config()["trainer"]["lr"],
            beta1=config()["trainer"]["beta1"],
            beta2=config()["trainer"]["beta2"],
            adam=config()["trainer"]["adam"],
            SGD=config()["trainer"]["SGD"],
            display=config()["trainer"]["display"],
        )

    else:
        train_test_model(
            train=args.train,
            test=args.test,
            dataset=args.dataset,
            batch_size=args.batch_size,
            split_size=args.split_size,
            epochs=args.epochs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            adam=args.adam,
            SGD=args.SGD,
            display=args.display,
        )


if __name__ == "__main__":
    cli()
