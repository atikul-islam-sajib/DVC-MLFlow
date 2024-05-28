import os
import torch
import mlflow
import argparse
import numpy as np
from tqdm import tqdm
from utils import config, dump, load
from helper import helpers
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.01,
        beta1=0.5,
        beta2=0.999,
        momentum=0.9,
        step_size=7,
        gamma=0.1,
        adam=True,
        SGD=False,
        device="cpu",
        is_display=True,
        lr_scheduler=False,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam = adam
        self.SGD = SGD
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self.device = device
        self.is_display = is_display
        self.lr_scheduler = lr_scheduler

        self.experiment_name = "Breast Cancer Classification"

        self.init = helpers(
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
        )

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["test_dataloader"]

        self.model = self.init["model"].to(self.device)
        self.criterion = self.init["criterion"]
        self.optimizer = self.init["optimizer"]

        if self.lr_scheduler:
            self.scheduler = StepLR(
                optimizer=self.optimizer, step_size=self.step_size, gamma=self.gamma
            )

        self.history = {"train_loss": [], "test_loss": []}
        self.loss = float("inf")
        self.config = config()

        self.raw_data_path = self.config["path"]["RAW_PATH"]
        self.files_path = self.config["path"]["FILES_PATH"]
        self.train_models_path = self.config["path"]["TRAIN_MODEL_PATH"]
        self.best_model_path = self.config["path"]["BEST_MODEL_PATH"]

    def update_model(self, X, y):
        if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
            X = X.float()
            y = y.float()

            self.optimizer.zero_grad()

            predicted = self.model(X)
            predicted_loss = self.criterion(predicted, y)

            predicted_loss.backward()
            self.optimizer.step()

            return predicted, predicted_loss.item()

        else:
            raise TypeError("X and y must be torch.Tensor")

    def display_progress(self, **kwargs):
        if self.is_display:
            print(
                "Epochs - [{}/{}] - Train Loss: {:.4f} - Test Loss: {:.4f} - train_accuracy: {:.4f} - test_accuracy: {:.4f}".format(
                    kwargs["epoch"] + 1,
                    self.epochs,
                    np.mean(kwargs["train_loss"]),
                    np.mean(kwargs["test_loss"]),
                    accuracy_score(kwargs["train_actual"], kwargs["train_pred"]),
                    accuracy_score(kwargs["valid_actual"], kwargs["valid_pred"]),
                )
            )
        else:
            print(
                "Epochs - [{}/{}] is completed".format(kwargs["epoch"] + 1, self.epochs)
            )

    def saved_models(self, epoch=None, train_loss=None):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.train_models_path, "model{}.pth".format(epoch)),
        )

        if train_loss is not None:
            if self.loss > train_loss:
                self.loss = train_loss
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "loss": train_loss,
                        "epoch": epoch,
                    },
                    os.path.join(self.best_model_path, "best_model.pth"),
                )
        else:
            raise ValueError("Train loss cannot be None".capitalize())

    def compute_model_performace(self, actual, predicted):
        return {
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision_score(actual, predicted),
            "recall": recall_score(actual, predicted),
            "f1_score": f1_score(actual, predicted),
        }

    def train(self):
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(
            description="This is used for to track the Breast Cancer Classification Model loss"
        ) as run:

            for epoch in tqdm(range(self.epochs)):
                train_predcit = []
                valid_predict = []
                train_actual = []
                valid_actual = []
                train_loss = []
                test_loss = []

                for _, (X, y) in enumerate(self.train_dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    X = X.float()
                    y = y.float()

                    predicted, loss = self.update_model(X=X, y=y.unsqueeze(1))

                    train_predcit.append(
                        torch.where(predicted > 0.5, 1, 0).detach().cpu().numpy()
                    )
                    train_actual.append(
                        torch.where(y > 0.5, 1, 0).detach().cpu().numpy()
                    )

                    train_loss.append(loss)

                for _, (X, y) in enumerate(self.valid_dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    X = X.float()
                    y = y.float()

                    predicted = self.model(X)
                    loss = self.criterion(predicted, y.unsqueeze(1))

                    valid_predict.append(
                        torch.where(predicted > 0.5, 1, 0).detach().cpu().numpy()
                    )
                    valid_actual.append(
                        torch.where(y > 0.5, 1, 0).detach().cpu().numpy()
                    )

                    test_loss.append(loss.item())

                if self.lr_scheduler:
                    self.scheduler.step()

                train_actual = np.concatenate(train_actual)
                train_predcit = np.concatenate(train_predcit)
                valid_actual = np.concatenate(valid_actual)
                valid_predict = np.concatenate(valid_predict)

                self.display_progress(
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    train_pred=train_predcit,
                    train_actual=train_actual,
                    valid_pred=valid_predict,
                    valid_actual=valid_actual,
                )

                self.saved_models(epoch=epoch + 1, train_loss=np.mean(train_loss))

                self.history["train_loss"].append(np.mean(train_loss))
                self.history["test_loss"].append(np.mean(test_loss))

                train_performance = self.compute_model_performace(
                    actual=train_actual, predicted=train_predcit
                )
                valid_performance = self.compute_model_performace(
                    actual=valid_actual, predicted=valid_predict
                )

                for index, performace in enumerate(
                    [train_performance, valid_performance]
                ):
                    for metric, value in performace.items():
                        mlflow.log_metric(
                            key="train_" + metric if index == 0 else "valid_" + metric,
                            value=value,
                            step=epoch + 1,
                        )

                for type, loss in [
                    ("train_loss", np.mean(train_loss)),
                    ("test_loss", np.mean(test_loss)),
                ]:
                    mlflow.log_metric(key=type, value=loss, step=epoch + 1)

            print("Training is completed".title())

            dump(
                value=self.history,
                filename=os.path.join(config()["path"]["FILES_PATH"], "history.pkl"),
            )

            mlflow.log_params(
                {
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "beta1": self.beta1,
                    "beta2": self.beta2,
                    "adam": self.adam,
                    "SGD": self.SGD,
                    "momentun": self.momentum,
                    "device": self.device,
                    "display": self.is_display,
                }
            )

            mlflow.pytorch.log_model(self.model, "Breast_Cancer_Model")

            mlflow.log_artifact(
                os.path.join(self.files_path, "Model.jpeg"),
            )

            mlflow.log_artifacts(
                os.path.join(
                    self.raw_data_path,
                ),
                artifact_path="dataset",
            )

    @staticmethod
    def plot_history():
        plt.figure(figsize=(10, 5))

        history = load(
            filename=os.path.join(config()["path"]["FILES_PATH"], "history.pkl")
        )

        for filename, loss in history.items():
            plt.plot(loss, label=filename)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()

        plt.tight_layout
        plt.savefig(os.path.join(config()["path"]["FILES_PATH"], "loss.png"))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer code for Breast Cancer".title()
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

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        adam=args.adam,
        SGD=args.SGD,
        is_display=args.display,
    )

    trainer.train()

    trainer.plot_history()
