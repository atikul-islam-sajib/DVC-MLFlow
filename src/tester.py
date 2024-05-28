import os
import torch
import argparse
from utils import load, config
from model import Model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


class TestModel:
    def __init__(self):

        self.model = Model()

    def load_dataloader(self):
        return load(
            filename=os.path.join(
                config()["path"]["PROCESSED_PATH"], "test_dataloader.pkl"
            )
        )

    def select_best_model(self):
        if os.path.exists(config()["path"]["BEST_MODEL_PATH"]):
            best_model_path = config()["path"]["BEST_MODEL_PATH"]

            best_model = torch.load(os.path.join(best_model_path, "best_model.pth"))

            self.model.load_state_dict(best_model["model"])

    def saved_performace(self, **kwargs):
        with open(
            os.path.join(config()["path"]["OUTPUTS_PATH"], "valid_performace.txt"), "w"
        ) as file:
            file.write(
                "Accuracy: {}\nPrecision:{}\nRecall:{}\nF1_Score:{}\n{}".format(
                    str(kwargs["accuracy"]),
                    str(kwargs["precision"]),
                    str(kwargs["recall"]),
                    str(kwargs["f1_score"]),
                    "*" * 100,
                )
            )
            file.write(
                "\nConfusion Metrics\n{}\n{}".format(
                    str(confusion_matrix(kwargs["y_true"], kwargs["y_pred"])), "*" * 100
                )
            )
            file.write(
                "\nClassification report\n\n{}:\n".format(
                    str(classification_report(kwargs["y_true"], kwargs["y_pred"]))
                )
            )

    def test(self):
        dataloader = self.load_dataloader()

        print(dataloader)

        self.select_best_model()

        self.predicted = []
        self.actual = []

        for X, y in dataloader:
            X = X.float()
            y = y.float()

            predicted = self.model(X)
            predicted = predicted.view(-1)
            predicted = torch.where(predicted > 0.5, 1, 0)
            predicted = predicted.detach().flatten()

            self.actual.extend(y.detach().flatten())
            self.predicted.extend(predicted)

        accuracy = accuracy_score(self.predicted, self.actual)
        precision = precision_score(self.predicted, self.actual)
        recall = recall_score(self.predicted, self.actual)
        f1 = f1_score(self.predicted, self.actual)

        confusion = confusion_matrix(self.predicted, self.actual)
        classification = classification_report(self.predicted, self.actual)

        self.saved_performace(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            y_true=self.actual,
            y_pred=self.predicted,
        )

        print(f"Test Accuracy: {accuracy}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test F1 Score: {f1}")

        print("\n")

        print(f"Confusion Matrix: \n {confusion}")
        print("\n")

        print(f"Classification Report: \n {classification}")

        print(
            "Model performace saved in the folder {}".format(
                config()["path"]["OUTPUTS_PATH"]
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Model for Breast Cancer".title())
    parser.add_argument("--test", action="store_true", help="Test Model".capitalize())

    args = parser.parse_args()

    if args.test:
        test = TestModel()

        test.test()
