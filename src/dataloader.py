import os
import torch
import argparse
import pandas as pd
from utils import dump, load, config
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Loader:
    def __init__(self, dataset=None, batch_size=64, split_size=0.25):
        self.datframe = dataset
        self.batch_size = batch_size
        self.split_size = split_size

        self.PROCESSED_PATH = config()["path"]["PROCESSED_PATH"]

    def normalized_dataset(self, dataset):
        if isinstance(dataset, torch.Tensor):
            scaler = StandardScaler()
            return torch.tensor(scaler.fit_transform(dataset))
        else:
            raise TypeError("The dataset must be a torch tensor")

    def preprocessing(self):
        self.dataset = pd.read_csv(self.datframe)

        if isinstance(self.dataset, pd.DataFrame):
            if "id" in self.dataset.columns:
                self.dataset.drop(["id"], axis=1, inplace=True)
            else:
                print("The dataset does not have an id column")

            if "diagnosis" in self.dataset.columns:
                if (
                    "M" in self.dataset["diagnosis"].unique()
                    and "B" in self.dataset["diagnosis"].unique()
                ):
                    self.dataset["diagnosis"] = self.dataset["diagnosis"].map(
                        {"B": 0, "M": 1}
                    )
                else:
                    print("The dataset does not have the correct diagnosis values")
            else:
                print("The dataset does not have a diagnosis column")

            X = self.dataset.iloc[:, 1:]
            y = self.dataset.iloc[:, 0]

            y = y.astype(int)

            X = torch.tensor(data=X.values, dtype=torch.float)
            y = torch.tensor(data=y.values, dtype=torch.long)

            X = self.normalized_dataset(dataset=X)

            return {"X": X, "y": y}

        else:
            raise TypeError("The dataset must be a pandas DataFrame")

    def create_dataloader(self):
        data = self.preprocessing()

        if isinstance(data["X"], torch.Tensor) and isinstance(data["y"], torch.Tensor):
            X = data["X"]
            y = data["y"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size, random_state=42
            )

            train_dataloader = DataLoader(
                dataset=list(zip(X_train, y_train)),
                batch_size=self.batch_size,
                shuffle=True,
            )

            test_dataloader = DataLoader(
                dataset=list(zip(X_test, y_test)),
                batch_size=self.batch_size,
                shuffle=True,
            )

            os.makedirs(self.PROCESSED_PATH, exist_ok=True)
            for filename, value in [
                ("train_dataloader", train_dataloader),
                ("test_dataloader", test_dataloader),
            ]:
                dump(
                    value=value,
                    filename=os.path.join(self.PROCESSED_PATH, f"{filename}.pkl"),
                )

            print(
                "The dataloader has been saved in the folder {}".format(
                    self.PROCESSED_PATH
                )
            )

    @staticmethod
    def dataset_details():
        PROCESSED_PATH = config()["path"]["PROCESSED_PATH"]
        FILES_PATH = config()["path"]["FILES_PATH"]

        if os.path.exists(PROCESSED_PATH):
            train_dataloader = load(
                filename=os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
            )
            test_dataloader = load(
                filename=os.path.join(PROCESSED_PATH, "test_dataloader.pkl")
            )

            X, y = next(iter(train_dataloader))

        pd.DataFrame(
            {
                "Train Data(Total)": sum(X.size(0) for X, y in train_dataloader),
                "Test Data(Total)": sum(X.size(0) for X, y in test_dataloader),
                "Data Shpe (Train)": str(X.size()),
            },
            index=["Quantity"],
        ).T.to_csv(
            os.path.join(FILES_PATH, "dataset_details.csv"),
        )


if __name__ == "__main__":
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
        help="Defin the split size".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        default=config()["data"]["batch_size"],
        help="Defin the batch size".capitalize(),
    )

    args = parser.parse_args()

    loader = Loader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        split_size=args.split_size,
    )

    loader.create_dataloader()

    loader.dataset_details()
