# A simple Classification Project

This project provides tools for training and testing a machine learning model for breast cancer classification using PyTorch and MLflow for experiment tracking and data versioning. It includes data loading, model training, evaluation, and logging functionalities.

## Requirements

- Python 3.9+
- PyTorch
- MLflow
- numpy
- scikit-learn
- tqdm
- matplotlib
- argparse
- dagshub

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/DVC-MLFlow.git
cd MLFlowDVC-MLFlow
pip install -r requirements.txt
```

## Usage

The project can be run through the command-line interface (CLI). The CLI allows for both training and testing of the model.

### Training

To train the model and log the experiments with MLflow, use the following command:

```bash
python cli.py --dataset /path/to/dataset.csv --batch_size 64 --split_size 0.30 --epochs 200 --lr 0.01 --adam True --display True --train
```

### Testing

To test the model, use the following command:

```bash
python cli.py --test
```

## CLI Arguments

- `--dataset`: Path to the dataset.
- `--batch_size`: Batch size for the dataloader.
- `--split_size`: Split size for the dataloader.
- `--epochs`: Number of epochs (default: 10).
- `--lr`: Learning rate (default: 0.01).
- `--beta1`: Beta1 for Adam optimizer (default: 0.5).
- `--beta2`: Beta2 for Adam optimizer (default: 0.999).
- `--adam`: Use Adam optimizer (default: True).
- `--SGD`: Use SGD optimizer (default: False).
- `--device`: Device to use (default: 'cpu').
- `--display`: Display training progress (default: True).
- `--train`: Flag to indicate training the model.
- `--test`: Flag to indicate testing the model.

## MLflow Tracking

The training process utilizes MLflow for tracking experiments. Metrics such as loss, accuracy, precision, and recall are logged at each epoch. Additionally, confusion matrices and classification reports are saved as artifacts. Model checkpoints and the best model are also saved.

To view the MLflow tracking UI, run:

```bash
mlflow ui
```

This will start the MLflow server, and you can access the UI at `http://localhost:5000` to visualize your experiment runs and metrics.

## Project Structure

- `trainer.py`: Contains the `Trainer` class for training the model.
- `test.py`: Contains the `TestModel` class for testing the model.
- `dataloader.py`: Contains the `Loader` class for handling data loading and splitting.
- `utils/`: Utility functions and configuration handling.
- `helper.py`: Helper functions for initializing components like model, optimizer, etc.
- `cli.py`: Command-line interface for running training and testing.

## Data Versioning with DVC
To ensure you have the correct version of the dataset and model artifacts.

Reproducing the Pipeline
To reproduce the entire pipeline and ensure consistency, use:

```bash
dvc repro
```

## Docker Usage

To simplify the process of setting up the environment and running the project, a Docker image is available. You can pull and run the Docker image with the following commands.

### Pulling the Docker Image

Pull the Docker image from Docker Hub:

```bash
docker pull sajibds/clf:latest
```

### Running the Docker Container

Run the Docker container with the following command. Make sure to mount the dataset directory to the container:

```bash
docker run -v /path/to/dataset:/app/dataset sajibds/clf:latest --dataset /app/dataset/dataset.csv --batch_size 64 --split_size 0.30 --epochs 200 --lr 0.01 --adam True --display True --train
```

### Testing with Docker

To test the model using the Docker container, run:

```bash
docker run sajibds/clf:latest --test
```

## Contact

For any questions or issues, please contact Atikul Islam Sajib.