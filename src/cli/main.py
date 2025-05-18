import typer
from typing import Final
import typer
from model.pipeline import PredictiveMaintenanceFlow
from data_generation.syntetic import generate_data

app = typer.Typer()


@app.command()
def train(epochs: int = 5, batch_size: int = 64, lr: float = 0.001):
    """
    Train a CNN model on the MNIST dataset.
    """
    print("Hello")

@app.command()
def generate_data(path: str):
    if not path:
        raise ValueError(f"Path invalid {path}")
    generate_data(path=path)


if __name__ == "__main__":
    app()
