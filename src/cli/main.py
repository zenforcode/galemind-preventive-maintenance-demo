import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import typer
from tqdm.rich import tqdm, TqdmExperimentalWarning
from rich.progress import Progress
from gm_models.model import CNN
from typing import Final
import warnings
import typer


app = typer.Typer()
MODEL_PATH: Final[str] = "digit_detector.pth"


@app.command()
def train(epochs: int = 5, batch_size: int = 64, lr: float = 0.001):
    """
    Train a CNN model on the MNIST dataset.
    """
    if not os.path.isfile(MODEL_PATH):
        typer.echo("Start model training...")

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        train_dataset = datasets.MNIST(
            ".", train=True, download=True, transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = CNN()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            for data, target in tqdm(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        typer.echo(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        typer.echo("Model saved.")
    else:
        typer.echo("Model already exists. Delete the file to retrain.")


@app.command()
def generate_data(path: str):
    if not path:
        raise ValueError(f"Path invalid {path}")
    typer.echo(f"Deleting project")


if __name__ == "__main__":
    app()
