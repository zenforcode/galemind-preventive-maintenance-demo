import typer
from typing import Final
import typer
from data_generation.syntetic import generate_syn_data

app = typer.Typer()


@app.command()
def train(epochs: int = 5, batch_size: int = 64, lr: float = 0.001):
    """
    """
    print("Hello")

@app.command()
def generate(path: str):
    if not path:
        raise ValueError(f"Path invalid {path}")
    generate_syn_data(path=path)


if __name__ == "__main__":
    app()
