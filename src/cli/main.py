import typer

app = typer.Typer()

@app.command()
def train():
    pass

@app.command()
def generate_data(path: str):
    if not path:
        raise ValueError(f'Path invalid {path}')
    typer.echo(f"Deleting project")

if __name__ == "__main__":
    app()
