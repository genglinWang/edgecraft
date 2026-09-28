"""Local dataset commands. These commands do not create an LLM client."""
import json
from importlib.resources import files
from pathlib import Path

import click


def dataset_catalog():
    return json.loads(files("edgecraft.config").joinpath("datasets.json").read_text())["datasets"]


@click.group("data")
def data_commands():
    """List datasets and inspect local inputs without model API calls."""


@data_commands.command("list")
def list_datasets():
    """List the paper's 50 public datasets and their local directory names."""
    for row in dataset_catalog():
        click.echo(f'{row["task_id"]}  {row["name"]:<27} {row["modality"]:<11} {row["directory"]}')


@data_commands.command("inspect")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--rows", type=click.IntRange(1, 32), default=4, show_default=True)
def inspect_data(directory, rows):
    """Read small samples and report format, shape, labels, and split hints."""
    from edgecraft.utils.analyzer import build_sample_observation

    result = build_sample_observation(directory, max_rows=rows)
    click.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if result.get("status") != "success":
        raise click.ClickException("No readable sample found; check the dataset layout and required format dependencies.")
