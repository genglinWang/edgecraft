from types import SimpleNamespace

import pytest
import yaml

from edgecraft.agent.modality_handler import StructuredModalityHandler
from edgecraft.core.modality import TaskType


def test_preflight_yaml_keeps_dataset_and_split_contract(tmp_path):
    data = tmp_path / "inputs"
    data.mkdir()
    table = data / "samples.csv"
    table.write_text("feature,target\n1,0\n2,1\n")
    manifest = tmp_path / "splits.json"
    manifest.write_text('{}')
    source = tmp_path / "preflight.yaml"
    source.write_text(yaml.safe_dump({
        "dataset_path": "inputs", "data_root": str(data),
        "files": ["inputs/samples.csv"],
        "schema": {"files": ["inputs/samples.csv"], "label_column": "target"},
        "split_manifest": {"path": "splits.json", "content_hash": "example"},
    }))
    output = tmp_path / "trial" / "config"
    StructuredModalityHandler().materialize_config(
        SimpleNamespace(task_type=TaskType.CLASSIFICATION), output, str(source),
    )
    result = yaml.safe_load((output / "data.yaml").read_text())
    assert result["dataset_path"] == str(data)
    assert result["data_root"] == str(data)
    assert result["schema"] == {"files": [str(table)], "label_column": "target"}
    assert result["files"] == [str(table)]
    assert result["split_manifest"] == {"path": str(manifest), "content_hash": "example"}
    assert result["modality"] == "structured"


def test_invalid_dataset_config_fails_explicitly(tmp_path):
    source = tmp_path / "data.yaml"
    source.write_text("- this is not a dataset configuration\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        StructuredModalityHandler().materialize_config(
            SimpleNamespace(task_type=TaskType.CLASSIFICATION), tmp_path / "output", str(source),
        )
