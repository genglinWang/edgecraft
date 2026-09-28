from click.testing import CliRunner

from edgecraft.cli import main
from edgecraft.cli.data import dataset_catalog


def test_catalog_has_all_fifty_paper_tasks():
    rows = dataset_catalog()
    assert len(rows) == 50
    assert [r["task_id"] for r in rows] == [f"T{i:02}" for i in range(1, 51)]
    assert len({r["id"] for r in rows}) == 50
    assert len({r["modality"] for r in rows}) == 6
    assert len({r["device"] for r in rows}) == 5
    assert all(r["source"].startswith(("http://", "https://")) for r in rows)
    assert rows[4]["id"] == "pascal_voc"
    assert rows[19]["id"] == "loghub_hdfs"


def test_dataset_list_needs_no_provider():
    result = CliRunner().invoke(main, ["data", "list"])
    assert result.exit_code == 0, result.output
    assert "T01" in result.output and "T50" in result.output


def test_raw_delimited_table_is_read_without_guessing_a_header(tmp_path):
    from edgecraft.data.observations import additional_observation

    path = tmp_path / "sample.data"
    path.write_text("a,1,2\nb,3,4\n")
    result = additional_observation(tmp_path, [path], 4)
    assert result["input_summary"]["rows"] == [["a", "1", "2"], ["b", "3", "4"]]
