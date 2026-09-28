"""Small, read-only observations for raw audio and text dataset formats."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def additional_observation(root: Path, files: list[Path], max_rows: int) -> dict[str, Any] | None:
    audio = sorted(p for p in files if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"})
    if audio:
        import soundfile as sf

        examples = []
        for path in audio[:max_rows]:
            with sf.SoundFile(path) as stream:
                frames = stream.read(min(64, len(stream)), always_2d=True)
                info = {"path": str(path.relative_to(root)), "sample_rate": stream.samplerate,
                        "channels": stream.channels, "frames": len(stream),
                        "decoded_shape": list(frames.shape)}
            examples.append({"input": info, "label": None})
        return {"status": "success", "format": "audio_files",
                "num_observed_samples": len(examples), "examples": examples,
                "input_summary": examples[0]["input"],
                "label_summary": {"note": "Read annotation files or the documented filename convention for labels."}}

    # torchvision checks the official CIFAR payload's checksums before loading.
    if any(p.name == "data_batch_1" for p in files):
        from torchvision.datasets import CIFAR10

        batch = next(p for p in files if p.name == "data_batch_1")
        dataset = CIFAR10(str(batch.parent.parent), train=True, download=False)
        examples = [{"input": {"size": list(dataset[i][0].size), "mode": dataset[i][0].mode},
                     "label": int(dataset[i][1])} for i in range(min(max_rows, len(dataset)))]
        return {"status": "success", "format": "cifar_python",
                "num_observed_samples": len(examples), "examples": examples,
                "input_summary": examples[0]["input"],
                "label_summary": {"classes": dataset.classes}}

    for path in sorted(files):
        if path.suffix.lower() not in {"", ".txt", ".data"} or path.stat().st_size > 500_000_000:
            continue
        try:
            with path.open(encoding="utf-8") as stream:
                lines = [stream.readline(8192) for _ in range(max_rows)]
            lines = [line.rstrip("\n") for line in lines if line.strip()]
            if len(lines) < 2:
                continue
            delimiter = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
            if delimiter is None:
                continue
            rows = list(csv.reader(lines, delimiter=delimiter))
            if len(rows[0]) < 2:
                continue
            return {"status": "success", "format": "delimited_text",
                    "num_observed_samples": len(rows), "source_file": str(path.relative_to(root)),
                    "input_summary": {"delimiter": delimiter, "columns": len(rows[0]), "rows": rows},
                    "label_summary": {"note": "Column roles follow the dataset's documentation."}}
        except (UnicodeError, OSError, csv.Error):
            continue
    return None
