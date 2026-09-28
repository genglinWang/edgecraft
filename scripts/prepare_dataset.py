#!/usr/bin/env python3
"""Download selected public input datasets; never calls a model API."""
import argparse
from pathlib import Path

HF_DATASETS = {
    "ag_news": ("fancyzhx/ag_news", None),
    "banking77": ("PolyAI/banking77", None),
    "tweet_eval": ("cardiffnlp/tweet_eval", "sentiment"),
    "go_emotions": ("google-research-datasets/go_emotions", "simplified"),
    "sst2": ("stanfordnlp/sst2", None),
    "glue_stsb": ("nyu-mll/glue", "stsb"),
    "boolq": ("google/boolq", None),
    "flickr8k": ("jxie/flickr8k", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[*HF_DATASETS, "cifar10"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Output already exists. Inspect it or choose a new destination.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "cifar10":
        from torchvision.datasets import CIFAR10

        CIFAR10(str(output), train=True, download=True)
        CIFAR10(str(output), train=False, download=True)
    else:
        from datasets import load_dataset

        dataset_id, config = HF_DATASETS[args.dataset]
        data = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)
        data.save_to_disk(str(output))
    print(f"Prepared {args.dataset}: {output}")
    print(f"Next: edgecraft data inspect {output}")


if __name__ == "__main__":
    main()
