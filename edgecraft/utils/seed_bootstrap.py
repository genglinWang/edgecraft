"""Execute a generated training script under one process-level random seed."""
from __future__ import annotations

import os
import random
import runpy
import sys


def apply_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("seed_bootstrap requires a training script path")
    seed = int(os.environ["EDGECRAFT_TRAIN_SEED"])
    target, args = sys.argv[1], sys.argv[2:]
    apply_seed(seed)
    sys.argv = [target, *args]
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
