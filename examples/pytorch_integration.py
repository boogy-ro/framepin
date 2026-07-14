#!/usr/bin/env python3
"""framepin + PyTorch: pin the exact dataset a training run saw, in 3 lines.

The pattern (everything framepin needs is marked with # <-- framepin):

    import framepin

    with framepin.track(name="exp-12", params={"lr": 3e-4}) as run:   # <-- framepin
        run.use_dataset("data/clips")                                 # <-- framepin
        for epoch in range(epochs):
            for batch in loader:
                ...
        run.log_metric("val_loss", best_val)                          # <-- framepin

Nothing about your Dataset/DataLoader changes — framepin fingerprints the
files on disk, it never wraps or slows the data path. If your dataset is
defined by txt manifests instead of a directory:

    man = framepin.snapshot_from_lists(["train_a.txt", "train_b.txt"])
    run.use_dataset(man)

This script runs a miniature end-to-end version of the above (a tiny tensor
"dataset" + a fake train loop). It needs torch; without torch it just prints
the pattern and exits 0 so CI/docs environments don't break.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

try:
    import framepin
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import framepin

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    print(__doc__)
    print("torch is not installed — printed the integration pattern above. "
          "Run `pip install torch` to execute the live demo.")
    raise SystemExit(0)


class ClipDataset(Dataset):
    """Loads the tiny .pt 'clips' created below — stands in for your real data."""

    def __init__(self, root: str):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.load(self.paths[i])


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="framepin-torch-")
    data = os.path.join(workdir, "clips")
    os.makedirs(data)
    try:
        for i in range(8):
            torch.save(torch.full((3, 4, 4), float(i)), os.path.join(data, f"clip_{i}.pt"))

        repo = framepin.Repo.init(workdir)
        with framepin.track(name="torch-demo", params={"lr": 3e-4}, repo=repo) as run:
            version = run.use_dataset(data)          # <-- pinned before training
            loader = DataLoader(ClipDataset(data), batch_size=4)
            loss = 0.0
            for batch in loader:                      # your real train loop here
                loss = float(batch.mean())
            run.log_metric("val_loss", round(loss, 4))

        print(f"dataset version: {version[:12]}")
        print(f"run {run.id}: val_loss={run.metrics['val_loss']} — "
              f"pinned to those exact bytes forever")
        print("later: `framepin regress <old> <new> -m val_loss` tells you "
              "if a metric move was code or data.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
