"""Resolve the external TileScale checkout used as the benchmark baseline."""

import os
import sys
from pathlib import Path


def add_tilescale_example(example: str) -> None:
    root_value = os.environ.get("TILESCALE_ROOT")
    if not root_value:
        raise RuntimeError("Set TILESCALE_ROOT to your TileScale checkout directory")
    path = Path(root_value).expanduser().resolve() / "examples" / "distributed" / example
    if not path.is_dir():
        raise RuntimeError(f"TileScale example directory not found: {path}")
    sys.path.insert(0, str(path))


def add_local_kernels() -> None:
    path = Path(__file__).resolve().parents[1] / "kernels"
    sys.path.insert(0, str(path))
