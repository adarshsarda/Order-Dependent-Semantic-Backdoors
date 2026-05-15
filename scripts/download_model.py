"""Pre-download Qwen2.5-7B-Instruct from Hugging Face.

Useful for environments where the training step would otherwise fail with a
network error mid-run.  Requires that you've accepted the model's license
on HF and set HF_TOKEN.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--cache_dir", type=Path, default=None)
    args = p.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = str(args.cache_dir)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.model,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
