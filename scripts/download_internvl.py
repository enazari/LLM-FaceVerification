"""Download InternVL2-2B weights to data/checkpoints/InternVL2-2B/.

Works on both local machines and HPC login nodes.
Idempotent — skips if already downloaded.

Usage:
    python scripts/download_internvl.py
    python scripts/download_internvl.py --cache-dir /path/to/custom/dir
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download InternVL2-2B weights")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("data", "checkpoints", "InternVL2-2B"),
        help="Where to save the model (default: data/checkpoints/InternVL2-2B)",
    )
    parser.add_argument(
        "--model-id",
        default="OpenGVLab/InternVL2-2B",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)

    # Check if already downloaded (look for config.json as sentinel)
    sentinel = os.path.join(cache_dir, "config.json")
    if os.path.exists(sentinel):
        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(cache_dir) for f in fns
        ) / 1e9
        print(f"InternVL2-2B already at {cache_dir} ({size_gb:.1f} GB)")
        return

    print(f"Downloading {args.model_id} → {cache_dir}")
    print("This is ~4.4 GB and may take a few minutes...\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    snapshot_download(
        repo_id=args.model_id,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,  # real files, not symlinks
    )

    size_gb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(cache_dir) for f in fns
    ) / 1e9
    print(f"\nDone. Saved to {cache_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
