"""
Pack pathology patch images into tar shards + JSONL index.

Why:
  - WSL2 reading millions of small files from /mnt/<drive> (Windows NTFS) is often slow.
  - Packing into fewer large .tar files dramatically reduces per-sample open/stat overhead.

Expected input layout (same as virchow2.dataset.MultiCenterPathologyDataset):
  RootA/
    Slide_001/
      *.png / *.jpg / ...
    Slide_002/
      ...
  RootB/
    ...

Outputs:
  output_dir/
    shards/
      shard-000000.tar
      shard-000001.tar
      ...
    index.jsonl
    meta.json

Index JSONL line example:
  {"shard":"shards/shard-000000.tar","member":"RootA/Slide_001/patch_0001.png","center_root":"RootA","slide_id":"Slide_001"}

Run on Windows:
  python scripts/pack_patches_to_tar_shards.py --input_roots outputs/tiles/stream --output_dir outputs/tiles/tar_shards
Run on WSL2:
  python3 scripts/pack_patches_to_tar_shards.py --input_roots outputs/tiles/stream --output_dir outputs/tiles/tar_shards
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from tqdm import tqdm


@dataclass(frozen=True)
class _PatchItem:
    center_root: str
    slide_id: str
    path: Path
    member: str  # Relative path inside the tar archive


def _parse_exts(exts: str) -> Tuple[str, ...]:
    xs = [e.strip().lower() for e in exts.split(",") if e.strip()]
    xs = [e if e.startswith(".") else f".{e}" for e in xs]
    return tuple(xs)


def _discover_items(
        input_roots: Sequence[str],
        *,
        extensions: Tuple[str, ...],
        max_samples: Optional[int],
) -> List[_PatchItem]:
    items: List[_PatchItem] = []
    print(f"[INFO] Scanning input directories...")

    for root in input_roots:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"[ERROR] Input root not found: {str(root_path)}")

        center_root = root_path.name
        # Sort the slide directories.
        slide_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])

        # Show scanning progress.
        for slide_dir in tqdm(slide_dirs, desc=f"Scanning cohort: {center_root}", unit="slide"):
            slide_id = slide_dir.name
            for img_path in slide_dir.rglob("*"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in extensions:
                    continue

                # Use POSIX archive paths for cross-platform compatibility.
                rel = f"{center_root}/{slide_id}/{img_path.name}"
                items.append(_PatchItem(
                    center_root=center_root,
                    slide_id=slide_id,
                    path=img_path,
                    member=rel
                ))

                if max_samples is not None and len(items) >= max_samples:
                    print(f"[WARN] Reached the sample limit: {max_samples}")
                    return items

    print(f"[INFO] Found {len(items):,d} eligible images.")
    return items


def _open_new_shard(shards_dir: Path, shard_idx: int) -> Tuple[tarfile.TarFile, Path]:
    shard_name = f"shard-{shard_idx:06d}.tar"
    shard_path = shards_dir / shard_name
    # Use uncompressed tar archives to improve write speed.
    tf = tarfile.open(shard_path, mode="w")
    return tf, shard_path


def main() -> int:
    p = argparse.ArgumentParser("Pack pathology patch images into tar shards", add_help=True)
    p.add_argument("--input_roots", type=str, nargs="+", default=[r"outputs/tiles/stream"],
                   help='Input dataset roots')
    p.add_argument("--output_dir", type=str, default=r"outputs/tiles/tar_shards", help='Output directory')
    p.add_argument("--extensions", type=str, default=".png,.jpg,.jpeg,.tif,.tiff,.bmp", help='Allowed image extensions')
    p.add_argument("--max_images_per_shard", type=int, default=50_000, help='Maximum images per shard')
    p.add_argument("--max_bytes_per_shard", type=int, default=0, help='Shard byte limit. Set 0 for no limit.')
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False, help='Shuffle image order')
    p.add_argument("--seed", type=int, default=42, help='Random seed')
    p.add_argument("--max_samples", type=int, default=0, help='Limit the number of samples for debugging')
    args = p.parse_args()

    exts = _parse_exts(args.extensions)
    max_samples = None if args.max_samples <= 0 else int(args.max_samples)

    # Prepare the output directory.
    out_dir = Path(args.output_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    meta_path = out_dir / "meta.json"

    # Discover input images.
    items = _discover_items(args.input_roots, extensions=exts, max_samples=max_samples)
    if not items:
        print('[ERROR] No eligible images. Check the input paths and extensions.')
        return 1

    # Shuffle the selected images when requested.
    if args.shuffle:
        print(f"[INFO] Shuffling input images with seed {args.seed})...")
        random.seed(args.seed)
        random.shuffle(items)

    # Write the tar shards and index.
    print(f"[INFO] Writing tar shards to {shards_dir}")
    shard_idx = 0
    tf, shard_path = _open_new_shard(shards_dir, shard_idx)
    shard_rel = str(Path("shards") / shard_path.name).replace("\\", "/")

    images_in_shard = 0
    bytes_in_shard = 0
    total = 0
    shard_counts = []

    with index_path.open("w", encoding="utf-8") as w:
        pbar = tqdm(total=len(items), desc=f"Shard {shard_idx:04d}", unit="img")
        try:
            for it in items:
                try:
                    fsize = it.path.stat().st_size
                except OSError:
                    fsize = 0

                # Check whether the current shard has reached its limit.
                need_roll = False
                if images_in_shard >= args.max_images_per_shard:
                    need_roll = True
                if args.max_bytes_per_shard > 0 and (
                        bytes_in_shard + fsize) > args.max_bytes_per_shard and images_in_shard > 0:
                    need_roll = True

                if need_roll:
                    tf.close()
                    shard_counts.append(images_in_shard)
                    shard_idx += 1
                    tf, shard_path = _open_new_shard(shards_dir, shard_idx)
                    shard_rel = str(Path("shards") / shard_path.name).replace("\\", "/")
                    images_in_shard = 0
                    bytes_in_shard = 0
                    pbar.set_description(f"Shard {shard_idx:04d}")

                # Write the image to the tar archive.
                tf.add(str(it.path), arcname=it.member, recursive=False)

                # Record the archive location in the index.
                rec = {
                    "shard": shard_rel,
                    "member": it.member,
                    "center_root": it.center_root,
                    "slide_id": it.slide_id,
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

                images_in_shard += 1
                bytes_in_shard += fsize
                total += 1
                pbar.update(1)
        finally:
            tf.close()
            pbar.close()
            shard_counts.append(images_in_shard)

    # Write the dataset metadata.
    meta = {
        "input_roots": [str(Path(x)) for x in args.input_roots],
        "output_dir": str(out_dir),
        "num_samples": total,
        "num_shards": shard_idx + 1,
        "shard_counts": shard_counts,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 50)
    print(f"[SUCCESS] Finished writing tar shards.")
    print(f"[INFO] Total images: {total:,d}")
    print(f"[INFO] Total shards: {len(shard_counts)}")
    print(f"[INFO] Index path: {index_path}")
    print("-" * 50)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())


