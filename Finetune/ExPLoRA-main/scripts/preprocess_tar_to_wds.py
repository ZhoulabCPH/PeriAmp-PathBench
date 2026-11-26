from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class _PatchItem:
    center_root: str
    slide_id: str
    path: Path
    member: str


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
    print('[INFO] Scanning input directories...')

    for root in input_roots:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"[ERROR] Input root not found: {root_path}")

        center_root = root_path.name
        slide_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])

        for slide_dir in tqdm(slide_dirs, desc=f"Scanning cohort: {center_root}", unit="slide"):
            slide_id = slide_dir.name
            for img_path in slide_dir.rglob("*"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in extensions:
                    continue

                rel = f"{center_root}/{slide_id}/{img_path.name}"
                items.append(
                    _PatchItem(
                        center_root=center_root,
                        slide_id=slide_id,
                        path=img_path,
                        member=rel,
                    )
                )

                if max_samples is not None and len(items) >= max_samples:
                    print(f"[WARN] Reached the sample limit: {max_samples}")
                    return items

    print(f"[INFO] Found {len(items):,d} eligible images.")
    return items


def _get_resample(name: str):
    name = name.lower()
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unsupported resample: {name}")


def _resize_image(img: Image.Image, max_side: int, resample) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = int(round(h * max_side / w))
    else:
        new_h = max_side
        new_w = int(round(w * max_side / h))
    return img.resize((new_w, new_h), resample)


def _encode_image(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "jpg":
        img.save(
            buf,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    else:
        img.save(buf, format="PNG", compress_level=4)
    return buf.getvalue()


def _open_new_shard(
    output_dir: Path, shard_idx: int, prefix: str = "shard"
) -> Tuple[tarfile.TarFile, str]:
    name = f"{prefix}-{shard_idx:06d}.tar"
    path = output_dir / name
    tf = tarfile.open(str(path), mode="w", bufsize=8 * 1024 * 1024)
    return tf, name


def _add_bytes_to_tar(tf: tarfile.TarFile, arcname: str, data: bytes):
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _process_one(
    index: int,
    item: _PatchItem,
    resize: int,
    out_ext: str,
    quality: int,
    resample_name: str,
):
    try:
        with Image.open(item.path) as img:
            img.load()
            img = img.convert("RGB")
            if resize > 0:
                img = _resize_image(img, resize, _get_resample(resample_name))
            width, height = img.size
            img_data = _encode_image(img, out_ext, quality)

        metadata = {
            "center_root": item.center_root,
            "slide_id": item.slide_id,
            "original_name": item.path.name,
            "original_member": item.member,
            "width": width,
            "height": height,
        }
        meta_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        sample_bytes = len(img_data) + len(meta_bytes) + 1024

        return {
            "ok": True,
            "index": index,
            "item": item,
            "img_data": img_data,
            "meta_bytes": meta_bytes,
            "sample_bytes": sample_bytes,
        }
    except Exception as e:
        return {
            "ok": False,
            "index": index,
            "item": item,
            "error": str(e),
        }


def main() -> int:
    p = argparse.ArgumentParser(
        "Pack pathology patches into WebDataset tar shards",
        description=(
            'Pack pathology patches into WebDataset tar shards. '
            'Each sample includes {key}.{img_ext} and {key}.json. '
            'Supports resizing and PNG-to-JPEG conversion.'
        ),
    )
    p.add_argument(
        "--input_roots",
        type=str,
        nargs="+",
        default=[r"outputs/tiles/stream"],
        help='Input roots containing slide_id/patch.png',
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=r"outputs/tiles/webdataset",
        help='Output directory for WebDataset tar shards',
    )
    p.add_argument(
        "--extensions",
        type=str,
        default=".png,.jpg,.jpeg,.tif,.tiff,.bmp",
        help='Allowed image extensions',
    )
    p.add_argument(
        "--max_images_per_shard",
        type=int,
        default=50_000,
        help='Maximum samples per shard',
    )
    p.add_argument(
        "--max_bytes_per_shard",
        type=int,
        default=256,
        help='Maximum shard size in MB. Set 0 for no limit.',
    )
    p.add_argument(
        "--resize",
        type=int,
        default=0,
        help='Resize the long edge to this many pixels. Set 0 to preserve the input size.',
    )
    p.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["jpg", "png"],
        help='Choose lossy JPEG or lossless PNG.',
    )
    p.add_argument(
        "--quality",
        type=int,
        default=95,
        help='JPEG quality from 1 to 100, used only with --format jpg',
    )
    p.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Shuffle image order',
    )
    p.add_argument("--seed", type=int, default=42, help='Random seed')
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help='Debug sample limit. Set 0 for no limit.',
    )

    # Parallel image processing settings
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help='Number of image processing threads',
    )
    p.add_argument(
        "--prefetch",
        type=int,
        default=0,
        help='Prefetch task count. Set 0 to use workers * 4.',
    )
    p.add_argument(
        "--resample",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help='Resize interpolation. Bilinear is faster than bicubic or Lanczos.',
    )

    args = p.parse_args()

    exts = _parse_exts(args.extensions)
    max_samples = None if args.max_samples <= 0 else int(args.max_samples)
    out_ext = args.format
    workers = max(1, int(args.workers))
    prefetch = int(args.prefetch) if args.prefetch > 0 else workers * 4

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _discover_items(args.input_roots, extensions=exts, max_samples=max_samples)
    if not items:
        print('[ERROR] No eligible images. Check the input paths and extensions.')
        return 1

    if args.shuffle:
        print(f"[INFO] Shuffling input images with seed {args.seed})...")
        random.seed(args.seed)
        random.shuffle(items)

    print(f"[INFO] Total images: {len(items):,d}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Resize long edge: {args.resize if args.resize > 0 else 'disabled'}")
    print(f"[INFO] Output image format: {out_ext}" + (f" (quality={args.quality})" if out_ext == "jpg" else ""))
    print(f"[INFO] Images per shard: {args.max_images_per_shard}")
    if args.max_bytes_per_shard > 0:
        print(f"[INFO] Shard byte limit: {args.max_bytes_per_shard} MB")
    print(f"[INFO] workers: {workers}")
    print(f"[INFO] prefetch: {prefetch}")
    print(f"[INFO] resample: {args.resample}")

    max_shard_bytes = (
        args.max_bytes_per_shard * 1024 * 1024 if args.max_bytes_per_shard > 0 else 0
    )

    shard_idx = 0
    tf, shard_name = _open_new_shard(out_dir, shard_idx)
    shard_names: List[str] = [shard_name]

    images_in_shard = 0
    bytes_in_shard = 0
    total = 0
    total_img_bytes = 0
    shard_counts: List[int] = []
    skipped = 0

    pbar = tqdm(total=len(items), desc=f"Shard {shard_idx:04d}", unit="img")

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures: Deque[Future] = deque()

            def submit_one(i: int):
                return ex.submit(
                    _process_one,
                    i,
                    items[i],
                    args.resize,
                    out_ext,
                    args.quality,
                    args.resample,
                )

            next_submit = 0

            while next_submit < len(items) and len(futures) < prefetch:
                futures.append(submit_one(next_submit))
                next_submit += 1

            while futures:
                fut = futures.popleft()
                result = fut.result()

                if next_submit < len(items):
                    futures.append(submit_one(next_submit))
                    next_submit += 1

                if not result["ok"]:
                    print(f"\n[WARN] Skipping {result['item'].path}: {result['error']}")
                    skipped += 1
                    pbar.update(1)
                    continue

                img_data = result["img_data"]
                meta_bytes = result["meta_bytes"]
                sample_bytes = result["sample_bytes"]

                need_roll = False
                if images_in_shard >= args.max_images_per_shard:
                    need_roll = True
                if (
                    max_shard_bytes > 0
                    and (bytes_in_shard + sample_bytes) > max_shard_bytes
                    and images_in_shard > 0
                ):
                    need_roll = True

                if need_roll:
                    tf.close()
                    shard_counts.append(images_in_shard)
                    shard_idx += 1
                    tf, shard_name = _open_new_shard(out_dir, shard_idx)
                    shard_names.append(shard_name)
                    images_in_shard = 0
                    bytes_in_shard = 0
                    pbar.set_description(f"Shard {shard_idx:04d}")

                key = f"{total:08d}"
                _add_bytes_to_tar(tf, f"{key}.{out_ext}", img_data)
                _add_bytes_to_tar(tf, f"{key}.json", meta_bytes)

                images_in_shard += 1
                bytes_in_shard += sample_bytes
                total_img_bytes += len(img_data)
                total += 1
                pbar.update(1)

    finally:
        tf.close()
        pbar.close()
        shard_counts.append(images_in_shard)

    shards_txt = out_dir / "shards.txt"
    shards_txt.write_text("\n".join(shard_names) + "\n", encoding="utf-8")

    meta = {
        "input_roots": [str(Path(x)) for x in args.input_roots],
        "output_dir": str(out_dir),
        "num_samples": total,
        "num_shards": len(shard_names),
        "shard_counts": shard_counts,
        "format": out_ext,
        "resize": args.resize,
        "quality": args.quality if out_ext == "jpg" else None,
        "skipped": skipped,
        "workers": workers,
        "prefetch": prefetch,
        "resample": args.resample,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_out_bytes = sum(os.path.getsize(out_dir / n) for n in shard_names)

    print("-" * 50)
    print('[SUCCESS] Finished writing WebDataset shards.')
    print(f"[INFO] Total images: {total:,d}")
    if skipped > 0:
        print(f"[INFO] Skipped images: {skipped:,d}")
    print(f"[INFO] Total shards: {len(shard_names)}")
    print(f"[INFO] Output size: {total_out_bytes / (1024 ** 3):.2f} GB")
    print(f"[INFO] shards.txt: {shards_txt}")
    print(f"[INFO] meta.json: {meta_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
