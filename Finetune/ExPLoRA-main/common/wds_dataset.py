"""
Stream pathology samples sequentially from WebDataset tar shards.
Sequential reads reduce random seeks on network file systems. Shards and samples are shuffled, and DDP ranks receive distinct shards. Small shards improve page cache reuse. This loader requires webdataset.
"""

from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image


def _build_shard_urls(
    wds_root: str,
    shard_pattern: str = "shard-{000000..999999}.tar",
    shard_list: Optional[str] = None,
) -> Union[str, list]:
    root = Path(wds_root)
    if shard_list:
        list_path = Path(shard_list)
        if not list_path.is_absolute():
            list_path = root / list_path
        names = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [str(root / n) for n in names]
    return str(root / shard_pattern)


def _decode_sample(sample: dict) -> dict:
    """Decode a WebDataset sample containing a JPEG image and JSON metadata."""
    jpg_data = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    if jpg_data is None:
        raise KeyError(f"No image found in sample keys: {list(sample.keys())}")
    img = Image.open(io.BytesIO(jpg_data)).convert("RGB")

    meta = {}
    json_data = sample.get("json")
    if json_data is not None:
        if isinstance(json_data, bytes):
            meta = json.loads(json_data.decode("utf-8"))
        elif isinstance(json_data, str):
            meta = json.loads(json_data)
        else:
            meta = json_data
    return {"image": img, "meta": meta, "__key__": sample.get("__key__", "")}


def build_wds_pipeline(
    wds_root: str,
    transform: Optional[Callable] = None,
    shard_list: Optional[str] = None,
    epoch_length: Optional[int] = None,
    shuffle_buffer: int = 5000,
    num_workers: int = 8,
    batch_size: int = 16,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
):
    import webdataset as wds

    urls = _build_shard_urls(wds_root, shard_list=shard_list or "shards.txt")

    dataset = (
        wds.WebDataset(
            urls,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
            seed=seed,
        )
        .shuffle(shuffle_buffer, rng=None)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .to_tuple("jpg;jpeg;png", "json", handler=wds.handlers.warn_and_continue)
        .map_tuple(
            lambda img: img,
            lambda meta_bytes: (
                json.loads(meta_bytes) if isinstance(meta_bytes, (str, bytes))
                else meta_bytes
            ),
        )
    )

    def _apply_transform(sample):
        img, meta = sample
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))
        img = img.convert("RGB")
        if transform is not None:
            aug = transform(img)
        else:
            aug = img
        meta_dict = meta if isinstance(meta, dict) else {}
        return aug, meta_dict

    dataset = dataset.map(_apply_transform)

    if epoch_length is not None and epoch_length > 0:
        # with_epoch(n) limits each DataLoader worker independently.
        # With num_workers workers, the total is n * num_workers.
        # Divide the epoch length by the worker count to preserve the requested total.
        # See WebDataset issue #303.
        effective_workers = max(num_workers, 1)
        per_worker_epoch = max(1, epoch_length // effective_workers)
        dataset = dataset.with_epoch(per_worker_epoch)

    return dataset


def build_wds_dataloader(
    wds_root: str,
    transform: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    shard_list: Optional[str] = None,
    epoch_length: Optional[int] = None,
    shuffle_buffer: int = 5000,
    num_workers: int = 8,
    batch_size: int = 16,
    prefetch_factor: int = 4,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    import webdataset as wds
    from torch.utils.data import DataLoader

    dataset = build_wds_pipeline(
        wds_root=wds_root,
        transform=transform,
        shard_list=shard_list,
        epoch_length=epoch_length,
        shuffle_buffer=shuffle_buffer,
        num_workers=num_workers,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        seed=seed,
    )

    dl_kwargs = {}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        **dl_kwargs,
    )
    return loader
