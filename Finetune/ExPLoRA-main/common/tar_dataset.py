"""
Tar-sharded pathology dataset.

Motivation:
  - WSL2 reading millions of small patch files from /mnt/<drive> is often slow due to high per-file open/stat overhead.
  - Packing patches into a smaller number of .tar shard files drastically reduces filesystem metadata calls.

Format:
  - Shards: shard-000000.tar, shard-000001.tar, ...
  - Index: JSONL, one sample per line:
      {"shard":"shards/shard-000000.tar","member":"CenterA/Slide_001/x.png","center_root":"CenterA","slide_id":"Slide_001"}

This dataset returns (aug_dict, meta) to stay compatible with dinov2 collate_data_and_cast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict

import tarfile

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TarSampleRecord:
    shard: str
    member: str
    center_root: str
    slide_id: str


class _TarShardHandle:
    """
    Per-process shard handle with a name->TarInfo index.
    Intended to be used inside DataLoader workers (process-local).
    """

    def __init__(self, shard_path: str) -> None:
        self.shard_path = shard_path
        self._fp = open(shard_path, "rb")
        self._tar = tarfile.open(fileobj=self._fp, mode="r:*")
        members = self._tar.getmembers()
        self._name_to_info: Dict[str, tarfile.TarInfo] = {m.name: m for m in members if m.isfile()}

    def get_bytes(self, member_name: str) -> bytes:
        info = self._name_to_info.get(member_name)
        if info is None:
            raise KeyError(f"member not found in shard: {member_name} (shard={self.shard_path})")
        f = self._tar.extractfile(info)
        if f is None:
            raise RuntimeError(f"can not extract member: {member_name} (shard={self.shard_path})")
        try:
            return f.read()
        finally:
            f.close()

    def close(self) -> None:
        try:
            self._tar.close()
        finally:
            self._fp.close()


def _resolve_shard_path(shards_root: Optional[str], shard_field: str) -> str:
    p = Path(shard_field)
    if p.is_absolute():
        return str(p)
    if not shards_root:
        return str(p)
    return str(Path(shards_root) / p)


class TarShardedPathologyDataset(Dataset):
    """
    Dataset backed by tar shards + JSONL index.

    Returns:
      (aug_dict, meta)
        - aug_dict: output of transform(image) (same as MultiCenterPathologyDataset)
        - meta: {"path": "...", "slide_id": "...", "center_root": "...", "shard": "...", "member": "..."}
    """

    def __init__(
        self,
        index_path: str,
        *,
        shards_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        shard_cache_size: int = 32,
    ) -> None:
        self.index_path = str(index_path)
        self.shards_root = str(shards_root) if shards_root else None
        self.transform = transform
        self.shard_cache_size = int(max(shard_cache_size, 1))
        self._shard_cache: "OrderedDict[str, _TarShardHandle]" = OrderedDict()

        self.records: List[TarSampleRecord] = self._load_index(self.index_path, max_samples=max_samples)
        if len(self.records) == 0:
            raise RuntimeError(f"index is empty: {self.index_path}")

    @staticmethod
    def _load_index(path: str, max_samples: Optional[int]) -> List[TarSampleRecord]:
        records: List[TarSampleRecord] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"tar index not found: {path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(
                    TarSampleRecord(
                        shard=str(obj["shard"]),
                        member=str(obj["member"]),
                        center_root=str(obj.get("center_root", "")),
                        slide_id=str(obj.get("slide_id", "")),
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    break
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _get_shard_handle(self, shard_path: str) -> _TarShardHandle:
        h = self._shard_cache.get(shard_path)
        if h is not None:
            self._shard_cache.move_to_end(shard_path, last=True)
            return h
        h = _TarShardHandle(shard_path)
        self._shard_cache[shard_path] = h
        self._shard_cache.move_to_end(shard_path, last=True)
        while len(self._shard_cache) > self.shard_cache_size:
            _, old = self._shard_cache.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return h

    def _read_image(self, rec: TarSampleRecord) -> Image.Image:
        shard_path = _resolve_shard_path(self.shards_root, rec.shard)
        handle = self._get_shard_handle(shard_path)
        data = handle.get_bytes(rec.member)
        return Image.open(BytesIO(data)).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        rec = self.records[idx]
        img = self._read_image(rec)
        if self.transform is not None:
            aug = self.transform(img)
        else:
            aug = img
        meta = {
            "path": f"tar://{rec.shard}::{rec.member}",
            "slide_id": rec.slide_id,
            "center_root": rec.center_root,
            "shard": rec.shard,
            "member": rec.member,
        }
        return aug, meta

    def __del__(self) -> None:
        try:
            for h in list(self._shard_cache.values()):
                try:
                    h.close()
                except Exception:
                    pass
            self._shard_cache.clear()
        except Exception:
            pass
