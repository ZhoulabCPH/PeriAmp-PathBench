"""
Samplers for ExPLoRA pathology training.

The tar-sharded dataset (`TarShardedPathologyDataset`) benefits a lot from shard locality:
randomly bouncing across many .tar shards causes heavy open/seek/decode overhead.

`ShardGroupedDistributedSampler` assigns shards to ranks (balanced by sample counts), then yields
indices grouped by shard per-rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class _ShardBucket:
    shard: str
    indices: List[int]


def _build_shard_buckets(records: Sequence, shard_attr: str = "shard") -> List[_ShardBucket]:
    by: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        shard = getattr(r, shard_attr)
        by.setdefault(str(shard), []).append(i)
    return [_ShardBucket(shard=s, indices=idxs) for s, idxs in by.items()]


def _assign_shards_to_ranks(buckets: List[_ShardBucket], world_size: int) -> List[List[_ShardBucket]]:
    buckets = sorted(buckets, key=lambda b: len(b.indices), reverse=True)
    ranks: List[List[_ShardBucket]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    for b in buckets:
        r = min(range(world_size), key=lambda i: loads[i])
        ranks[r].append(b)
        loads[r] += len(b.indices)
    return ranks


class ShardGroupedDistributedSampler(Sampler[int]):
    """
    Distributed sampler that preserves shard locality.

    Requirements:
      - dataset.records exists and each record has `.shard` attribute (see TarSampleRecord).
    """

    def __init__(
        self,
        *,
        records: Sequence,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.records = records
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        buckets = _build_shard_buckets(records)
        self._rank_buckets = _assign_shards_to_ranks(buckets, self.num_replicas)

        self._lens = [sum(len(b.indices) for b in rb) for rb in self._rank_buckets]
        self._max_len = max(self._lens) if self._lens else 0

        if self.drop_last:
            self._num_samples = min(self._lens) if self._lens else 0
        else:
            self._num_samples = self._max_len

    def __len__(self) -> int:
        return int(self._num_samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        my = list(self._rank_buckets[self.rank])
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            if len(my) > 1:
                order = torch.randperm(len(my), generator=g).tolist()
                my = [my[i] for i in order]

        out: List[int] = []
        for b in my:
            idxs = list(b.indices)
            if self.shuffle and len(idxs) > 1:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]
            out.extend(idxs)

        if self.drop_last:
            return iter(out[: self._num_samples])

        if len(out) == 0:
            return iter([])
        if len(out) < self._num_samples:
            need = self._num_samples - len(out)
            out.extend(out[:need])
        else:
            out = out[: self._num_samples]
        return iter(out)
