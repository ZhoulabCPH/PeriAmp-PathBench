import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PatchRecord:
    center_root: str
    slide_id: str
    path: str


class MultiCenterPathologyDataset(Dataset):
    """Discover PNG patches recursively under multiple roots. A typical layout is root/slide_id/patch.png."""

    def __init__(
        self,
        root_dirs: Sequence[str],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png",),
        max_samples: Optional[int] = None,
    ):
        self.root_dirs = [str(Path(p)) for p in root_dirs]
        self.transform = transform
        self.extensions = tuple(e.lower() for e in extensions)

        for p in self.root_dirs:
            if not Path(p).exists():
                raise FileNotFoundError(f"Input root does not exist: {p}")

        self.records: List[PatchRecord] = self._discover(
            self.root_dirs,
            extensions=self.extensions,
            max_samples=max_samples,
        )
        if len(self.records) == 0:
            raise RuntimeError(
                'No PNG patches found. Check root_dirs and the root/slide_id/patch.png layout.'
            )

    @staticmethod
    def _discover(
        root_dirs: Iterable[str],
        extensions: Tuple[str, ...],
        max_samples: Optional[int] = None,
    ) -> List[PatchRecord]:
        records: List[PatchRecord] = []
        for root in root_dirs:
            root_path = Path(root)
            center_root = root_path.name
            for slide_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
                slide_id = slide_dir.name
                for img_path in slide_dir.rglob("*"):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in extensions:
                        continue
                    records.append(PatchRecord(center_root=center_root, slide_id=slide_id, path=str(img_path)))
                    if max_samples is not None and len(records) >= max_samples:
                        return records
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec.path).convert("RGB")
        if self.transform is not None:
            aug = self.transform(img)
        else:
            aug = img
        return aug, {"path": rec.path, "slide_id": rec.slide_id, "center_root": rec.center_root}
