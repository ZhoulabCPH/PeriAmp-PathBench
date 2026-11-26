"""
Summarize a PyTorch .pth or .pt checkpoint.
Use --ckpt to provide a checkpoint. The script prints top-level keys, nested structures, and example state-dict keys to help determine whether the checkpoint can resume ExPLoRA or MAE fine-tuning. It checks structure without constructing the model.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Tuple


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def _keys_preview(d: Dict[str, Any], limit: int = 30) -> List[str]:
    ks = list(d.keys())
    ks_sorted = sorted(ks)
    if len(ks_sorted) <= limit:
        return ks_sorted
    return ks_sorted[:limit] + [f"... (+{len(ks_sorted) - limit} more)"]


def _iter_items_preview(it: Iterable[str], limit: int) -> List[str]:
    out: List[str] = []
    for i, k in enumerate(it):
        if i >= limit:
            break
        out.append(k)
    return out


def _guess_state_dict(obj: Any) -> Tuple[str, Dict[str, Any] | None]:
    """Return the detected state-dict kind and tensors for Virchow2, MAE fine-tuning, plain state dicts, or unknown layouts."""
    if not _is_mapping(obj):
        return "unknown", None

    # Virchow2 checkpoints store backbone and head state dictionaries under model and teacher.
    if "model" in obj and _is_mapping(obj["model"]):
        m = obj["model"]
        if "backbone" in m and _is_mapping(m["backbone"]):
            return "virchow2.backbone", m["backbone"]
        if "head" in m and _is_mapping(m["head"]):
            return "virchow2.head", m["head"]

    if "teacher" in obj and _is_mapping(obj["teacher"]):
        t = obj["teacher"]
        if "backbone" in t and _is_mapping(t["backbone"]):
            return "virchow2.teacher.backbone", t["backbone"]
        if "head" in t and _is_mapping(t["head"]):
            return "virchow2.teacher.head", t["head"]

    # Fine-tuning, MAE, and linear probing checkpoints typically include model, optimizer, and epoch.
    if "model" in obj and _is_mapping(obj["model"]):
        return "finetune.model", obj["model"]

    # Plain state dictionaries map parameter names directly to tensors.
    # Parameter names typically contain a dot, and values are usually tensors.
    keys = list(obj.keys())
    if keys and isinstance(keys[0], str) and "." in keys[0]:
        return "plain_state_dict", obj  # type: ignore[return-value]

    return "unknown", None


def _has_lora_keys(sd: Dict[str, Any]) -> bool:
    # Look for typical LoRA parameter names.
    lora_markers = ("lora", "lora_A", "lora_B", "lora_down", "lora_up")
    for k in sd.keys():
        lk = k.lower()
        if any(m in lk for m in lora_markers):
            return True
    return False


def _schema_check_virchow2(ckpt: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    # The training script's try_resume() requires these keys.
    for k in ("model", "teacher", "optimizer"):
        if k not in ckpt:
            missing.append(k)
    if "model" in ckpt and _is_mapping(ckpt["model"]):
        m = ckpt["model"]
        for k in ("backbone", "head"):
            if k not in m:
                missing.append(f"model.{k}")
    else:
        if "model" in ckpt:
            missing.append("model(backbone/head dict expected)")
    if "teacher" in ckpt and _is_mapping(ckpt["teacher"]):
        t = ckpt["teacher"]
        for k in ("backbone", "head"):
            if k not in t:
                missing.append(f"teacher.{k}")
    else:
        if "teacher" in ckpt:
            missing.append("teacher(backbone/head dict expected)")
    ok = len(missing) == 0
    return ok, missing


def _schema_check_finetune(ckpt: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    m = ckpt.get("model")
    if "model" not in ckpt or not _is_mapping(m):
        missing.append("model(state_dict expected)")
        return False, missing

    # MAE fine-tuning expects checkpoint["model"] to be a model state dictionary.
    # Keys are strings, usually with dots.
    # Values are tensors or parameters.
    # A nested Virchow2 backbone and head use a different checkpoint layout.
    if "backbone" in m or "head" in m:
        # Confirm the nested layout used by Virchow2.
        if _is_mapping(m.get("backbone")) or _is_mapping(m.get("head")):
            missing.append("model(looks like virchow2 nested dict: {backbone, head})")
            return False, missing

    # Sample value types to distinguish state dictionaries from other mappings.
    sample_keys = list(m.keys())[:20]
    if not sample_keys:
        missing.append("model(empty state_dict?)")
        return False, missing
    for k in sample_keys:
        v = m.get(k)
        if isinstance(v, dict):
            missing.append(f"model[{k!r}] is dict (nested), not a Tensor state_dict")
            return False, missing

    return True, []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"weights/model_checkpoint.pth", type=str, help='Path to the .pth or .pt checkpoint')
    ap.add_argument("--keys_limit", default=10000, type=int, help='Maximum number of keys to print')
    ap.add_argument("--sd_keys_limit", default=50, type=int, help='Maximum number of state-dict keys to print')
    args = ap.parse_args()

    ckpt_path = os.path.expanduser(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    import torch

    obj = torch.load(ckpt_path, map_location="cpu")

    print("=== checkpoint basic ===")
    print(f"path: {ckpt_path}")
    print(f"type: {type(obj)}")

    if not _is_mapping(obj):
        print('The file does not contain a checkpoint or state-dict mapping.')
        return 2

    ckpt: Dict[str, Any] = obj
    print(f"top_level_keys({len(ckpt)}): {', '.join(_keys_preview(ckpt, limit=args.keys_limit))}")

    # Common checkpoint fields
    for k in ("epoch", "global_step", "args"):
        if k in ckpt:
            v = ckpt.get(k)
            if k == "args" and _is_mapping(v):
                print(f"{k}: dict ({len(v)} keys)")
            else:
                print(f"{k}: {v!r}")

    # Identify the checkpoint layout.
    kind, sd = _guess_state_dict(ckpt)
    print("\n=== structure guess ===")
    print(f"guess: {kind}")

    # Inspect the Virchow2 layout.
    v_ok, v_missing = _schema_check_virchow2(ckpt)
    f_ok, f_missing = _schema_check_finetune(ckpt)

    print("\n=== compatibility (by schema) ===")
    print(f"virchow2/train_virchow2_ddp.py try_resume(): {'OK' if v_ok else 'NO'}")
    if not v_ok:
        print(f"  missing: {', '.join(v_missing)}")
    print(f"finetune/util/misc.py resume: {'OK' if f_ok else 'NO'}")
    if not f_ok:
        print(f"  missing: {', '.join(f_missing)}")

    # Show sample state-dict keys.
    if sd is not None:
        print("\n=== state_dict preview ===")
        sd_keys = list(sd.keys())
        print(f"state_dict_keys({len(sd_keys)}), first_{min(len(sd_keys), args.sd_keys_limit)}:")
        for k in _iter_items_preview(sd_keys, args.sd_keys_limit):
            print(f"  - {k}")
        print(f"has_lora_like_keys: {_has_lora_keys(sd)}")

    # Show nested state dictionaries for Virchow2 checkpoints.
    if "model" in ckpt and _is_mapping(ckpt["model"]):
        m = ckpt["model"]
        print("\n=== model subtree ===")
        print(f"model keys: {', '.join(_keys_preview(m, limit=args.keys_limit))}")
        if "backbone" in m and _is_mapping(m["backbone"]):
            bb = m["backbone"]
            bb_keys = list(bb.keys())
            print(f"model.backbone state_dict keys: {len(bb_keys)} (showing up to {min(len(bb_keys), 20)})")
            for k in _iter_items_preview(bb_keys, 20):
                print(f"  - {k}")
        if "head" in m and _is_mapping(m["head"]):
            hd = m["head"]
            hd_keys = list(hd.keys())
            print(f"model.head state_dict keys: {len(hd_keys)} (showing up to {min(len(hd_keys), 20)})")
            for k in _iter_items_preview(hd_keys, 20):
                print(f"  - {k}")

    if "teacher" in ckpt and _is_mapping(ckpt["teacher"]):
        t = ckpt["teacher"]
        print("\n=== teacher subtree ===")
        print(f"teacher keys: {', '.join(_keys_preview(t, limit=args.keys_limit))}")

    print("\n=== suggestion ===")
    if v_ok:
        print('This checkpoint matches the Virchow2 resume layout. Use:')
        print('  python -m virchow2.train_virchow2_ddp --config <cfg.json> --resume_path "<this.pth>"')
    elif f_ok:
        print('This checkpoint resembles the MAE fine-tuning layout.')
        print('Virchow2 resume may require converting the keys and nested layout.')
    else:
        print('The top-level keys do not match either supported resume layout.')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


