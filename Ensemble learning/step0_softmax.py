"""
Convert base-model logits to probabilities with softmax.
Read CSV_data/<MIL>/<Backbone>.csv and write 01_softmax/probability_csv/<MIL>__<Backbone>.csv under --output_dir. models.json lists the discovered models. Use --mils to select a comma-separated subset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

from hybridb_common import (
    discover_model_csvs,
    ensure_dir,
    now_tag,
    parse_mils_arg,
)
from softmax_probs import run_softmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default=r"CSV_data\top3 models")
    p.add_argument("--output_dir", type=str, default=r"Ensemble_Results\3 models")
    # Use --force to recompute existing CSVs.
    p.add_argument("--force", action="store_true")
    p.add_argument("--mils", type=str, default="", help='Comma-separated MIL names, for example ABMIL,CLAM,TransMIL,WiKG')
    p.add_argument("--quiet", action="store_true", help='Reduce console output')
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else (Path("Ensemble_Results") / f"hybridb_run_{now_tag()}")
    ensure_dir(out_dir)

    mils = parse_mils_arg(args.mils)
    refs = discover_model_csvs(input_dir, mils_filter=mils)

    probability_dir = out_dir / "01_softmax" / "probability_csv"
    ensure_dir(probability_dir)

    model_to_probs: Dict[str, str] = {}
    total = len(refs)
    skipped = 0
    processed = 0
    t0_all = time.time()
    if not args.quiet:
        print(f"[Step 0] Found {total} model CSVs. force={args.force}")
    for ref in refs:
        out_csv = probability_dir / f"{ref.model_id}.csv"
        model_to_probs[ref.model_id] = str(out_csv)
        if out_csv.exists() and (not args.force):
            skipped += 1
            if not args.quiet:
                print(f"[Step0] skip ({processed+skipped}/{total}) {ref.model_id}  -> Already exists: {out_csv.name}")
            continue
        processed += 1
        t0 = time.time()
        if not args.quiet:
            print(f"[Step0] run  ({processed+skipped}/{total}) {ref.model_id}  input={ref.path.name}")
        run_softmax(
            input_csv=str(ref.path),
            output_csv=str(out_csv),
        )
        if not args.quiet:
            dt = time.time() - t0
            print(f"[Step0] done ({processed+skipped}/{total}) {ref.model_id}  cost={dt:.1f}s  out={out_csv.name}")

    with open(out_dir / "models.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "model_id": r.model_id,
                    "mil": r.mil,
                    "backbone": r.backbone,
                    "path": str(r.path),
                    "probability_csv": model_to_probs[r.model_id],
                }
                for r in refs
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    if not args.quiet:
        dt_all = time.time() - t0_all
        print(f"[Step0] done. processed={processed}, skipped={skipped}, total={total}, cost={dt_all:.1f}s")
    print(f"[Step0] output_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()


