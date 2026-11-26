import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_log(log_path):
    pattern = re.compile(
        r"\[train\]\s+epoch=(?P<epoch>\d+)\s+"
        r"step=(?P<step>\d+)/(?P<max_step>\d+)\s+"
        r"lr=(?P<lr>[0-9.eE+-]+)\s+"
        r"loss=(?P<loss>[-0-9.]+)\s+"
        r"\(dino_local=(?P<dino_local>[-0-9.]+),\s+"
        r"dino_global=(?P<dino_global>[-0-9.]+),\s+"
        r"ibot=(?P<ibot>[-0-9.]+),\s+"
        r"koleo=(?P<koleo>[-0-9.]+)\)"
    )

    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                row = m.groupdict()
                rows.append({
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "max_step": int(row["max_step"]),
                    "lr": float(row["lr"]),
                    "total_loss": float(row["loss"]),
                    "dino_local": float(row["dino_local"]),
                    "dino_global": float(row["dino_global"]),
                    "ibot": float(row["ibot"]),
                    "koleo": float(row["koleo"]),
                })

    if not rows:
        raise ValueError('No training loss records were found. Check the log format.')

    return pd.DataFrame(rows)


def plot_loss_pdf(df, output_pdf):
    curves = [
        ("total_loss", "Total Loss"),
        ("dino_local", "DINO Local Loss"),
        ("dino_global", "DINO Global Loss"),
        ("ibot", "iBOT Loss"),
        ("koleo", "KoLeo Loss"),
    ]

    with PdfPages(output_pdf) as pdf:
        for col, title in curves:
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(df["step"], df[col], linewidth=1.8)
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=r"outputs/training/train.log", help='Training log path')
    parser.add_argument("--out", default="loss_curves.pdf", help='Output PDF filename')
    parser.add_argument("--csv", default="loss_csv.csv", help='Optional CSV for parsed records')
    args = parser.parse_args()

    df = parse_log(args.log)

    if args.csv:
        df.to_csv(args.csv, index=False)

    plot_loss_pdf(df, args.out)

    print(f"Read {len(df)} training records.")
    print(f"Saved PDF to {args.out}")
    if args.csv:
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
