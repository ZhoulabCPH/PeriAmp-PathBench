# make_patch_grid.py

import argparse
import random
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    # Package import (for example ``import heatmap.make_patch_grid``).
    from .tumor_region import (  # type: ignore[import-not-found]  # noqa: E402
        add_tumor_region_arguments,
        filter_image_paths,
        tumor_filter_kwargs,
    )
except ImportError:
    # Direct script execution from the ``heatmap`` directory.
    from tumor_region import (  # noqa: E402
        add_tumor_region_arguments,
        filter_image_paths,
        tumor_filter_kwargs,
    )

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ============================================================
# Default settings
# Command-line arguments override these defaults.
# ============================================================

DEFAULT_ANNOTATION_DIR = Path(
    r"outputs/interpretability/tumor_area_output"
)
DEFAULT_ANNOTATION_CSV = None                  # Prefer this annotation file when processing one slide.
DEFAULT_TUMOR_FILTER = True                    # Set False to process all patches.

DEFAULT_INPUT_DIR = (
    r"data/selected_patches"
)
DEFAULT_OUT_DIR = (
    r"outputs/interpretability/patch_grid"
)
DEFAULT_LABELS = ["positive", "negative"]
DEFAULT_TITLES = None
DEFAULT_TITLE_BGS = None
DEFAULT_TITLE_COLORS = None
DEFAULT_MODE = "auto"                          # {auto, single, cases}
DEFAULT_REQUIRE_ALL_LABELS = False

DEFAULT_CANVAS_BG = "#FFFFFF"
DEFAULT_BORDER_COLOR = "#333333"
DEFAULT_FONT_SIZE = 34
DEFAULT_PATCH_SIZE = 128
DEFAULT_GAP = 6
DEFAULT_TITLE_HEIGHT = 52
DEFAULT_BORDER_WIDTH = 3
DEFAULT_GRID_ROWS = 4
DEFAULT_GRID_COLS = 4
DEFAULT_SEED = 42
DEFAULT_RECURSIVE = False
DEFAULT_NO_RANDOM = False

# ============================================================
# PDF output settings
#
# Embed TrueType fonts.
# Keep PDF image compression enabled for large grids.
# Disable image compositing in Matplotlib.
# This keeps one editable image per patch where possible.
# ============================================================

mpl.rcParams["pdf.compression"] = 6
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["image.composite_image"] = False

# ============================================================


def collect_images(folder, recursive=False):
    folder = Path(folder)

    if not folder.exists():
        return []

    if recursive:
        files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    return sorted(files)


def load_font(font_size=32):
    """Load Arial from the platform's available font files."""
    arial_candidates = [
        "arial.ttf",
        "Arial.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    ]

    for font_file in arial_candidates:
        try:
            return ImageFont.truetype(font_file, font_size)
        except Exception:
            pass

    raise RuntimeError(
        'Arial was not found. Install Arial or make it available to the system font loader. On Windows, a typical path is:'
        "C:\\Windows\\Fonts\\arial.ttf"
    )


def safe_filename(name):
    """Convert a display label into a usable filename."""
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = name.strip()
    return name if name else "label"


def get_list_value(values, index, default):
    """Resolve a per-label setting from the command line. Broadcast one value to every label and use the fallback for missing entries."""
    if values is None or len(values) == 0:
        return default

    if len(values) == 1:
        return values[0]

    if index < len(values):
        return values[index]

    return default


def save_editable_pdf(
    chosen,
    pdf_path,
    title,
    title_bg,
    title_color,
    canvas_bg,
    border_color,
    font_size,
    patch_size,
    gap,
    title_height,
    border_width,
    grid_rows,
    grid_cols,
):
    """Write each patch as a separate image and keep titles, borders, and backgrounds as vectors. Match the PNG layout and set drawing order so the background does not cover the images."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    grid_w = grid_cols * patch_size + (grid_cols + 1) * gap
    grid_h = grid_rows * patch_size + (grid_rows + 1) * gap

    canvas_w = grid_w + 2 * border_width
    canvas_h = title_height + grid_h + 2 * border_width

    # Use 100 DPI for a direct mapping between data units and pixels.
    # DPI sets the physical page size for the PDF.
    layout_dpi = 100.0
    fig_w = canvas_w / layout_dpi
    fig_h = canvas_h / layout_dpi

    fig = plt.figure(
        figsize=(fig_w, fig_h),
        dpi=layout_dpi,
        facecolor=canvas_bg,
    )
    ax = fig.add_axes([0, 0, 1, 1])

    # Use pixel coordinates with the origin at the top left.
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.set_aspect("auto")
    ax.axis("off")

    # ---------------------------------------------------------
    # Draw the background below every patch.
    # ---------------------------------------------------------
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            canvas_w,
            canvas_h,
            facecolor=canvas_bg,
            edgecolor="none",
            linewidth=0,
            zorder=-10,
        )
    )

    # ---------------------------------------------------------
    # Patch images
    # Use the same offsets and spacing as the PNG.
    # ---------------------------------------------------------
    start_x = border_width + gap
    start_y = title_height + gap

    for idx, img_path in enumerate(chosen):
        row = idx // grid_cols
        col = idx % grid_cols

        x = start_x + col * (patch_size + gap)
        y = start_y + row * (patch_size + gap)

        # Let the PDF backend embed each image at its original resolution.
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_array = np.asarray(img)

        # Draw each patch above the background.
        ax.imshow(
            img_array,
            extent=[
                x,
                x + patch_size,
                y + patch_size,
                y,
            ],
            interpolation="none",
            aspect="auto",
            zorder=1,
        )

    # ---------------------------------------------------------
    # Draw the title bar above the patches.
    # Match the PNG title bar.
    # draw.rectangle([border_width, border_width,
    #                 canvas_w-border_width-1, title_height])
    # ---------------------------------------------------------
    title_bar_x = border_width
    title_bar_y = border_width
    title_bar_w = max(0, canvas_w - 2 * border_width)
    title_bar_h = max(0, title_height - border_width)

    ax.add_patch(
        plt.Rectangle(
            (title_bar_x, title_bar_y),
            title_bar_w,
            title_bar_h,
            facecolor=title_bg,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
    )

    # ---------------------------------------------------------
    # Draw the title text.
    #
    # Convert the PIL font size from pixels to Matplotlib points.
    # At 100 DPI, multiply pixels by 72 / 100.
    # ---------------------------------------------------------
    pdf_font_size = max(6.0, float(font_size) * 72.0 / layout_dpi)

    title_y = border_width + (title_height - border_width) / 2.0

    ax.text(
        canvas_w / 2.0,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=pdf_font_size,
        color=title_color,
        fontfamily="Arial",
        zorder=3,
    )

    # ---------------------------------------------------------
    # Draw the outer border as a stroke only.
    # ---------------------------------------------------------
    border_linewidth_pt = max(
        0.5,
        float(border_width) * 72.0 / layout_dpi,
    )

    ax.add_patch(
        plt.Rectangle(
            (border_width / 2.0, border_width / 2.0),
            max(0, canvas_w - border_width),
            max(0, canvas_h - border_width),
            facecolor="none",
            fill=False,
            edgecolor=border_color,
            linewidth=border_linewidth_pt,
            zorder=10,
        )
    )

    # Avoid tight bounding-box cropping.
    # Cropping would change the PDF margins relative to the PNG.
    fig.savefig(
        pdf_path,
        format="pdf",
        facecolor=canvas_bg,
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )

    plt.close(fig)


def make_grid(
    image_paths,
    output_prefix,
    title,
    title_bg="#B7E1A1",
    title_color="#F0F0F0",
    canvas_bg="#FFFFFF",
    border_color="#333333",
    font_size=34,
    patch_size=128,
    gap=6,
    title_height=52,
    border_width=3,
    grid_rows=4,
    grid_cols=4,
    seed=None,
    random_sample=True,
):
    need_n = grid_rows * grid_cols

    if len(image_paths) < need_n:
        raise ValueError(
            f"{title} needs {need_n} images and has only {len(image_paths)}."
        )

    if seed is not None:
        random.seed(seed)

    if random_sample:
        chosen = random.sample(image_paths, need_n)
    else:
        chosen = image_paths[:need_n]

    grid_w = grid_cols * patch_size + (grid_cols + 1) * gap
    grid_h = grid_rows * patch_size + (grid_rows + 1) * gap

    canvas_w = grid_w + 2 * border_width
    canvas_h = title_height + grid_h + 2 * border_width

    canvas = Image.new("RGB", (canvas_w, canvas_h), canvas_bg)
    draw = ImageDraw.Draw(canvas)

    # Outer border
    draw.rectangle(
        [0, 0, canvas_w - 1, canvas_h - 1],
        outline=border_color,
        width=border_width,
    )

    # Title bar
    draw.rectangle(
        [border_width, border_width, canvas_w - border_width - 1, title_height],
        fill=title_bg,
    )

    font = load_font(font_size)

    # Center the title.
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = (canvas_w - text_w) // 2
    text_y = border_width + (title_height - border_width - text_h) // 2 - 2

    draw.text((text_x, text_y), title, fill=title_color, font=font)

    # Patch area for the PNG
    start_x = border_width + gap
    start_y = title_height + gap

    for idx, img_path in enumerate(chosen):
        row = idx // grid_cols
        col = idx % grid_cols

        x = start_x + col * (patch_size + gap)
        y = start_y + row * (patch_size + gap)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize(
                (patch_size, patch_size),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(img, (x, y))

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    # Save the PNG with PIL.
    canvas.save(png_path, dpi=(300, 300))

    # Export an editable PDF with independent patch images.
    save_editable_pdf(
        chosen=chosen,
        pdf_path=pdf_path,
        title=title,
        title_bg=title_bg,
        title_color=title_color,
        canvas_bg=canvas_bg,
        border_color=border_color,
        font_size=font_size,
        patch_size=patch_size,
        gap=gap,
        title_height=title_height,
        border_width=border_width,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def has_label_dirs(base_dir, labels, require_all_labels=False):
    base_dir = Path(base_dir)

    exists = [(base_dir / label).is_dir() for label in labels]

    if require_all_labels:
        return all(exists)

    return any(exists)


def collect_cases(input_dir, labels, mode="auto", require_all_labels=False):
    """Read either one directory containing label folders or multiple case directories containing label folders. In auto mode, select single-case mode when labels are present at the input root."""
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    cases = []

    if mode == "auto":
        if has_label_dirs(
            input_dir,
            labels,
            require_all_labels=require_all_labels,
        ):
            mode = "single"
        else:
            mode = "cases"

    if mode == "single":
        label_dirs = {}

        for label in labels:
            label_dir = input_dir / label
            if label_dir.is_dir():
                label_dirs[label] = label_dir
            else:
                print(f"Skip label folder: {label_dir}: directory not found")

        if require_all_labels and len(label_dirs) < len(labels):
            raise RuntimeError(
                f"Single-case mode is missing label directories: {labels}"
            )

        if len(label_dirs) == 0:
            raise RuntimeError(
                f"Single-case mode found no requested label directories under {input_dir}: {labels}"
            )

        cases.append(
            {
                "name": input_dir.name,
                "case_dir": input_dir,
                "label_dirs": label_dirs,
            }
        )

    elif mode == "cases":
        for case_dir in sorted(input_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            label_dirs = {}

            for label in labels:
                label_dir = case_dir / label
                if label_dir.is_dir():
                    label_dirs[label] = label_dir

            if require_all_labels and len(label_dirs) < len(labels):
                print(
                    f"Skip case: {case_dir.name}: missing required label directories: {labels}"
                )
                continue

            if len(label_dirs) == 0:
                print(
                    f"Skip case: {case_dir.name}: no requested label directories found: {labels}"
                )
                continue

            cases.append(
                {
                    "name": case_dir.name,
                    "case_dir": case_dir,
                    "label_dirs": label_dirs,
                }
            )

    else:
        raise ValueError(f"Unknown directory mode: {mode}")

    return cases


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help='Input root containing label folders or per-case directories',
    )

    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help='Output directory',
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help='Label folder names, for example --labels positive negative tumor normal',
    )

    parser.add_argument(
        "--titles",
        nargs="*",
        default=DEFAULT_TITLES,
        help='One title per label. Defaults to the label names.',
    )

    parser.add_argument(
        "--title_bgs",
        nargs="*",
        default=DEFAULT_TITLE_BGS,
        help='One title background color per label, for example #B7E1A1 #F4B6B6',
    )

    parser.add_argument(
        "--title_colors",
        nargs="*",
        default=DEFAULT_TITLE_COLORS,
        help='One title text color per label, for example #F0F0F0 #F0F0F0',
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "single", "cases"],
        default=DEFAULT_MODE,
        help=(
            'Choose auto to detect the layout or single for a root containing label folders. '
            'Use cases for a root containing case directories.'
        ),
    )

    parser.add_argument(
        "--require_all_labels",
        action="store_true",
        default=DEFAULT_REQUIRE_ALL_LABELS,
        help='Skip cases that lack any of the requested labels',
    )

    parser.add_argument("--canvas_bg", default=DEFAULT_CANVAS_BG)
    parser.add_argument("--border_color", default=DEFAULT_BORDER_COLOR)

    parser.add_argument("--font_size", type=int, default=DEFAULT_FONT_SIZE)

    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--gap", type=int, default=DEFAULT_GAP)
    parser.add_argument("--title_height", type=int, default=DEFAULT_TITLE_HEIGHT)
    parser.add_argument("--border_width", type=int, default=DEFAULT_BORDER_WIDTH)

    parser.add_argument(
        "--grid_rows",
        type=int,
        default=DEFAULT_GRID_ROWS,
        help='Grid rows',
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=DEFAULT_GRID_COLS,
        help='Grid columns',
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=DEFAULT_RECURSIVE,
        help='Search each label directory recursively for images',
    )
    parser.add_argument(
        "--no_random",
        action="store_true",
        default=DEFAULT_NO_RANDOM,
        help='Use the first grid_rows * grid_cols images in sorted order',
    )

    add_tumor_region_arguments(
        parser,
        single_csv=True,
        annotation_dir_default=DEFAULT_ANNOTATION_DIR,
        annotation_csv_default=DEFAULT_ANNOTATION_CSV,
        tumor_filter_default=DEFAULT_TUMOR_FILTER,
    )
    parser.set_defaults(tumor_filter=DEFAULT_TUMOR_FILTER)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels

    default_bg_palette = [
        "#B7E1A1",
        "#F4B6B6",
        "#B6D7F4",
        "#F4D6A6",
        "#D6B6F4",
        "#B6F4E8",
        "#E8E8E8",
    ]

    cases = collect_cases(
        input_dir=input_dir,
        labels=labels,
        mode=args.mode,
        require_all_labels=args.require_all_labels,
    )

    if len(cases) == 0:
        raise RuntimeError(
            f"No matching label directories found under {input_dir}: {labels}"
        )

    print(f"Found {len(cases)} cases.")
    print(f"Labels: {labels}")
    print(f"Grid size: {args.grid_rows} x {args.grid_cols}")

    for case in cases:
        case_name = case["name"]
        label_dirs = case["label_dirs"]

        print(f"\nProcessing case: {case_name}")

        case_out_dir = out_dir / case_name
        case_out_dir.mkdir(parents=True, exist_ok=True)

        for label_index, label in enumerate(labels):
            if label not in label_dirs:
                print(f"Skip label: {case_name} / {label}: directory not found")
                continue

            label_dir = label_dirs[label]

            title = get_list_value(
                args.titles,
                label_index,
                label.capitalize(),
            )

            title_bg = get_list_value(
                args.title_bgs,
                label_index,
                default_bg_palette[label_index % len(default_bg_palette)],
            )

            title_color = get_list_value(
                args.title_colors,
                label_index,
                "#F0F0F0",
            )

            print(f"Label: {label}")
            print(f"Input dir: {label_dir}")

            image_paths = collect_images(
                label_dir,
                recursive=args.recursive,
            )

            try:
                image_paths = filter_image_paths(
                    image_paths,
                    case_name,
                    **tumor_filter_kwargs(args),
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"Skip label: {case_name} / {label}. Reason: {e}")
                continue

            output_prefix = (
                case_out_dir
                / (
                    f"{case_name}_{safe_filename(label)}_"
                    f"{args.grid_rows}x{args.grid_cols}_patch_grid"
                )
            )

            try:
                make_grid(
                    image_paths=image_paths,
                    output_prefix=output_prefix,
                    title=title,
                    title_bg=title_bg,
                    title_color=title_color,
                    canvas_bg=args.canvas_bg,
                    border_color=args.border_color,
                    font_size=args.font_size,
                    patch_size=args.patch_size,
                    gap=args.gap,
                    title_height=args.title_height,
                    border_width=args.border_width,
                    grid_rows=args.grid_rows,
                    grid_cols=args.grid_cols,
                    seed=args.seed,
                    random_sample=not args.no_random,
                )
            except ValueError as e:
                print(f"Skip label: {case_name} / {label}. Reason: {e}")


if __name__ == "__main__":
    main()
