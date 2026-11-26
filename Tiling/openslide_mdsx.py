"""Report the OpenSlide format and dimensions of user-supplied slide files."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slides", type=Path, nargs="+", help="Slide files to inspect")
    args = parser.parse_args()

    dll_dir = os.environ.get("OPENSLIDE_DLL_DIR")
    dll_handle = os.add_dll_directory(dll_dir) if os.name == "nt" and dll_dir else None
    try:
        import openslide

        print(f"OpenSlide version: {openslide.__version__}")
        failed = False
        for path in args.slides:
            if not path.is_file():
                print(f"File not found: {path}")
                failed = True
                continue
            try:
                with openslide.OpenSlide(str(path)) as slide:
                    print(f"Opened: {path}")
                    print(f"  Format: {slide.detect_format(str(path))}")
                    print(f"  Levels: {slide.level_count}")
                    print(f"  Level-0 dimensions: {slide.dimensions}")
                    print(f"  Sample property keys: {list(slide.properties)[:3]}")
            except openslide.OpenSlideError as exc:
                print(f"OpenSlide error for {path}: {exc}")
                failed = True
        return int(failed)
    finally:
        if dll_handle is not None:
            dll_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
