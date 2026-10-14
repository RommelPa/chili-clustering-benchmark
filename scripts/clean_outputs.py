"""Remove only known generated outputs, never raw data or MLflow history."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "data/processed": ("*.csv", "*.txt", "*.log"),
    "outputs/tables": ("*.csv",),
    "outputs/figures": ("*.png", "*.svg"),
    "outputs/reports": ("*.md", "*.json"),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--apply", action="store_true", help="Delete listed generated files")
args = parser.parse_args()
for directory, patterns in PATTERNS.items():
    for pattern in patterns:
        for path in sorted((ROOT / directory).glob(pattern)):
            if path.is_file() and not path.is_symlink():
                print(path.relative_to(ROOT))
                if args.apply:
                    path.unlink()
