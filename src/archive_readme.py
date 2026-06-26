from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .model_config import PROJECT_ROOT, README_FILE

ARCHIVE_DIR = PROJECT_ROOT / "readme_archive"
INDEX_FILE = ARCHIVE_DIR / "README_ARCHIVE_INDEX.md"

ARCHIVE_RE = re.compile(r"README_(\d{4})_")


def next_archive_number() -> int:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    nums: list[int] = []
    for p in ARCHIVE_DIR.glob("README_*.md"):
        m = ARCHIVE_RE.match(p.name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                pass
    return max(nums, default=0) + 1


def build_index() -> str:
    rows = []
    for p in sorted(ARCHIVE_DIR.glob("README_*.md")):
        m = ARCHIVE_RE.match(p.name)
        number = int(m.group(1)) if m else 0
        stat = p.stat()
        rows.append({
            "number": number,
            "file": p.name,
            "size_kb": stat.st_size / 1024,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        })

    lines = [
        "# README Archive",
        "",
        "This directory stores a numbered snapshot of the generated project README after each successful workflow run.",
        "The latest root `README.md` remains the current report; files here preserve previous reports for comparison and rollback.",
        "",
        "| number | file | size_kb | modified_utc |",
        "|---:|---|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: x["number"], reverse=True):
        lines.append(f"| {r['number']:04d} | [{r['file']}]({r['file']}) | {r['size_kb']:.1f} | {r['modified_utc']} |")
    if not rows:
        lines.append("| - | _No archived README yet._ | - | - |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not README_FILE.exists() or README_FILE.stat().st_size == 0:
        raise FileNotFoundError(f"README file not found or empty: {README_FILE}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    number = next_archive_number()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    archive_file = ARCHIVE_DIR / f"README_{number:04d}_{ts}.md"

    content = README_FILE.read_text(encoding="utf-8")
    archive_header = (
        f"<!-- README archive snapshot {number:04d}; generated at {ts}; "
        "source: root README.md -->\n\n"
    )
    archive_file.write_text(archive_header + content, encoding="utf-8")
    INDEX_FILE.write_text(build_index(), encoding="utf-8")
    print(f"Archived README to {archive_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
