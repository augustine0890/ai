"""compress_courses.py

Compresses each course folder inside ~/Downloads/365DataScience into a
separate .zip file sitting alongside the course folder.

Compression strategy:
  - Already-compressed media (.mp4, .mkv, .webm, …) → STORE (no re-compression;
    avoids bloating files and wastes no CPU time).
  - Text-based files (.html, .txt, .vtt, .srt, .json, …) → LZMA (best ratio in
    Python's stdlib; typically 70-90% size reduction on text).
  - Everything else → DEFLATE level 9 (safe default).

Usage:
    uv run python compress_courses.py                     # compress all courses
    uv run python compress_courses.py "Intro to Revenue"  # partial name match
    uv run python compress_courses.py --list              # list courses + sizes
"""

import sys
import zipfile
import datetime
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path.home() / "Downloads" / "365DataScience"

# Already-compressed: re-compressing wastes CPU and gains nothing (or bloats).
STORE_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".avi", ".mov",
    ".mp3", ".aac", ".ogg", ".flac", ".m4a",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic",
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf",
}

# High-compressibility text: LZMA gives 60-90% reduction.
LZMA_EXTENSIONS = {
    ".html", ".htm", ".txt", ".vtt", ".srt",
    ".json", ".xml", ".csv", ".md", ".py", ".js", ".css",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def pick_compression(path: Path) -> tuple[int, int]:
    """Return (compression_method, compress_level) for a given file path."""
    ext = path.suffix.lower()
    if ext in STORE_EXTENSIONS:
        return zipfile.ZIP_STORED, 0
    if ext in LZMA_EXTENSIONS:
        return zipfile.ZIP_LZMA, 0   # LZMA ignores level in zipfile
    return zipfile.ZIP_DEFLATED, 9   # max DEFLATE for everything else


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


def compress_course(course_dir: Path, out_dir: Path) -> Path:
    """Zip a single course folder and return the path of the created zip."""
    zip_path = out_dir / f"{course_dir.name}.zip"
    all_files = [f for f in course_dir.rglob("*") if f.is_file()]
    total = len(all_files)
    original_size = sum(f.stat().st_size for f in all_files)

    log(f"  '{course_dir.name}'  ({total} files, {human_size(original_size)} uncompressed)")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for i, file_path in enumerate(all_files, start=1):
            method, level = pick_compression(file_path)
            arcname = file_path.relative_to(course_dir.parent)
            if method == zipfile.ZIP_DEFLATED:
                zf.write(file_path, arcname, compress_type=method, compresslevel=level)
            else:
                zf.write(file_path, arcname, compress_type=method)
            if i % 20 == 0 or i == total:
                print(f"\r    {i}/{total} files...", end="", flush=True)

    compressed_size = zip_path.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100 if original_size else 0
    print(
        f"\r    ✅ {human_size(original_size)} → {human_size(compressed_size)}"
        f"  ({ratio:.1f}% smaller)  →  {zip_path.name}"
    )
    return zip_path


def list_courses(base_dir: Path) -> list[Path]:
    return sorted(d for d in base_dir.iterdir() if d.is_dir())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BASE_DIR.exists():
        print(f"❌  Base directory not found: {BASE_DIR}")
        sys.exit(1)

    args = sys.argv[1:]

    courses = list_courses(BASE_DIR)

    # --list: show sizes then exit
    if "--list" in args:
        if not courses:
            print("No course folders found.")
            return
        print(f"Found {len(courses)} course(s) in {BASE_DIR}:\n")
        for c in courses:
            files = [f for f in c.rglob("*") if f.is_file()]
            total_size = sum(f.stat().st_size for f in files)
            print(f"  📁 {c.name}  ({len(files)} files, {human_size(total_size)})")
        return

    if not courses:
        print("No course folders found.")
        return

    # Partial name filter
    if args:
        query = " ".join(args).lower()
        courses = [c for c in courses if query in c.name.lower()]
        if not courses:
            print(f"❌  No course matching '{query}' found.")
            sys.exit(1)

    log(f"Starting compression of {len(courses)} course(s)")
    print("    Strategy: LZMA for text/HTML/VTT  ·  STORE for video/audio  ·  DEFLATE-9 for others\n")

    created: list[Path] = []
    total_original = 0
    total_compressed = 0

    for course_dir in courses:
        try:
            original = sum(f.stat().st_size for f in course_dir.rglob("*") if f.is_file())
            zip_path = compress_course(course_dir, BASE_DIR)
            compressed = zip_path.stat().st_size
            total_original += original
            total_compressed += compressed
            created.append(zip_path)
        except Exception as exc:
            log(f"  ⚠️  Failed: {course_dir.name} — {exc}")

    overall_ratio = (1 - total_compressed / total_original) * 100 if total_original else 0
    print()
    log(f"✅  Done! {len(created)} zip file(s) created.")
    log(f"   Total: {human_size(total_original)} → {human_size(total_compressed)}"
        f"  ({overall_ratio:.1f}% overall reduction)")


if __name__ == "__main__":
    main()
