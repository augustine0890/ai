# -*- coding: utf-8 -*-
"""compress_courses.py

Compresses each course folder inside ~/Downloads/365DataScience into a
separate .zip file sitting alongside the course folder.

Compression strategy:
  - Already-compressed media (.mp4, .mkv, .webm, â€¦) â†’ STORE (no re-compression;
    avoids bloating files and wastes no CPU time).
  - Text-based files (.html, .txt, .vtt, .srt, .json, â€¦) â†’ LZMA (best ratio in
    Python's stdlib; typically 70-90% size reduction on text).
  - Everything else â†’ DEFLATE level 9 (safe default).

Optional video transcoding (--transcode):
  - Re-encodes .mp4 files to H.265 (CRF 26) before zipping.
  - Skips files already encoded in H.265/HEVC.
  - Transcodes to a temp file first, then replaces the original on success.
  - Typically reduces video size by 40-50% for lecture/screencast content.

Usage:
    uv run python -m dds.datascience365.compress_courses                          # compress all courses
    uv run python -m dds.datascience365.compress_courses "Intro to Revenue"       # partial name match
    uv run python -m dds.datascience365.compress_courses --list                   # list courses + sizes
    uv run python -m dds.datascience365.compress_courses --transcode              # transcode videos then zip
    uv run python -m dds.datascience365.compress_courses --transcode "SQL"        # transcode + zip matching course
    uv run python -m dds.datascience365.compress_courses --transcode --crf 28     # custom CRF value
"""

import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from dds.common.logger import log_event

# Force UTF-8 output on Windows so emoji/Unicode in print() don't crash.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

# Video extensions eligible for H.265 transcoding.
TRANSCODE_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

DEFAULT_CRF = 26

# Folder name tags used to track compression state.
# Folders with ARCHIVED_TAG are skipped on next run (already done).
# Folders with COMPRESSING_TAG were interrupted mid-run and will be resumed.
ARCHIVED_TAG = "[archived] "
COMPRESSING_TAG = "[â€¢] "


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_MODULE = "dds.compress"


def _write_trace(event: str, **fields: Any) -> None:
    log_event(_MODULE, event, **fields)


def log(msg: str, event: str = "compress_log") -> None:
    """Print timestamped message to console and write a trace entry."""
    import datetime
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")
    log_event(_MODULE, event, message=msg)


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


def get_ffmpeg() -> str:
    """Return path to the ffmpeg binary, preferring system ffmpeg then imageio fallback."""
    import shutil
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg  # pyright: ignore[reportMissingImports]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg not found. Install it or run: uv sync") from exc


# Encoder candidates in priority order (fastest first).
# Each entry: (encoder_name, extra_args_fn(crf) -> list[str], label)
# Hardware encoders are 5-20x faster than libx265 at comparable quality.
_ENCODER_CANDIDATES = [
    # NVIDIA GPU (NVENC) â€” fastest on most Windows/Linux machines with an NVIDIA card
    ("hevc_nvenc",       lambda crf: ["-rc", "vbr", "-cq", str(crf), "-preset", "p4", "-b:v", "0"], "NVIDIA NVENC"),
    # Intel Quick Sync â€” built into Intel CPUs with integrated graphics
    ("hevc_qsv",         lambda crf: ["-global_quality", str(crf), "-preset", "faster"],             "Intel QSV"),
    # AMD AMF â€” AMD GPUs (Windows/Linux)
    ("hevc_amf",         lambda crf: ["-quality", "speed", "-qp_i", str(crf), "-qp_p", str(crf)],   "AMD AMF"),
    # Apple VideoToolbox â€” Apple Silicon / macOS
    ("hevc_videotoolbox", lambda crf: ["-q:v", str(max(0, min(100, (51 - crf) * 2)))],              "Apple VideoToolbox"),
    # Software fallback â€” always available, slowest
    ("libx265",          lambda crf: ["-crf", str(crf), "-preset", "ultrafast"],                     "libx265 (software)"),
]


def detect_encoder(ffmpeg: str, crf: int) -> tuple[str, list[str], str]:
    """
    Probe available encoders and return (encoder, extra_args, label) for the
    fastest one supported on this machine.  Runs a 1-second test encode to
    confirm each hardware encoder actually works before committing to it.
    """
    import tempfile, os
    # Build a tiny silent test source (1s, 64x64 lavfi testsrc)
    test_base = [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=1",
                 "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1"]

    for encoder, extra_fn, label in _ENCODER_CANDIDATES:
        extra = extra_fn(crf)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tmp = tf.name
        try:
            r = subprocess.run(
                test_base + ["-c:v", encoder] + extra + ["-c:a", "aac", tmp],
                capture_output=True, timeout=15,
            )
            if r.returncode == 0 and Path(tmp).stat().st_size > 0:
                log(f"  Encoder selected: {label}")
                return encoder, extra_fn(crf), label
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    raise RuntimeError("No H.265 encoder found. Make sure ffmpeg has libx265 support.")


def is_hevc(video_path: Path, ffmpeg: str) -> bool:
    """Return True if the video is already encoded with H.265/HEVC."""
    try:
        result = subprocess.run(
            [ffmpeg, "-i", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        # ffmpeg prints stream info to stderr
        return "hevc" in result.stderr.lower() or "h265" in result.stderr.lower()
    except Exception:
        return False


# Return type codes for transcode_video():
#   (int, int)  â†’ success: (original_bytes, new_bytes)
#   (None, str) â†’ no-op:   (None, reason)  e.g. "already_hevc" | "wrong_ext"
#                           | "ffmpeg_error" | "timeout" | "zero_output" | "exception"

def transcode_video(
    video_path: Path,
    ffmpeg: str,
    encoder: str,
    encoder_args: list[str],
) -> "tuple[int, int] | tuple[None, str]":
    """
    Transcode video_path to H.265 in-place using the given encoder.
    Returns (original_bytes, new_bytes) on success.
    Returns (None, reason_str) on skip or failure so callers can distinguish.
    Writes to a .tmp file first; only replaces original if ffmpeg exits cleanly.
    """
    if video_path.suffix.lower() not in TRANSCODE_EXTENSIONS:
        return None, "wrong_ext"

    if is_hevc(video_path, ffmpeg):
        return None, "already_hevc"

    original_size = video_path.stat().st_size
    tmp_path = video_path.with_suffix(".tmp.mp4")

    try:
        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(video_path),
                "-c:v", encoder,
                *encoder_args,
                "-c:a", "copy",          # audio passthrough, no re-encode
                "-tag:v", "hvc1",        # broad player compatibility (Apple, etc.)
                "-movflags", "+faststart",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1-hour ceiling per file
        )
        if result.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            stderr_tail = result.stderr[-400:] if result.stderr else "(no stderr)"
            print(f"\n    âš ï¸  ffmpeg error on {video_path.name}:\n{stderr_tail}")
            _write_trace(
                "transcode_ffmpeg_error",
                file=str(video_path),
                returncode=result.returncode,
                stderr_tail=stderr_tail,
            )
            return None, "ffmpeg_error"

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            _write_trace("transcode_zero_output", file=str(video_path))
            return None, "zero_output"

        new_size = tmp_path.stat().st_size

        # Replace original only after verified success.
        # Guard: if unlink succeeds but rename fails, log and abort cleanly.
        video_path.unlink()
        try:
            tmp_path.rename(video_path)
        except Exception as rename_exc:
            _write_trace(
                "transcode_rename_error",
                file=str(video_path),
                tmp=str(tmp_path),
                error=str(rename_exc),
            )
            print(f"\n    âš ï¸  Rename failed for {video_path.name}: {rename_exc}")
            return None, "rename_error"

        return original_size, new_size

    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        _write_trace("transcode_timeout", file=str(video_path))
        print(f"\n    âš ï¸  Timeout transcoding {video_path.name} â€” skipped")
        return None, "timeout"
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        _write_trace("transcode_exception", file=str(video_path), error=str(exc), error_type=type(exc).__name__)
        print(f"\n    âš ï¸  Failed transcoding {video_path.name}: {exc}")
        return None, "exception"


# â”€â”€ Course operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def transcode_course(
    course_dir: Path,
    ffmpeg: str,
    encoder: str,
    encoder_args: list[str],
    crf: int,
) -> tuple[int, int]:
    """
    Transcode all eligible videos in a course folder to H.265.
    Returns (total_original_bytes_of_videos, total_new_bytes_of_videos).
    """
    # Exclude leftover .tmp.mp4 files from previous interrupted runs.
    # These are created by transcode_video() as a staging area; if a prior run
    # crashed mid-transcode, the temp file may still linger on disk.
    videos = [
        f for f in course_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in TRANSCODE_EXTENSIONS
        and not f.name.endswith(".tmp.mp4")
    ]
    if not videos:
        return 0, 0

    total_before = 0
    total_after = 0
    log(f"  Transcoding {len(videos)} video(s) in '{course_dir.name}' (H.265 CRF {crf})")

    for i, vf in enumerate(videos, start=1):
        # Snapshot the size BEFORE the transcode attempt.
        # If the file disappears during transcoding we still have a consistent
        # byte count to add to total_before.
        try:
            size_before = vf.stat().st_size
        except FileNotFoundError:
            _write_trace(
                "transcode_file_missing_before",
                file=str(vf),
                index=i,
                error="File disappeared before transcode started",
            )
            print(f"\n    âš ï¸  File missing (before transcode): {vf.name}")
            continue

        print(f"\r    [{i}/{len(videos)}] {vf.name[:60]:<60}", end="", flush=True)

        try:
            result = transcode_video(vf, ffmpeg, encoder, encoder_args)
        except Exception as exc:
            # Defensive: transcode_video() should never raise, but guard anyway.
            _write_trace(
                "transcode_unexpected_error",
                file=str(vf),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            print(f"\n    âš ï¸  Unexpected error on {vf.name}: {exc}")
            total_before += size_before
            total_after += size_before  # treat as unchanged
            continue

        first, second = result  # always a 2-tuple now

        if first is not None:
            # Success: (original_bytes, new_bytes)
            orig, new = first, second
            saved = orig - new
            pct = (saved / orig * 100) if orig else 0
            total_before += orig
            total_after += new
            _write_trace(
                "transcode_video_done",
                file=str(vf),
                original_bytes=orig,
                new_bytes=new,
                saved_bytes=saved,
            )
            print(
                f"\r    âœ… {vf.name[:50]:<50}"
                f"  {human_size(orig)} â†’ {human_size(new)}  ({pct:.1f}% smaller)"
            )
        else:
            # Skip or failure: second is the reason string.
            reason: str = second  # type: ignore[assignment]
            total_before += size_before

            # Guard: the file may no longer exist if rename failed after unlink.
            if vf.exists():
                try:
                    current_size = vf.stat().st_size
                except OSError as stat_exc:
                    _write_trace(
                        "transcode_stat_error",
                        file=str(vf),
                        reason=reason,
                        error=str(stat_exc),
                    )
                    current_size = size_before  # fall back to pre-transcode size
            else:
                # File is truly gone (rename_error case). Log it explicitly.
                _write_trace(
                    "transcode_file_missing_after",
                    file=str(vf),
                    reason=reason,
                    error="FileNotFoundError: file vanished after transcode attempt",
                )
                print(f"\n    âŒ  File missing after transcode: {vf.name} (reason: {reason})")
                current_size = 0  # unknown; use 0 to flag discrepancy

            total_after += current_size

            # Use distinct labels so logs differentiate intentional skips from errors.
            label = "â­ " if reason in ("already_hevc", "wrong_ext") else "âš ï¸ "
            print(f"\r    {label} {vf.name[:60]:<60} ({reason})")

    return total_before, total_after


def compress_course(course_dir: Path, out_dir: Path, zip_name: str | None = None) -> Path:
    """Zip a single course folder and return the path of the created zip.

    zip_name: override the name used for the zip file and the root folder inside
    the archive (useful when course_dir has a temporary tag in its name).
    """
    name = zip_name or course_dir.name
    zip_path = out_dir / f"{name}.zip"
    all_files = [
        f for f in course_dir.rglob("*")
        if f.is_file() and not f.name.endswith(".tmp.mp4")
    ]
    total = len(all_files)
    original_size = sum(f.stat().st_size for f in all_files)

    log(f"  '{name}'  ({total} files, {human_size(original_size)} uncompressed)")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for i, file_path in enumerate(all_files, start=1):
            method, level = pick_compression(file_path)
            # Use clean name as root folder inside the archive (strip any tags).
            arcname = Path(name) / file_path.relative_to(course_dir)
            if method == zipfile.ZIP_DEFLATED:
                zf.write(file_path, arcname, compress_type=method, compresslevel=level)
            else:
                zf.write(file_path, arcname, compress_type=method)
            if i % 20 == 0 or i == total:
                print(f"\r    {i}/{total} files...", end="", flush=True)

    compressed_size = zip_path.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100 if original_size else 0
    print(
        f"\r    âœ… {human_size(original_size)} â†’ {human_size(compressed_size)}"
        f"  ({ratio:.1f}% smaller)  â†’  {zip_path.name}"
    )
    return zip_path


def list_courses(base_dir: Path) -> list[Path]:
    """Return course folders that are pending or interrupted (exclude archived and system dirs)."""
    return sorted(
        d for d in base_dir.iterdir()
        if d.is_dir()
        and d.name != "archives"
        and ARCHIVED_TAG not in d.name
    )


def list_all_courses(base_dir: Path) -> list[Path]:
    """Return all course folders including archived ones (used by --list)."""
    return sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name != "archives"
    )


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    if not BASE_DIR.exists():
        print(f"âŒ  Base directory not found: {BASE_DIR}")
        sys.exit(1)

    # Create archives folder for compressed files
    archives_dir = BASE_DIR / "archives"
    archives_dir.mkdir(exist_ok=True)

    args = sys.argv[1:]
    do_transcode = "--transcode" in args
    args = [a for a in args if a != "--transcode"]

    # Parse optional --crf N
    crf = DEFAULT_CRF
    if "--crf" in args:
        idx = args.index("--crf")
        try:
            crf = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        except (IndexError, ValueError):
            print("âŒ  --crf requires an integer value (e.g. --crf 28)")
            sys.exit(1)

    # Parse optional --batch N
    batch_size: int | None = None
    if "--batch" in args:
        idx = args.index("--batch")
        try:
            batch_size = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        except (IndexError, ValueError):
            print("âŒ  --batch requires an integer value (e.g. --batch 5)")
            sys.exit(1)

    # --list: show all courses with status then exit
    if "--list" in args:
        all_dirs = list_all_courses(BASE_DIR)
        if not all_dirs:
            print("No course folders found.")
            return
        archived = [d for d in all_dirs if ARCHIVED_TAG in d.name]
        pending = [d for d in all_dirs if ARCHIVED_TAG not in d.name and COMPRESSING_TAG not in d.name]
        interrupted = [d for d in all_dirs if COMPRESSING_TAG in d.name]
        print(f"Courses in {BASE_DIR}:  {len(pending)} pending  Â·  {len(archived)} archived  Â·  {len(interrupted)} interrupted\n")
        for c in all_dirs:
            files = [f for f in c.rglob("*") if f.is_file()]
            total_size = sum(f.stat().st_size for f in files)
            if ARCHIVED_TAG in c.name:
                status = "âœ…"
            elif COMPRESSING_TAG in c.name:
                status = "â³"
            else:
                status = "ðŸ“"
            print(f"  {status} {c.name}  ({len(files)} files, {human_size(total_size)})")
        return

    courses = list_courses(BASE_DIR)

    if not courses:
        print("No pending courses found. All are archived or none exist.")
        return

    # Partial name filter
    if args:
        query = " ".join(args).lower()
        courses = [c for c in courses if query in c.name.lower()]
        if not courses:
            print(f"âŒ  No course matching '{query}' found.")
            sys.exit(1)

    # Apply batch limit
    if batch_size is not None:
        courses = courses[:batch_size]
        print(f"  Batch mode: processing {len(courses)} course(s) this run.\n")

    # â”€â”€ Transcode phase â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    transcode_stats: list[tuple[str, int, int]] = []  # (name, before, after)

    if do_transcode:
        ffmpeg = get_ffmpeg()
        log(f"FFmpeg: {ffmpeg}", "transcode_ffmpeg")
        encoder, encoder_args, enc_label = detect_encoder(ffmpeg, crf)
        log(f"Transcoding {len(courses)} course(s) to H.265 via {enc_label} (CRF {crf})\n", "transcode_batch_start")

        for course_dir in courses:
            before, after = transcode_course(course_dir, ffmpeg, encoder, encoder_args, crf)
            transcode_stats.append((course_dir.name, before, after))
            _write_trace("transcode_course_done", course=course_dir.name, before_bytes=before, after_bytes=after)

        print()

    # â”€â”€ Zip phase â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log(f"Starting compression of {len(courses)} course(s)", "compress_batch_start")
    print("    Strategy: LZMA for text/HTML/VTT  Â·  STORE for video/audio  Â·  DEFLATE-9 for others\n")

    created: list[Path] = []
    total_original = 0
    total_compressed = 0

    for course_dir in courses:
        # If folder already has COMPRESSING_TAG it was interrupted last run â€” resume it.
        if COMPRESSING_TAG in course_dir.name:
            original_name = course_dir.name.replace(COMPRESSING_TAG, "")
            active_dir = course_dir
            log(f"  â†©  '{original_name}'  (resuming interrupted compression)", "compress_course_resume")
        else:
            original_name = course_dir.name
            active_dir = BASE_DIR / f"{COMPRESSING_TAG}{original_name}"
            try:
                course_dir.rename(active_dir)
            except Exception as exc:
                log(f"  âš ï¸  Cannot rename '{original_name}': {exc}", "compress_rename_error")
                continue

        try:
            original = sum(f.stat().st_size for f in active_dir.rglob("*") if f.is_file())
            zip_path = compress_course(active_dir, archives_dir, zip_name=original_name)
            compressed = zip_path.stat().st_size
            total_original += original
            total_compressed += compressed
            created.append(zip_path)
            _write_trace("compress_course_done", course=original_name, original_bytes=original, compressed_bytes=compressed)

            # Mark folder as archived
            archived_dir = BASE_DIR / f"{ARCHIVED_TAG}{original_name}"
            active_dir.rename(archived_dir)

        except Exception as exc:
            # Restore original folder name so it can be retried next run
            try:
                active_dir.rename(BASE_DIR / original_name)
            except Exception:
                pass
            log(f"  âš ï¸  Failed: {original_name} â€” {exc}", "compress_course_error")

    overall_ratio = (1 - total_compressed / total_original) * 100 if total_original else 0
    print()
    log(f"âœ…  Done! {len(created)} zip file(s) created.", "compress_batch_done")
    log(
        f"   Total: {human_size(total_original)} â†’ {human_size(total_compressed)}"
        f"  ({overall_ratio:.1f}% overall reduction)",
        "compress_batch_summary",
    )

    # â”€â”€ Transcode comparison table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if transcode_stats:
        print()
        log("Video transcoding summary:", "transcode_summary")
        print(f"\n  {'Course':<45} {'Before':>10} {'After':>10} {'Saved':>10} {'Reduction':>10}")
        print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        grand_before = grand_after = 0
        for name, before, after in transcode_stats:
            saved = before - after
            pct = (saved / before * 100) if before else 0
            grand_before += before
            grand_after += after
            print(
                f"  {name[:45]:<45} {human_size(before):>10} {human_size(after):>10}"
                f" {human_size(saved):>10} {pct:>9.1f}%"
            )
        grand_saved = grand_before - grand_after
        grand_pct = (grand_saved / grand_before * 100) if grand_before else 0
        print(f"  {'â”€'*45} {'â”€'*10} {'â”€'*10} {'â”€'*10} {'â”€'*10}")
        print(
            f"  {'TOTAL':<45} {human_size(grand_before):>10} {human_size(grand_after):>10}"
            f" {human_size(grand_saved):>10} {grand_pct:>9.1f}%"
        )


if __name__ == "__main__":
    main()

