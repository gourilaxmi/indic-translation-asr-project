import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _timestamped_filename(prefix: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def save_transcript(text: str, output_dir: str) -> str:
    """
    Save a transcription string to a timestamped .txt file.

    Returns the full path of the saved file.
    """
    _ensure_dir(output_dir)
    filename = _timestamped_filename("transcript", "txt")
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Transcript saved: {filepath}")
    return filepath


def save_transliteration(text: str, output_dir: str) -> str:
    """
    Save a transliteration string to a timestamped .txt file.

    Returns the full path of the saved file.
    """
    _ensure_dir(output_dir)
    filename = _timestamped_filename("transliteration", "txt")
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Transliteration saved: {filepath}")
    return filepath


def read_text_file(filepath: str) -> str:
    """Read and return the contents of a UTF-8 text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def list_output_files(directory: str, ext: str = "txt") -> list:
    """List all files with the given extension in a directory, sorted newest-first."""
    if not os.path.isdir(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(f".{ext}")
    ]
    return sorted(files, reverse=True)


def format_duration(seconds: float) -> str:
    """Human-readable duration string, e.g. '1m 23s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"
