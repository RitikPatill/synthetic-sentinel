"""Command-line interface for Synthetic Sentinel.

Usage:
    python -m sentinel.cli <path|url> [<path|url> ...]
    sentinel <path|url> [<path|url> ...]

Exits with code 1 if any image is labelled AI, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _resolve_to_path(target: str) -> tuple[Path, bool]:
    """Return (path, is_temp). Downloads URL to a NamedTemporaryFile if needed."""
    if target.startswith(("http://", "https://")):
        import httpx

        with httpx.Client(follow_redirects=True, timeout=30) as client:
            response = client.get(target)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        suffix = ".png" if "png" in content_type else ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.close()
        return Path(tmp.name), True
    return Path(target), False


def _print_table(rows: list[dict]) -> None:
    """Print results as an aligned table to stdout."""
    try:
        from tabulate import tabulate

        print(tabulate(rows, headers="keys", tablefmt="simple"))
    except ImportError:
        header = f"{'target':<40} {'label':<10} {'score':<8}  reasoning"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{str(row['target']):<40} {row['label']:<10}"
                f" {row['score']:<8.4f}  {row['reasoning']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Detect whether images are real or AI-generated.",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="path|url",
        help="File path(s) or URL(s) to analyse.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Score threshold above which a result counts as AI for the exit code (default: 0.6).",
    )
    args = parser.parse_args()

    from sentinel.detector import Detector

    detector = Detector()
    rows: list[dict] = []
    any_ai = False

    for target in args.targets:
        tmp_path = None
        is_temp = False
        try:
            path, is_temp = _resolve_to_path(target)
            tmp_path = path if is_temp else None
            result = detector.detect(path)
            if result["label"] == "AI" or result["score"] >= args.threshold:
                any_ai = True
            rows.append(
                {
                    "target": target,
                    "label": result["label"],
                    "score": result["score"],
                    "reasoning": result["reasoning"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "target": target,
                    "label": "ERROR",
                    "score": 0.0,
                    "reasoning": str(exc),
                }
            )
        finally:
            if is_temp and tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    _print_table(rows)
    sys.exit(1 if any_ai else 0)


if __name__ == "__main__":
    main()
