"""Offline benchmark script.

Usage:
    python sentinel/eval.py --real-dir data/real --ai-dir data/ai
    python sentinel/eval.py --real-dir tests/fixtures/real.jpg \\
                            --ai-dir   tests/fixtures/synthetic.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabulate import tabulate

from sentinel.detector import Detector

_IMAGE_GLOB = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def _collect_paths(arg: str) -> list[Path]:
    p = Path(arg)
    if p.is_file():
        return [p]
    paths: list[Path] = []
    for pattern in _IMAGE_GLOB:
        paths.extend(p.glob(pattern))
    return sorted(paths)


def evaluate(
    real_paths: list[Path],
    ai_paths: list[Path],
    threshold: float = 0.6,
) -> dict:
    """Run Detector on every path; return accuracy/precision/recall/f1 + per-image rows.

    When both lists are empty accuracy is returned as 0.0.

    Returns
    -------
    dict with keys:
        accuracy   – float
        precision  – float
        recall     – float
        f1         – float
        rows       – list[dict] (target, true_label, pred_label, score)
    """
    detector = Detector()
    rows: list[dict] = []

    for path in real_paths:
        result = detector.detect(path)
        pred = result["label"]
        rows.append(
            {
                "target": str(path),
                "true_label": "REAL",
                "pred_label": pred,
                "score": result["score"],
            }
        )

    for path in ai_paths:
        result = detector.detect(path)
        pred = result["label"]
        rows.append(
            {
                "target": str(path),
                "true_label": "AI",
                "pred_label": pred,
                "score": result["score"],
            }
        )

    if not rows:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "rows": rows}

    tp = sum(1 for r in rows if r["true_label"] == "AI" and r["pred_label"] == "AI")
    fp = sum(1 for r in rows if r["true_label"] == "REAL" and r["pred_label"] == "AI")
    fn = sum(1 for r in rows if r["true_label"] == "AI" and r["pred_label"] != "AI")
    correct = sum(1 for r in rows if r["true_label"] == r["pred_label"])

    accuracy = correct / len(rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Synthetic Sentinel on labelled images.")
    parser.add_argument("--real-dir", required=True, help="File or directory of real images.")
    parser.add_argument("--ai-dir", required=True, help="File or directory of AI-generated images.")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Score threshold for AI label (default 0.6)."
    )
    args = parser.parse_args()

    real_paths = _collect_paths(args.real_dir)
    ai_paths = _collect_paths(args.ai_dir)

    print(f"Loaded {len(real_paths)} real image(s), {len(ai_paths)} AI image(s).\n")

    metrics = evaluate(real_paths, ai_paths, threshold=args.threshold)

    # Per-image table
    table = [
        [r["target"], r["true_label"], r["pred_label"], f"{r['score']:.4f}"]
        for r in metrics["rows"]
    ]
    headers = ["target", "true_label", "pred_label", "score"]
    print(tabulate(table, headers=headers, tablefmt="simple"))
    print()

    # Aggregate metrics
    summary = [
        ["accuracy", f"{metrics['accuracy']:.4f}"],
        ["precision", f"{metrics['precision']:.4f}"],
        ["recall", f"{metrics['recall']:.4f}"],
        ["f1", f"{metrics['f1']:.4f}"],
    ]
    print(tabulate(summary, headers=["metric", "value"], tablefmt="simple"))


if __name__ == "__main__":
    main()
