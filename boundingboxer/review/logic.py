"""Pure functions for the BoundingBoxer Review UI — no Streamlit imports."""
from pathlib import Path

from ..config import OUTPUT_REPORT_JSON
from ..reporter import Reporter


def load_report(input_dir):
    """Load report.json from input_dir. Returns dict."""
    input_dir = Path(input_dir)
    return Reporter().load(input_dir / OUTPUT_REPORT_JSON)


def save_report(report, input_dir):
    """Save report dict to input_dir/report.json."""
    input_dir = Path(input_dir)
    Reporter().save(report, input_dir / OUTPUT_REPORT_JSON)


def filter_results(results, min_confidence=0.0, only_needs_review=False,
                   only_unreviewed=False, class_filter=None):
    """Filter report entries by various criteria.

    - min_confidence: show entries with combined_confidence >= min_confidence
    - only_needs_review: show only needs_review=True entries
    - only_unreviewed: show only reviewed=False entries
    - class_filter: show only this expected_class (None or "all" = no filter)
    """
    filtered = []
    for entry in results:
        if entry["combined_confidence"] < min_confidence:
            continue
        if only_needs_review and not entry["needs_review"]:
            continue
        if only_unreviewed and entry["reviewed"]:
            continue
        if class_filter is not None and class_filter != "all" and entry["expected_class"] != class_filter:
            continue
        filtered.append(entry)
    return filtered


def bbox_yolo_to_pixels(bbox_yolo, img_w, img_h):
    """Convert YOLO [cx, cy, w, h] normalized → pixels.

    Returns {"x": int, "y": int, "width": int, "height": int}.
    If bbox_yolo is None, return None.
    """
    if bbox_yolo is None:
        return None
    cx, cy, w, h = bbox_yolo
    x = round((cx - w / 2) * img_w)
    y = round((cy - h / 2) * img_h)
    width = round(w * img_w)
    height = round(h * img_h)
    return {"x": x, "y": y, "width": width, "height": height}


def bbox_pixels_to_yolo(x, y, width, height, img_w, img_h):
    """Convert pixel coords → YOLO normalized [cx, cy, w, h]."""
    cx = (x + width / 2) / img_w
    cy = (y + height / 2) / img_h
    w = width / img_w
    h = height / img_h
    return [cx, cy, w, h]


def build_summary_table(report):
    """Build per-class summary rows plus an aggregate "ALL" row from a report dict.

    Returns (list[dict], dict): per-class rows and an aggregate "ALL" row.
    """
    summary = Reporter().get_summary(report)
    rows = []
    for class_name, stats in sorted(summary.per_class_stats.items()):
        rows.append({
            "Class": class_name,
            "Total": stats.total,
            "Detected": stats.detected,
            "Not detected": stats.not_detected,
            "Avg confidence": f"{stats.average_confidence:.2f}",
        })
    row_all = {
        "Class": "ALL",
        "Total": summary.total_images,
        "Detected": summary.total_detected,
        "Not detected": summary.total_not_detected,
        "Avg confidence": f"{summary.average_confidence:.2f}",
    }
    return rows, row_all


def build_image_path(input_dir, image_relative):
    """Build full image path: input_dir/images/<image_relative>."""
    input_dir = Path(input_dir)
    return input_dir / "images" / image_relative
