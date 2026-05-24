"""Streamlit Review UI for BoundingBoxer annotations."""
import argparse
import sys
from pathlib import Path

import cv2
import streamlit as st

from ..config import CLASS_NAMES
from .logic import (
    bbox_pixels_to_yolo,
    bbox_yolo_to_pixels,
    build_image_path,
    filter_results,
    load_report,
    save_report,
)


def _parse_args():
    """Extract CLI arguments passed after the '--' separator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--port", default=8501, type=int)
    sep = sys.argv.index("--") if "--" in sys.argv else -1
    cli_args = sys.argv[sep + 1:] if sep >= 0 else []
    try:
        return parser.parse_args(cli_args)
    except SystemExit:
        return argparse.Namespace(input=None, port=8501)


def _on_bbox_change():
    """Mark the bounding box as modified by the user."""
    st.session_state.bbox_modified = True


def _reset_entry_state(entry, img_w, img_h):
    """Reset session state bbox and class override from the current entry."""
    pixels = bbox_yolo_to_pixels(entry["bbox"], img_w, img_h)
    if pixels:
        st.session_state.bbox_x = pixels["x"]
        st.session_state.bbox_y = pixels["y"]
        st.session_state.bbox_w = pixels["width"]
        st.session_state.bbox_h = pixels["height"]
    else:
        st.session_state.bbox_x = 0
        st.session_state.bbox_y = 0
        st.session_state.bbox_w = 0
        st.session_state.bbox_h = 0
    st.session_state.bbox_modified = False
    detected = entry["detected_class"]
    st.session_state.class_override = (
        detected if detected in CLASS_NAMES else CLASS_NAMES[0]
    )


def main():
    """Run the Streamlit review UI (only called via ``streamlit run``)."""
    args = _parse_args()
    input_dir = Path(args.input) if args.input else None

    # -----------------------------------------------------------------------
    # Session state initialisation
    # -----------------------------------------------------------------------
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "bbox_modified" not in st.session_state:
        st.session_state.bbox_modified = False
    if "_prev_idx" not in st.session_state:
        st.session_state._prev_idx = -1

    # -----------------------------------------------------------------------
    # Load report
    # -----------------------------------------------------------------------
    if input_dir is None or not input_dir.exists():
        st.error("No input directory specified. Run with: "
                 "streamlit run app.py -- --input /path/to/output")
        st.stop()

    try:
        report = load_report(input_dir)
    except Exception:
        st.error(f"Could not load report.json from {input_dir}")
        st.stop()

    results = report.get("results", [])

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    st.sidebar.title("BoundingBoxer Review")

    min_conf = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0)
    only_needs_review = st.sidebar.checkbox("Only needs review")
    only_unreviewed = st.sidebar.checkbox("Only unreviewed")
    class_filter = st.sidebar.selectbox("Class", ["all"] + CLASS_NAMES)

    filtered_results = filter_results(
        results,
        min_confidence=min_conf,
        only_needs_review=only_needs_review,
        only_unreviewed=only_unreviewed,
        class_filter=class_filter,
    )

    # Progress display
    reviewed_count = sum(1 for r in filtered_results if r["reviewed"])
    total_count = len(filtered_results)

    st.sidebar.markdown("---")
    if total_count > 0:
        st.sidebar.write(
            f"Reviewed {reviewed_count}/{total_count} "
            f"({reviewed_count / total_count * 100:.0f}%)"
        )
        st.sidebar.progress(reviewed_count / total_count)
    else:
        st.sidebar.write("Reviewed 0/0 (0%)")

    st.sidebar.markdown("---")
    if st.sidebar.button("Export Reviewed"):
        st.info("Export not yet implemented.")

    # -----------------------------------------------------------------------
    # Main panel
    # -----------------------------------------------------------------------
    if not filtered_results:
        st.info("No images match the current filters.")
        st.stop()

    # Clamp current_idx to valid range
    max_idx = len(filtered_results) - 1
    if st.session_state.current_idx > max_idx:
        st.session_state.current_idx = max_idx
    if st.session_state.current_idx < 0:
        st.session_state.current_idx = 0

    current_idx = st.session_state.current_idx
    entry = filtered_results[current_idx]

    st.header(f"Image {current_idx + 1}/{len(filtered_results)}")

    # ---- Load and display image ----
    image_full_path = build_image_path(input_dir, entry["image"])
    bgr_image = cv2.imread(str(image_full_path))
    if bgr_image is None:
        st.error(f"Cannot load image: {image_full_path}")
        st.stop()

    img_h, img_w = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Draw bounding box overlay on the image
    if entry["bbox"] is not None:
        pixels = bbox_yolo_to_pixels(entry["bbox"], img_w, img_h)
        if pixels:
            color = (0, 255, 0) if entry["reviewed"] else (255, 0, 0)
            cv2.rectangle(
                rgb_image,
                (pixels["x"], pixels["y"]),
                (pixels["x"] + pixels["width"],
                 pixels["y"] + pixels["height"]),
                color,
                2,
            )

    st.image(rgb_image, channels="RGB")

    # ---- Info display ----
    st.write(
        f"Detected: {entry['detected_class']}  |  "
        f"Expected: {entry['expected_class']}"
    )
    st.write(
        f"Confidence: {entry['combined_confidence']:.2f}  |  "
        f"MediaPipe: {entry['mediapipe_confidence']:.2f}"
    )

    # ---- BBox editor ----
    # Reset editor state when switching to a different entry
    if st.session_state._prev_idx != current_idx:
        _reset_entry_state(entry, img_w, img_h)
        st.session_state._prev_idx = current_idx

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("X", key="bbox_x", min_value=0, step=1,
                        on_change=_on_bbox_change)
        st.number_input("Width", key="bbox_w", min_value=0, step=1,
                        on_change=_on_bbox_change)
    with col2:
        st.number_input("Y", key="bbox_y", min_value=0, step=1,
                        on_change=_on_bbox_change)
        st.number_input("Height", key="bbox_h", min_value=0, step=1,
                        on_change=_on_bbox_change)

    # ---- Class override ----
    override_class = st.selectbox(
        "Override class", CLASS_NAMES,
        key="class_override",
        on_change=_on_bbox_change,
    )

    # ---- Action buttons ----
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("\u25c0 Previous"):
            st.session_state.current_idx = max(0, current_idx - 1)
            st.rerun()
    with col_b:
        if st.button("Reset bbox"):
            _init_bbox_from_entry(entry, img_w, img_h)
            _reset_override_class(entry)
            st.rerun()
    with col_c:
        if st.button("Approve & Next"):
            original_class = entry["detected_class"]
            entry["bbox"] = bbox_pixels_to_yolo(
                st.session_state.bbox_x, st.session_state.bbox_y,
                st.session_state.bbox_w, st.session_state.bbox_h,
                img_w, img_h,
            )
            entry["detected_class"] = override_class
            entry["reviewed"] = True
            entry["manual_override"] = (
                st.session_state.bbox_modified
                or override_class != original_class
            )
            save_report(report, input_dir)
            st.session_state.current_idx = min(current_idx + 1, max_idx)
            st.rerun()


if __name__ == "__main__":
    main()
