"""Streamlit UI for BoundingBoxer — Process and Review modes."""
import argparse
import sys
from pathlib import Path

# Allow running with `streamlit run app.py` directly (not just as a module)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from boundingboxer.config import CLASS_NAMES
from boundingboxer.main import run_pipeline
from boundingboxer.review.logic import (
    bbox_pixels_to_yolo,
    bbox_yolo_to_pixels,
    build_image_path,
    build_summary_table,
    filter_results,
    load_report,
    save_report,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--port", default=8501, type=int)
    sep = sys.argv.index("--") if "--" in sys.argv else -1
    cli_args = sys.argv[sep + 1:] if sep >= 0 else []
    try:
        return parser.parse_args(cli_args)
    except SystemExit:
        return argparse.Namespace(input=None, port=8501)


def _on_bbox_widget_change():
    st.session_state.app_bbox_x = st.session_state.bbox_x_widget
    st.session_state.app_bbox_y = st.session_state.bbox_y_widget
    st.session_state.app_bbox_w = st.session_state.bbox_w_widget
    st.session_state.app_bbox_h = st.session_state.bbox_h_widget
    st.session_state.app_class_override = st.session_state.class_override_widget
    st.session_state.bbox_modified = True


def _init_bbox_app_state(entry, img_w, img_h):
    pixels = bbox_yolo_to_pixels(entry["bbox"], img_w, img_h)
    if pixels:
        st.session_state.app_bbox_x = pixels["x"]
        st.session_state.app_bbox_y = pixels["y"]
        st.session_state.app_bbox_w = pixels["width"]
        st.session_state.app_bbox_h = pixels["height"]
    else:
        st.session_state.app_bbox_x = 0
        st.session_state.app_bbox_y = 0
        st.session_state.app_bbox_w = 0
        st.session_state.app_bbox_h = 0
    st.session_state.bbox_modified = False
    detected = entry["detected_class"]
    st.session_state.app_class_override = (
        detected if detected in CLASS_NAMES else CLASS_NAMES[0]
    )


def _render_process_mode():
    st.header("Pipeline")

    input_dir = st.sidebar.text_input("Input", value="./data")
    output_dir = st.sidebar.text_input("Output", value="./output")
    export_format = st.sidebar.selectbox("Format", ["yolo", "coco"])
    confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.8)
    detection_conf = st.sidebar.slider("Detection confidence", 0.1, 1.0, 0.5)

    if st.sidebar.button("Run Pipeline"):
        st.session_state._run_pipeline = True
        st.session_state._pipe_input = input_dir
        st.session_state._pipe_output = output_dir
        st.session_state._pipe_format = export_format
        st.session_state._pipe_confidence = confidence
        st.session_state._pipe_detection_conf = detection_conf

    if st.session_state.get("_run_pipeline"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Processing {current}/{total} images...")

        try:
            report = run_pipeline(
                input_dir=st.session_state._pipe_input,
                output_dir=st.session_state._pipe_output,
                format=st.session_state._pipe_format,
                confidence_threshold=st.session_state._pipe_confidence,
                detection_confidence=st.session_state._pipe_detection_conf,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.session_state._run_pipeline = False
            return

        st.session_state.process_report = report
        st.session_state.process_output = st.session_state._pipe_output
        st.session_state._run_pipeline = False
        st.rerun()

    if st.session_state.get("process_report"):
        st.success("Processing complete!")

        rows, row_all = build_summary_table(st.session_state.process_report)
        if rows:
            st.table(rows + [row_all])
        else:
            st.info("No images processed.")

        if st.button("Open in Review"):
            st.session_state.app_mode = "review"
            st.session_state.review_input = st.session_state.process_output
            st.rerun()


def _render_review_mode():
    input_dir = None
    if st.session_state.get("review_input"):
        input_dir = Path(st.session_state.review_input)

    if input_dir is None or not input_dir.exists():
        st.error("No input directory. Run pipeline first or specify --input.")
        st.stop()

    try:
        report = load_report(input_dir)
    except Exception:
        st.error(f"Could not load report.json from {input_dir}")
        st.stop()

    results = report.get("results", [])

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

    if not filtered_results:
        st.info("No images match the current filters.")
        st.stop()

    max_idx = len(filtered_results) - 1
    if st.session_state.current_idx > max_idx:
        st.session_state.current_idx = max_idx
    if st.session_state.current_idx < 0:
        st.session_state.current_idx = 0

    current_idx = st.session_state.current_idx
    entry = filtered_results[current_idx]

    st.header(f"Image {current_idx + 1}/{len(filtered_results)}")

    image_full_path = build_image_path(input_dir, entry["image"])
    bgr_image = cv2.imread(str(image_full_path))
    if bgr_image is None:
        st.error(f"Cannot load image: {image_full_path}")
        st.stop()

    img_h, img_w = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    color = "#00ff00" if entry["reviewed"] else "#ff0000"
    initial_drawing = None
    if (st.session_state.app_bbox_w > 0 and st.session_state.app_bbox_h > 0):
        initial_drawing = {
            "version": "4.4.0",
            "objects": [{
                "type": "rect",
                "left": float(st.session_state.app_bbox_x),
                "top": float(st.session_state.app_bbox_y),
                "width": float(st.session_state.app_bbox_w),
                "height": float(st.session_state.app_bbox_h),
                "stroke": color,
                "strokeWidth": 2,
                "fill": "rgba(0,0,0,0)",
            }],
        }

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=2,
        stroke_color="#00ff00",
        background_image=rgb_image,
        drawing_mode="rect",
        initial_drawing=initial_drawing,
        height=img_h,
        width=img_w,
        key=f"canvas_{current_idx}",
    )

    if (canvas_result.json_data is not None
            and canvas_result.json_data.get("objects")):
        obj = canvas_result.json_data["objects"][-1]
        new_x = max(0, int(float(obj["left"])))
        new_y = max(0, int(float(obj["top"])))
        new_w = max(0, int(float(obj["width"])))
        new_h = max(0, int(float(obj["height"])))
        if (new_x, new_y, new_w, new_h) != (
            st.session_state.app_bbox_x,
            st.session_state.app_bbox_y,
            st.session_state.app_bbox_w,
            st.session_state.app_bbox_h,
        ):
            st.session_state.app_bbox_x = new_x
            st.session_state.app_bbox_y = new_y
            st.session_state.app_bbox_w = new_w
            st.session_state.app_bbox_h = new_h
            st.session_state.bbox_modified = True
            st.rerun()

    st.write(
        f"Detected: {entry['detected_class']}  |  "
        f"Expected: {entry['expected_class']}"
    )
    st.write(
        f"Confidence: {entry['combined_confidence']:.2f}  |  "
        f"MediaPipe: {entry['mediapipe_confidence']:.2f}"
    )

    if st.session_state._prev_idx != current_idx:
        _init_bbox_app_state(entry, img_w, img_h)
        st.session_state._prev_idx = current_idx

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("X",
                        value=st.session_state.app_bbox_x,
                        key="bbox_x_widget", min_value=0, step=1,
                        on_change=_on_bbox_widget_change)
        st.number_input("Width",
                        value=st.session_state.app_bbox_w,
                        key="bbox_w_widget", min_value=0, step=1,
                        on_change=_on_bbox_widget_change)
    with col2:
        st.number_input("Y",
                        value=st.session_state.app_bbox_y,
                        key="bbox_y_widget", min_value=0, step=1,
                        on_change=_on_bbox_widget_change)
        st.number_input("Height",
                        value=st.session_state.app_bbox_h,
                        key="bbox_h_widget", min_value=0, step=1,
                        on_change=_on_bbox_widget_change)

    st.selectbox(
        "Override class", CLASS_NAMES,
        index=CLASS_NAMES.index(st.session_state.app_class_override)
        if st.session_state.app_class_override in CLASS_NAMES else 0,
        key="class_override_widget",
        on_change=_on_bbox_widget_change,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("\u25c0 Previous"):
            st.session_state.current_idx = max(0, current_idx - 1)
            st.rerun()
    with col_b:
        if st.button("Reset bbox"):
            _init_bbox_app_state(entry, img_w, img_h)
            st.rerun()
    with col_c:
        if st.button("Approve & Next"):
            original_class = entry["detected_class"]
            entry["bbox"] = bbox_pixels_to_yolo(
                st.session_state.app_bbox_x, st.session_state.app_bbox_y,
                st.session_state.app_bbox_w, st.session_state.app_bbox_h,
                img_w, img_h,
            )
            entry["detected_class"] = st.session_state.app_class_override
            entry["reviewed"] = True
            entry["manual_override"] = (
                st.session_state.bbox_modified
                or st.session_state.app_class_override != original_class
            )
            save_report(report, input_dir)
            st.session_state.current_idx = min(current_idx + 1, max_idx)
            st.rerun()


def main():
    args = _parse_args()

    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "process"
    if "process_report" not in st.session_state:
        st.session_state.process_report = None
    if "review_input" not in st.session_state:
        st.session_state.review_input = None
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "bbox_modified" not in st.session_state:
        st.session_state.bbox_modified = False
    if "app_bbox_x" not in st.session_state:
        st.session_state.app_bbox_x = 0
    if "app_bbox_y" not in st.session_state:
        st.session_state.app_bbox_y = 0
    if "app_bbox_w" not in st.session_state:
        st.session_state.app_bbox_w = 0
    if "app_bbox_h" not in st.session_state:
        st.session_state.app_bbox_h = 0
    if "app_class_override" not in st.session_state:
        st.session_state.app_class_override = CLASS_NAMES[0]
    if "_prev_idx" not in st.session_state:
        st.session_state._prev_idx = -1
    if "_run_pipeline" not in st.session_state:
        st.session_state._run_pipeline = False

    if args.input:
        st.session_state.review_input = args.input
        st.session_state.app_mode = "review"

    def _on_mode_change():
        st.session_state.app_mode = st.session_state.mode_widget

    st.sidebar.title("BoundingBoxer")

    st.sidebar.selectbox(
        "Mode",
        ["process", "review"],
        format_func=lambda x: x.capitalize(),
        index=0 if st.session_state.app_mode == "process" else 1,
        key="mode_widget",
        on_change=_on_mode_change,
    )

    st.sidebar.markdown("---")

    if st.session_state.app_mode == "process":
        _render_process_mode()
    else:
        _render_review_mode()


if __name__ == "__main__":
    main()
