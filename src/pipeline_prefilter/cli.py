from __future__ import annotations

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import typer

from src.common.packaging import save_frames_as_jpegs, build_filmstrip, write_legend_json
from src.common.adapter import LlmCapabilities, choose_packaging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

app = typer.Typer(add_completion=False)


def downscale(img: np.ndarray, short_side: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) <= short_side:
        return img
    if h < w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_segment_flow_score(
    video_path: str,
    start_s: Optional[float],
    end_s: Optional[float],
    short_side: int = 320,
) -> float:
    """
    Compute 90th percentile optical flow magnitude for a segment.
    Returns a single score indicating activity level.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = 0 if start_s is None else max(0, int(start_s * fps))
    end_frame = (total - 1) if end_s is None else min(total - 1, int(end_s * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return 0.0

    prev = cv2.cvtColor(downscale(frame, short_side), cv2.COLOR_BGR2GRAY)
    mags: List[float] = []

    idx = start_frame + 1
    ok, frame = cap.read()
    while ok and idx <= end_frame:
        small = cv2.cvtColor(downscale(frame, short_side), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.median(mag)))
        prev = small
        idx += 1
        ok, frame = cap.read()

    cap.release()

    if not mags:
        return 0.0
    return float(np.percentile(mags, 90))


def extract_all_frames_2fps(
    video_path: str,
    start_s: Optional[float],
    end_s: Optional[float],
    target_fps: float = 2.0,
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    """
    Extract all frames at target_fps from the segment.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / target_fps)))

    start_frame = 0 if start_s is None else max(0, int(start_s * fps))
    end_frame = (total - 1) if end_s is None else min(total - 1, int(end_s * fps))

    frames: List[Image.Image] = []
    legend: List[Dict[str, Any]] = []

    idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    while ok and idx <= end_frame:
        if (idx - start_frame) % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            legend.append({
                "frame_index": int(idx),
                "timestamp_s": float(idx / fps),
                "notes": "prefilter_kept_segment"
            })
        idx += 1
        ok, frame = cap.read()

    cap.release()
    return frames, legend


@app.command()
def run(
    video: str,
    out_dir: str = "outputs/prefilter",
    flow_threshold: float = typer.Option(2.5, help="90th percentile flow threshold to keep segment"),
):
    logger.info(f"Prefilter pipeline starting: video={video}, threshold={flow_threshold}")
    os.makedirs(out_dir, exist_ok=True)

    # Compute flow score for the entire video (or segment if called from main.py)
    score = compute_segment_flow_score(video, None, None)
    logger.info(f"Segment flow score (p90): {score:.4f}")

    if score < flow_threshold:
        logger.info(f"Score {score:.4f} < threshold {flow_threshold}; skipping segment (no activity)")
        # Write empty outputs
        write_legend_json([], f"{out_dir}/legend.json")
        with open(f"{out_dir}/skipped.txt", "w") as f:
            f.write(f"Segment skipped: flow_score={score:.4f} < {flow_threshold}\n")
        return

    logger.info(f"Score {score:.4f} >= threshold {flow_threshold}; keeping all frames at 2fps")
    frames, legend = extract_all_frames_2fps(video, None, None, target_fps=2.0)

    est_bytes_per_image = 150_000
    caps = LlmCapabilities(
        text_context_tokens_max=128_000,
        supports_multi_image_parts=True,
        max_images_per_call=None,
        max_request_bytes=8_000_000,
        preferred_image_transport="base64",
        supported_mime=["image/jpeg", "image/png"],
        recommended_megapixels=1.0,
    )

    plan = choose_packaging(len(frames), {}, caps, est_bytes_per_image)

    if plan.mode == "multi_image":
        paths = save_frames_as_jpegs(frames, out_dir)
        write_legend_json(legend, f"{out_dir}/legend.json")
        logger.info(f"Saved {len(paths)} frames to {out_dir}")
    else:
        montage = build_filmstrip(frames, plan.grid_rc, plan.tile_size_px)
        montage_path = f"{out_dir}/montage.jpg"
        montage.save(montage_path, format="JPEG", quality=90)
        write_legend_json(legend, f"{out_dir}/legend.json")
        logger.info(f"Saved montage to {montage_path}")


if __name__ == "__main__":
    app()
