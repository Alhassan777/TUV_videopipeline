from __future__ import annotations

import os
from typing import List, Optional
import cv2
from PIL import Image
import typer

from src.common.packaging import save_frames_as_jpegs, build_filmstrip, write_legend_json
from src.common.adapter import LlmCapabilities, choose_packaging


app = typer.Typer(add_completion=False)


def read_video_frames_2fps(video_path: str, target_fps: float = 2.0, start_s: Optional[float] = None, end_s: Optional[float] = None) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / target_fps)))

    start_frame = 0 if start_s is None else max(0, int(start_s * fps))
    end_frame = (total - 1) if end_s is None else min(total - 1, int(end_s * fps))

    frames: List[Image.Image] = []
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx > end_frame:
            break
        if idx >= start_frame and idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
        ok, frame = cap.read()

    cap.release()
    return frames


@app.command()
def run(
    video: str,
    out_dir: str = "outputs/simple",
):
    os.makedirs(out_dir, exist_ok=True)

    frames = read_video_frames_2fps(video)

    # Estimate per-image bytes for planning (~150KB per JPEG @ q=90 for 512p wide)
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

    legend = [
        {"frame_index": i, "timestamp_s": None, "notes": "simple_2fps"}
        for i in range(len(frames))
    ]

    if plan.mode == "multi_image":
        paths = save_frames_as_jpegs(frames, out_dir)
        write_legend_json(legend, f"{out_dir}/legend.json")
        print(f"Saved {len(paths)} frames to {out_dir}")
    else:
        # filmstrip
        montage = build_filmstrip(frames, plan.grid_rc, plan.tile_size_px)
        montage_path = f"{out_dir}/montage.jpg"
        montage.save(montage_path, format="JPEG", quality=90)
        write_legend_json(legend, f"{out_dir}/legend.json")
        print(f"Saved montage to {montage_path}")


if __name__ == "__main__":
    app()
