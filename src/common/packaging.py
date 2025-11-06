from __future__ import annotations

from typing import List, Dict, Any, Tuple
import json
from PIL import Image
import numpy as np


def save_frames_as_jpegs(frames: List[Image.Image], out_dir: str, base_name: str = "frame") -> List[str]:
    paths: List[str] = []
    for i, img in enumerate(frames):
        path = f"{out_dir}/{base_name}_{i:04d}.jpg"
        img.save(path, format="JPEG", quality=90)
        paths.append(path)
    return paths


def build_filmstrip(frames: List[Image.Image], grid_rc: Tuple[int, int], tile_size: Tuple[int, int]) -> Image.Image:
    rows, cols = grid_rc
    tw, th = tile_size
    canvas = Image.new("RGB", (cols * tw, rows * th), color=(0, 0, 0))

    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        if frame.size != (tw, th):
            frame = frame.resize((tw, th), resample=Image.BICUBIC)
        canvas.paste(frame, (c * tw, r * th))
    return canvas


def write_legend_json(legend: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)
