from __future__ import annotations

import os
import math
from typing import Optional, List, Dict, Any
import typer

from src.pipeline_simple_2fps.cli import read_video_frames_2fps
from src.common.packaging import save_frames_as_jpegs, build_filmstrip, write_legend_json
from src.common.adapter import LlmCapabilities, choose_packaging


app = typer.Typer(add_completion=False)


def get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("LLM_API_KEY") or ""
    if not key:
        typer.echo("Warning: no LLM_API_KEY provided; running in dry-run mode.")
    return key


class LlmClientStub:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key

    def analyze(self, frames_paths: List[str], legend: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Stub: return a simple echo structure
        return {
            "provider": self.provider,
            "frames": len(frames_paths),
            "legend_items": len(legend),
            "note": "stubbed LLM response"
        }


def process_segment_with_mode(
    mode: str,
    video: str,
    start_s: Optional[float],
    end_s: Optional[float],
    out_dir: str,
    flow_threshold: float = 2.5,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    if mode == "simple":
        frames = read_video_frames_2fps(video, target_fps=2.0, start_s=start_s, end_s=end_s)
        legend = [{"frame_index": i, "timestamp_s": None, "notes": "simple_2fps"} for i in range(len(frames))]
    elif mode == "prefilter":
        from src.pipeline_prefilter.cli import compute_segment_flow_score, extract_all_frames_2fps
        import logging
        logger = logging.getLogger(__name__)
        
        score = compute_segment_flow_score(video, start_s, end_s)
        logger.info(f"Segment [{start_s},{end_s}] flow score (p90): {score:.4f}")
        
        if score < flow_threshold:
            logger.info(f"Score {score:.4f} < threshold {flow_threshold}; skipping segment")
            # Return empty to signal skip
            return {"paths": [], "legend": [], "skipped": True, "score": score}
        
        logger.info(f"Score {score:.4f} >= threshold; keeping all frames at 2fps")
        frames, legend = extract_all_frames_2fps(video, start_s, end_s, target_fps=2.0)
    else:
        raise typer.BadParameter("mode must be 'simple' or 'prefilter'")

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
        return {"paths": paths, "legend": legend}
    else:
        montage = build_filmstrip(frames, plan.grid_rc, plan.tile_size_px)
        montage_path = f"{out_dir}/montage.jpg"
        montage.save(montage_path, format="JPEG", quality=90)
        write_legend_json(legend, f"{out_dir}/legend.json")
        return {"paths": [montage_path], "legend": legend}


@app.command()
def run(
    video: str = typer.Argument(..., help="Path to video file"),
    mode: str = typer.Option("prefilter", help="Pipeline mode: 'prefilter' or 'simple'"),
    provider: str = typer.Option("stub", help="LLM provider name for logging/routing"),
    api_key: Optional[str] = typer.Option(None, help="API key or use env LLM_API_KEY"),
    out_root: str = typer.Option("outputs/main", help="Output root directory"),
    segment_seconds: int = typer.Option(0, help="If >0, process video in segments of this length"),
    pipeline_only: bool = typer.Option(False, help="Run pipelines only and skip any LLM calls"),
    flow_threshold: float = typer.Option(2.5, help="Prefilter: 90th percentile flow threshold to keep segment"),
):
    key = get_api_key(api_key)
    client = LlmClientStub(provider, key)

    os.makedirs(out_root, exist_ok=True)

    # Segment handling
    if segment_seconds and segment_seconds > 0:
        import cv2
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = total_frames / fps if total_frames else 0
        cap.release()

        num_segments = int(math.ceil(duration_s / segment_seconds))
        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_seconds
            seg_end = min((seg_idx + 1) * segment_seconds, duration_s)
            seg_dir = os.path.join(out_root, f"seg_{seg_idx:03d}")
            result = process_segment_with_mode(mode, video, seg_start, seg_end, seg_dir, flow_threshold)
            
            if result.get("skipped"):
                typer.echo(f"Segment {seg_idx+1}/{num_segments} skipped (score={result.get('score', 0):.2f})")
                with open(os.path.join(seg_dir, "skipped.txt"), "w") as f:
                    f.write(f"Segment skipped: flow_score={result.get('score', 0):.4f} < {flow_threshold}\n")
                continue
            
            if not pipeline_only:
                response = client.analyze(result["paths"], result["legend"])
                write_legend_json([{"segment": seg_idx, "response": response}], os.path.join(seg_dir, "llm_response.json"))
            typer.echo(f"Processed segment {seg_idx+1}/{num_segments}")
    else:
        result = process_segment_with_mode(mode, video, None, None, out_root, flow_threshold)
        if result.get("skipped"):
            typer.echo(f"Full video skipped (score={result.get('score', 0):.2f})")
        elif not pipeline_only:
            response = client.analyze(result["paths"], result["legend"])
            write_legend_json([{"segment": 0, "response": response}], os.path.join(out_root, "llm_response.json"))
            typer.echo("Processed full video")
        else:
            typer.echo("Processed full video")


if __name__ == "__main__":
    app()
