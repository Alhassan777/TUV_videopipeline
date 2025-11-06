from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any


Transport = Literal["url", "base64", "attachment"]


@dataclass
class LlmCapabilities:
    text_context_tokens_max: int
    supports_multi_image_parts: bool
    max_images_per_call: Optional[int]
    max_request_bytes: Optional[int]
    preferred_image_transport: Transport
    supported_mime: List[str]
    recommended_megapixels: Optional[float] = None


@dataclass
class PackagingPlan:
    mode: Literal["multi_image", "filmstrip", "pdf"]
    image_transport: Transport
    tile_size_px: Optional[tuple[int, int]] = None
    grid_rc: Optional[tuple[int, int]] = None
    reason: str = ""


def choose_packaging(
    num_frames: int,
    roi_min_sizes: Dict[str, int],
    caps: LlmCapabilities,
    est_bytes_per_image: int,
) -> PackagingPlan:
    # Prefer multi-image if supported and fits limits
    if caps.supports_multi_image_parts:
        if caps.max_images_per_call is None or num_frames <= caps.max_images_per_call:
            total_bytes = num_frames * est_bytes_per_image
            if caps.max_request_bytes is None or total_bytes <= caps.max_request_bytes:
                return PackagingPlan(
                    mode="multi_image",
                    image_transport=caps.preferred_image_transport,
                    reason="fits provider multi-image and size limits",
                )
    # Fallback to filmstrip montage
    # Rough grid: near-square
    import math

    cols = max(1, int(math.ceil(math.sqrt(num_frames))))
    rows = int(math.ceil(num_frames / cols))
    # Start with 512px tiles; final pipeline can scale tiles down if needed
    tile_w, tile_h = 512, 288  # 16:9 tile default
    return PackagingPlan(
        mode="filmstrip",
        image_transport=caps.preferred_image_transport,
        tile_size_px=(tile_w, tile_h),
        grid_rc=(rows, cols),
        reason="fallback to montage to fit all frames in one image",
    )
