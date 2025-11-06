# Video Pipeline

A Python-based video processing pipeline that intelligently extracts and packages video frames for LLM analysis using optical flow-based activity detection.

## Overview

This pipeline processes videos by:
1. **Segmenting** videos into time-based chunks (e.g., 10-second segments)
2. **Filtering** segments using optical flow analysis to detect activity
3. **Extracting** frames from active segments at 2 fps
4. **Packaging** frames optimally for LLM consumption (multi-image arrays or filmstrip montages)

## Features

- **Optical Flow Prefiltering**: Uses Farneback optical flow algorithm to calculate activity scores and skip low-activity segments
- **Adaptive Packaging**: Automatically chooses between multi-image mode or filmstrip montage based on LLM provider capabilities
- **Two Pipeline Modes**:
  - `prefilter`: Analyzes optical flow and only processes high-activity segments (recommended)
  - `simple`: Extracts frames at 2 fps without filtering
- **Configurable Thresholds**: Adjust flow score threshold to control segment sensitivity
- **Metadata Tracking**: Generates legend.json files with frame timestamps and processing notes

## Pipeline Modes Explained

The pipeline offers two distinct processing modes, each suited for different use cases:

### Prefilter Mode (Recommended)

**How it works:**
1. Analyzes each video segment using optical flow to detect motion and activity
2. Calculates a "flow score" (90th percentile of optical flow magnitudes)
3. Only processes segments with scores above a threshold (default: 2.5)
4. Extracts frames at 2 fps from active segments

**Advantages:**
- **Cost-effective**: Skips low-activity segments, reducing LLM API costs by 40-70% on typical videos
- **Efficiency**: Processes only meaningful content (scene changes, motion, action)
- **Intelligence**: Automatically filters out static shots, title cards, fade-to-black transitions
- **Configurable**: Adjust threshold to control sensitivity

**Best for:**
- Long videos with static periods (lectures, interviews, surveillance footage)
- Cost-conscious processing where you want to minimize LLM API calls
- Videos where you only care about active/dynamic scenes
- Production workflows where efficiency matters

**Example results:**
A 250-second video (25 segments at 10s each) might skip 13-15 low-activity segments, processing only 10-12 active ones.

### Simple Mode

**How it works:**
1. Extracts frames uniformly at 2 fps from the entire video
2. No analysis or filtering
3. Processes every segment regardless of content

**Advantages:**
- **Complete coverage**: Captures every part of the video
- **Predictable**: Always produces the same number of frames
- **Simple**: No configuration needed, no threshold tuning
- **Guaranteed**: Never misses content due to low motion

**Best for:**
- Short videos where every frame matters
- Content with subtle changes (slow animations, text-heavy presentations)
- When you need complete temporal coverage
- Videos where static scenes contain important information (diagrams, slides)

**Example results:**
A 250-second video always produces 500 frames (2 fps) across all 25 segments.

### Mode Comparison Table

| Feature | Prefilter Mode | Simple Mode |
|---------|---------------|-------------|
| **Processing** | Selective (activity-based) | Complete (all frames) |
| **Cost** | Lower (skips 40-70% typically) | Higher (processes everything) |
| **Setup** | Requires threshold tuning | No configuration |
| **Coverage** | Active scenes only | Full video |
| **Best Use** | Long videos, efficiency | Short videos, completeness |
| **Frame Count** | Variable (depends on activity) | Fixed (2 fps throughout) |

### Choosing the Right Mode

**Use Prefilter mode (`--mode prefilter`) when:**
- Processing long-form content (>5 minutes)
- Video has known static periods
- Reducing costs is important
- You only care about active/dynamic content

**Use Simple mode (`--mode simple`) when:**
- Processing short clips (<2 minutes)
- Every second matters for analysis
- Video contains important static content
- You want predictable, uniform sampling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Alhassan777/TUV_videopipeline.git
cd TUV_videopipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pillow
- Pydantic
- Typer

See `requirements.txt` for specific versions.

## Usage

### Basic Usage

Process a video with default prefilter mode:
```bash
python -m src.main <video_path>
```

### Process with 10-second segments:
```bash
python -m src.main <video_path> --segment-seconds 10
```

### Pipeline-only mode (skip LLM calls):
```bash
python -m src.main <video_path> --segment-seconds 10 --pipeline-only
```

### Adjust flow threshold:
```bash
python -m src.main <video_path> --flow-threshold 3.0
```

### Use simple mode (no prefiltering):
```bash
python -m src.main <video_path> --mode simple
```

## Command-Line Options

- `video`: Path to the input video file (required)
- `--mode`: Pipeline mode - `prefilter` (default) or `simple`
- `--provider`: LLM provider name for logging (default: `stub`)
- `--api-key`: API key for LLM provider (or set `LLM_API_KEY` env variable)
- `--out-root`: Output directory (default: `outputs/main`)
- `--segment-seconds`: Segment length in seconds (0 = process full video)
- `--pipeline-only`: Skip LLM analysis, only run frame extraction
- `--flow-threshold`: Minimum flow score to keep segment (default: 2.5)

## Output Structure

For segmented processing:
```
outputs/main/
├── seg_000/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   ├── ...
│   └── legend.json
├── seg_001/
│   └── skipped.txt  (low-activity segment)
├── seg_002/
│   ├── frame_0000.jpg
│   └── ...
```

## How It Works

### Prefilter Pipeline

1. **Segment Analysis**: For each video segment:
   - Downscales frames to 320px for efficient processing
   - Computes optical flow between consecutive frames using Farneback algorithm
   - Calculates median flow magnitude for each frame pair
   - Computes 90th percentile of magnitudes as segment score

2. **Filtering Decision**: 
   - If score < threshold (default 2.5): Skip segment, write `skipped.txt`
   - If score ≥ threshold: Extract frames at 2 fps

3. **Packaging**:
   - **Multi-image mode**: Individual JPEGs if within LLM limits
   - **Filmstrip mode**: Grid montage if too many frames or total size exceeds limits

### Flow Score Interpretation

The flow score represents the 90th percentile of optical flow magnitudes:
- **< 2.0**: Very low activity (static scenes, minimal motion)
- **2.0-4.0**: Moderate activity (some movement or scene changes)
- **> 4.0**: High activity (significant motion or frequent scene changes)

## Architecture

- `src/main.py`: Main entry point and CLI
- `src/pipeline_prefilter/`: Optical flow analysis and prefiltering logic
- `src/pipeline_simple_2fps/`: Simple 2 fps extraction without filtering
- `src/common/packaging.py`: Frame packaging utilities (JPEG saving, filmstrip generation)
- `src/common/adapter.py`: LLM capability detection and packaging strategy selection

## Flowchart

See `chart.md` for a detailed Mermaid flowchart of the complete pipeline logic.

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

## Contact

[Add contact information here]

