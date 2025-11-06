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

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd video_pipeline
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

