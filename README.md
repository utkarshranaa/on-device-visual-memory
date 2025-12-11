# On-Device Visual Memory

**Search your camera footage by meaning - entirely offline.**

Capture frames from a camera, compute CLIP embeddings on-device, store them in a local vector index, and search using natural language. No cloud APIs, no subscription fees, complete privacy.

## What It Solves

Ever needed to find something in hours of camera footage but only remembered what it *looked* like? On-Device Visual Memory lets you search your visual recordings using natural language queries like "a person sitting at a desk" or "bright room with windows". Everything runs locally on your device - perfect for edge deployments on NVIDIA Jetson or privacy-sensitive applications.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Camera    │─────▶│ Frame Capture│─────▶│  JPEG Files │
│  (webcam)   │      │   (opencv)   │      │ data/frames │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐      ┌─────────────┐
                     │   Metadata   │─────▶│   SQLite    │
                     │    Store     │      │ meta.db     │
                     └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐      ┌─────────────┐
                     │  CLIP Embed  │─────▶│   ANN Index │
                     │   (torch)    │      │faiss/hnswlib│
                     └──────────────┘      └─────────────┘
                            │                            │
                            ▼                            ▼
                     ┌──────────────┐            ┌─────────────┐
                     │  Text Query  │────────────│  Nearest    │
                     │  Embedding   │            │  Neighbors  │
                     └──────────────┘            └─────────────┘
                                                         │
                                                         ▼
                                                  ┌─────────────┐
                                                  │   Streamlit │
                                                  │   Web UI    │
                                                  └─────────────┘
```

## Features

- **Offline-first**: No cloud APIs, all computation on-device
- **Semantic search**: Find images by meaning, not just keywords
- **Incremental indexing**: Add new frames without rebuilding the entire index
- **Multi-platform**: Runs on NVIDIA Jetson, x86 Linux, macOS, and Windows (WSL2)
- **GPU acceleration**: Automatic CUDA detection and usage when available
- **Dual backend**: FAISS (preferred) with automatic fallback to hnswlib

## Requirements

- Python 3.10 or higher
- 4GB+ RAM recommended
- Camera device (USB webcam or built-in)
- 500MB+ disk space for model weights

## Installation

```bash
# Clone or download this repository
cd on-device-visual-memory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Platform-Specific Notes

**NVIDIA Jetson (JetPack 5.x/6.x):**
```bash
# PyTorch is pre-installed on JetPack
pip install numpy opencv-python streamlit open-clip-torch faiss-cpu psutil
```

**Windows (WSL2):**
```bash
# Install WSL2 with Ubuntu, then:
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
# Follow standard installation above
```

**macOS (Apple Silicon):**
```bash
# FAISS may require building from source
pip install faiss-cpu --no-cache-dir
```

## Quick Start

### 1. Capture Frames

```bash
python src/capture_frames.py
```

Options:
- `--device 0` - Camera device ID (default: 0)
- `--interval 2.0` - Seconds between captures (default: 2.0)
- `--width 1280` - Frame width (default: 1280)
- `--height 720` - Frame height (default: 720)
- `--max-frames 100` - Stop after N frames (default: unlimited)

Press Ctrl+C to stop.

### 2. Build the Vector Index

```bash
python src/build_index.py
```

Options:
- `--batch-size 32` - Frames per batch (default: 32)
- `--max-new 100` - Maximum new frames to index (default: all)

### 3. Search with Web UI

```bash
streamlit run src/search_app.py
```

The app will open in your browser at `http://localhost:8501`

### 4. Run Benchmarks

```bash
python src/benchmark.py
```

## Deployment

### NVIDIA Jetson

1. **Install JetPack SDK** (includes CUDA, cuDNN, PyTorch)
2. **Verify CUDA:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set camera permissions:**
   ```bash
   sudo usermod -aG video $USER
   # Log out and back in
   ```

### x86 Linux / macOS / Windows (WSL2)

Standard installation as above. Verify camera access:
```bash
# Linux
v4l2-ctl --list-devices

# macOS
system_profiler SP CamerasDataType

# WSL2
# Camera passthrough requires USB/IPD setup
```

## Troubleshooting

### FAISS Import Errors

**Error:** `ImportError: library not found`

**Solution:**
```bash
# Try pip install with specific version
pip install faiss-cpu==1.7.4

# Or build from source
sudo apt install libopenblas-dev
pip install faiss-cpu --no-binary faiss-cpu
```

### CUDA Not Available on Jetson

**Verify CUDA installation:**
```bash
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

**Common issues:**
- Ensure JetPack is properly installed
- Check LD_LIBRARY_PATH includes CUDA libraries
- Reboot after JetPack installation

### Camera Not Opening

**Linux/Jetson:**
```bash
# Check camera device
ls -l /dev/video*

# Test with ffmpeg
ffmpeg -f v4l2 -i /dev/video0 output.mpg
```

**Windows (WSL2):**
- Native WSL2 doesn't support USB cameras directly
- Use USB/IPD for camera passthrough
- Or run capture_frames.py on Windows and index on WSL

### Out of Memory Errors

- Reduce batch size: `--batch-size 16`
- Use smaller model: Change `CLIP_MODEL_NAME` in `config.py` to `"ViT-B-32"`
- Close other applications

## Limitations

- **Storage growth**: Each frame ~100KB JPEG; 1 hour @ 2fps = ~3.6MB
- **Capture interval tradeoff**: Shorter intervals = more storage, better temporal resolution
- **Index rebuild time**: First index build can take time proportional to frame count
- **Query language accuracy**: CLIP embeddings work best with descriptive English queries
- **Fixed resolution**: Downscaling very high-res inputs recommended

## Project Structure

```
on-device-visual-memory/
├── src/
│   ├── capture_frames.py   # Camera capture script
│   ├── build_index.py      # Vector index builder
│   ├── search_app.py       # Streamlit search UI
│   ├── embedder.py         # CLIP embedding wrapper
│   ├── store.py            # SQLite metadata store
│   ├── config.py           # Configuration constants
│   └── benchmark.py        # Performance benchmarking
├── data/
│   ├── frames/             # Captured frame images
│   └── index/              # Database and vector index
│       ├── meta.db         # SQLite metadata
│       └── vector.index    # FAISS/hnswlib index
├── requirements.txt
├── README.md
└── LICENSE
```

## Performance Benchmarks

Typical performance on NVIDIA Jetson Orin:

| Operation | Latency |
|-----------|---------|
| Image embedding (GPU) | ~15-25ms |
| Text embedding (GPU) | ~5-10ms |
| Search 10k vectors (k=10) | ~2-5ms |

*Benchmarks vary by hardware and model selection.*

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

## Acknowledgments

- OpenCLIP for the pretrained CLIP models
- FAISS by Facebook Research for efficient similarity search
- Streamlit for the web UI framework
