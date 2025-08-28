# Mono Latent Walk Generator

A sophisticated latent space interpolation system for generating 30-second video artworks using StyleGAN2-ADA. This tool creates smooth transitions through latent space to produce unique, frame-ready video content.

## 🎯 Features

- **900 Frame Generation**: Creates exactly 900 frames for 30-second videos at 30 FPS
- **37 Keyframe System**: Uses 37 latent vectors with 36 smooth transitions 
- **SLERP Interpolation**: Spherical Linear Interpolation for smooth latent walks
- **Reproducible Output**: Saves seed information for reproducible generation
- **UUID-based Organization**: Each generation session gets a unique identifier
- **StyleGAN2-ADA Compatible**: Works with trained StyleGAN2-ADA models

## 📊 Technical Specifications

- **Total Frames**: 900 (30 seconds × 30 FPS)
- **Keyframes**: 37 latent vectors
- **Transitions**: 36 (between each pair of keyframes)
- **Frames per Transition**: 25
- **Resolution**: 512×512 (configurable via model)
- **Interpolation**: SLERP (Spherical Linear Interpolation)
- **Output Format**: PNG sequence ready for FFmpeg compilation

## 🚀 Quick Start

### 1. Prerequisites

Install required dependencies:

```bash
# Install PyTorch (adjust for your system)
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### 2. Model Setup

Place your trained StyleGAN2-ADA model at:
```
/workspace/stylegan-stack/models/model.pkl
```

### 3. Generate Frames

Run the latent walk generator:

```bash
python3 latent_walk.py
```

### 4. Output

Frames will be saved to:
```
/workspace/stylegan-stack/generated_frames/{uuid}/
├── frame_0000.png
├── frame_0001.png
├── ...
├── frame_0899.png
└── latent_path.json
```

## 🛠️ Advanced Usage

### Custom Seeds

Generate reproducible output with fixed seeds:

```bash
python3 latent_walk.py --seeds 1 2 3 4 5 6 7 8 9 10
```

### Custom Model Path

Use a different model location:

```bash
python3 latent_walk.py --model /path/to/your/model.pkl
```

### Custom Output Directory

Specify a different output location:

```bash
python3 latent_walk.py --output_dir /path/to/output
```

### Truncation Control

Adjust the StyleGAN2 truncation parameter:

```bash
python3 latent_walk.py --truncation_psi 0.7
```

### Full Command Example

```bash
python3 latent_walk.py \
    --model /workspace/stylegan-stack/models/model.pkl \
    --output_dir /workspace/stylegan-stack/generated_frames \
    --seeds 42 123 456 789 \
    --truncation_psi 0.8 \
    --device cuda
```

## 📁 Directory Structure

```
stylegan-stack/
├── latent_walk.py          # Main generation script
├── requirements.txt        # Python dependencies
├── test_basic.py          # Basic validation tests
├── test_structure.py      # Full test suite (requires deps)
├── README.md              # This file
├── models/                # StyleGAN2 model storage
│   └── model.pkl         # Your trained model (place here)
└── generated_frames/      # Output directory
    └── {uuid}/           # Session-specific folders
        ├── frame_0000.png
        ├── ...
        ├── frame_0899.png
        └── latent_path.json
```

## 🎨 Generation Process

1. **Keyframe Generation**: Creates 37 random latent vectors in StyleGAN2's latent space
2. **Path Planning**: Determines interpolation path through all keyframes
3. **SLERP Interpolation**: Smoothly interpolates between each pair of keyframes
4. **Frame Rendering**: Generates 900 individual frames using StyleGAN2
5. **Sequential Saving**: Saves frames as numbered PNG files
6. **Metadata Logging**: Records generation parameters for reproducibility

## 🔧 Configuration Details

### Frame Calculation
- **Keyframes**: 37 vectors
- **Transitions**: 36 (between adjacent keyframes)
- **Frames per Transition**: 25
- **Total Frames**: 36 × 25 = 900
- **Duration**: 900 ÷ 30 FPS = 30 seconds

### Interpolation Method
The script uses SLERP (Spherical Linear Interpolation) which:
- Maintains constant magnitude during interpolation
- Provides smooth, natural transitions in latent space
- Avoids the "shortcut" artifacts of linear interpolation
- Creates more visually appealing morphing effects

### Output Format
- **Image Format**: PNG (lossless, high quality)
- **Naming Convention**: `frame_%04d.png` (frame_0000.png to frame_0899.png)
- **Metadata**: `latent_path.json` contains seeds and generation parameters

## 🎬 Video Compilation

After generating frames, compile to video using FFmpeg:

```bash
# Navigate to the output directory
cd /workspace/stylegan-stack/generated_frames/{uuid}/

# Create MP4 video
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4

# Create high-quality version
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p output_hq.mp4
```

## 🧪 Testing

Run basic validation tests:

```bash
python3 test_basic.py
```

Run full test suite (requires dependencies):

```bash
python3 test_structure.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure your StyleGAN2-ADA model is at `/workspace/stylegan-stack/models/model.pkl`
   - Check file permissions and path

2. **CUDA out of memory**
   - Use `--device cpu` to run on CPU
   - Reduce batch size in the code if needed

3. **Import errors**
   - Install StyleGAN2-ADA repository
   - Ensure all dependencies from `requirements.txt` are installed

4. **Generation too slow**
   - Use GPU acceleration with `--device cuda`
   - Consider using a smaller model or lower resolution

### Performance Tips

- **GPU Usage**: Always use `--device cuda` if available
- **Model Size**: Smaller models generate faster but with lower quality
- **Truncation**: Lower `truncation_psi` values (0.5-0.8) can speed up generation
- **Batch Processing**: The script processes one frame at a time for memory efficiency

## 📈 Output Analysis

Each generation session produces:

1. **900 PNG frames** ready for video compilation
2. **latent_path.json** containing:
   - All 37 seed values for reproducibility
   - Generation parameters
   - Timestamp and session ID
   - Latent vectors (optional, for analysis)

## 🎯 Integration Notes

This generator is designed to integrate with:

- **FFmpeg** for video compilation
- **Video processing pipelines** for post-effects
- **Batch generation systems** for multiple Monos
- **Quality analysis tools** for frame assessment

## 📄 License

This tool is part of the Mono Project generative video pipeline. Please ensure you have appropriate licenses for StyleGAN2-ADA and any trained models you use.

---

**Generated frames are ready for immediate video compilation. No additional processing required.**