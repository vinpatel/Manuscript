# Manuscript Video Dataset Download Instructions

This document provides step-by-step instructions for downloading the video samples needed for the Manuscript benchmark. The dataset consists of 50 human-recorded videos and 50 AI-generated videos.

## Prerequisites

- Python 3.8+
- curl or wget
- Sufficient disk space (~5GB recommended)
- API keys for some services (free tier available)

## Directory Structure

```
benchmark/datasets/video/
├── human/              # 50 human-recorded videos
├── ai_generated/       # 50 AI-generated videos
├── metadata.json       # Full metadata for all videos
├── download_videos.py  # Automated download script
└── DOWNLOAD_INSTRUCTIONS.md
```

## Video Specifications

All videos should meet these requirements:
- Duration: 3-30 seconds
- Format: MP4 or WebM
- Maximum file size: 50MB
- Resolution: Minimum 720p recommended

---

## Part 1: Human-Recorded Videos (50 samples)

### Source 1: Pexels Videos (25 videos)

Pexels provides high-quality free stock videos with a generous API.

**Setup:**
1. Create a free account at https://www.pexels.com/
2. Get your API key at https://www.pexels.com/api/new/
3. The API key is free and provides 200 requests/hour

**Automated Download:**
```bash
python download_videos.py --source pexels --pexels-key YOUR_API_KEY --count 25
```

**Manual Download:**
1. Visit https://www.pexels.com/videos/
2. Search for categories: nature, people, city, animals, technology, food, travel, sports, ocean, forest
3. Click on a video and select "Free Download"
4. Choose "Small" or "Medium" size (under 50MB)
5. Save with naming convention: `human_pexels_[category]_[number].mp4`

**Recommended Videos (by category):**
- Nature: landscape, river, forest scenes
- People: walking, talking, activities
- City: traffic, streets, buildings
- Animals: wildlife, pets
- Technology: computers, phones, screens

### Source 2: Pixabay Videos (15 videos)

Pixabay offers royalty-free videos under the Pixabay License.

**Setup:**
1. Create a free account at https://pixabay.com/
2. Get your API key at https://pixabay.com/api/docs/
3. Free tier: 100 requests/minute

**Automated Download:**
```bash
python download_videos.py --source pixabay --pixabay-key YOUR_API_KEY --count 15
```

**Manual Download:**
1. Visit https://pixabay.com/videos/
2. Search for various categories
3. Click "Free Download" and select 1280x720 or smaller
4. Save with naming convention: `human_pixabay_[category]_[number].mp4`

### Source 3: Internet Archive (10 videos)

Public domain videos from the Internet Archive.

**Manual Download Required:**

1. Visit https://archive.org/details/movies
2. Browse collections:
   - Prelinger Archives: https://archive.org/details/prelinger
   - News footage: https://archive.org/details/newsandpublicaffairs
   - Home movies: https://archive.org/details/home_movies

3. For each video:
   - Click "DOWNLOAD OPTIONS"
   - Choose "MPEG4" or "h.264"
   - Download the smallest version under 50MB
   - If video is too long, use ffmpeg to extract a 3-30 second clip:
   ```bash
   ffmpeg -i input.mp4 -ss 00:00:00 -t 00:00:30 -c copy output.mp4
   ```

4. Save with naming convention: `human_archive_[type]_[number].mp4`

**Recommended Archive Items:**
- Educational films from 1950s-1960s
- News reels
- Home movies
- Public service announcements
- Classic commercials (pre-1978)

---

## Part 2: AI-Generated Videos (50 samples)

### Source 1: OpenAI Sora (10 videos)

Sora demos are available on OpenAI's website.

**Download Instructions:**
1. Visit https://openai.com/sora
2. Scroll to view demo videos
3. Right-click on videos and "Save video as..."
4. If right-click is disabled, use browser developer tools:
   - Press F12 to open DevTools
   - Go to Network tab
   - Play the video
   - Filter by "media" or ".mp4"
   - Copy the video URL and download with curl
5. Save with naming convention: `ai_sora_[description]_[number].mp4`

**Alternative: Sora on X/Twitter**
Many Sora demos are shared on OpenAI's Twitter. Use a video downloader tool.

### Source 2: Runway ML Gen-2/Gen-3 (10 videos)

Runway requires a free account with limited credits.

**Setup:**
1. Create account at https://runwayml.com/
2. New accounts get free credits for generation
3. Go to Gen-2 or Gen-3 Alpha workspace

**Generate Videos:**
1. Choose "Text to Video" or "Image to Video"
2. Enter prompts like:
   - "A serene mountain lake at sunset"
   - "A person walking through a crowded market"
   - "Ocean waves crashing on rocks"
3. Generate and download
4. Save with naming convention: `ai_runway_[type]_[number].mp4`

**Alternative: Runway Research Page**
Visit https://runwayml.com/research/ for sample videos.

### Source 3: Pika Labs (5 videos)

**Setup:**
1. Join Pika at https://pika.art/
2. Sign up for free tier access

**Generate Videos:**
1. Use text prompts or upload images
2. Download generated videos
3. Save with naming convention: `ai_pika_[type]_[number].mp4`

### Source 4: Stable Video Diffusion (5 videos)

**Option A: Hugging Face Spaces**
1. Visit https://huggingface.co/spaces/stabilityai/stable-video-diffusion
2. Upload an image
3. Generate and download video

**Option B: Local Generation**
```bash
# Install dependencies
pip install diffusers transformers accelerate

# Run generation script (requires GPU)
python -c "
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    torch_dtype=torch.float16
)
pipe.to('cuda')

image = load_image('input_image.jpg')
image = image.resize((1024, 576))

frames = pipe(image, decode_chunk_size=8).frames[0]
export_to_video(frames, 'ai_svd_output.mp4', fps=7)
"
```

### Source 5: Deepfake Datasets (5 videos)

These datasets require academic access.

**FaceForensics++:**
1. Visit https://github.com/ondyari/FaceForensics
2. Fill out the Google Form for access
3. Wait for approval (usually 1-2 weeks)
4. Download selected manipulation types:
   - Deepfakes
   - Face2Face
   - FaceSwap
   - NeuralTextures

**DFDC (DeepFake Detection Challenge):**
1. Visit https://www.kaggle.com/c/deepfake-detection-challenge
2. Accept competition rules
3. Download sample videos from the dataset

**Celeb-DF:**
1. Visit https://github.com/yuezunli/celeb-deepfakeforensics
2. Submit access request via Google Form
3. Download upon approval

### Source 6: Other AI Generators (15 videos)

**Kling AI (3 videos):**
- Visit https://klingai.com/
- Create account and generate videos

**Luma Dream Machine (3 videos):**
- Visit https://lumalabs.ai/dream-machine
- Free tier available with limited generations

**Haiper AI (2 videos):**
- Visit https://haiper.ai/
- Create account for free credits

**MiniMax (2 videos):**
- Visit https://www.minimax.chat/
- Access Video-01 generation feature

**Genmo Mochi (2 videos):**
- Visit https://www.genmo.ai/
- Or run Mochi locally: https://github.com/genmoai/mochi

**CogVideoX (1 video):**
- Run locally via Hugging Face
- https://huggingface.co/THUDM/CogVideoX-5b

**AnimateDiff (1 video):**
- Use with ComfyUI or Automatic1111
- https://github.com/guoyww/AnimateDiff

**ModelScope (1 video):**
- https://huggingface.co/damo-vilab/text-to-video-ms-1.7b

---

## Validation Script

After downloading, run this script to validate your dataset:

```python
import os
import json
from pathlib import Path

def validate_dataset():
    base_dir = Path(__file__).parent
    human_dir = base_dir / "human"
    ai_dir = base_dir / "ai_generated"

    human_count = len(list(human_dir.glob("*.mp4"))) + len(list(human_dir.glob("*.webm")))
    ai_count = len(list(ai_dir.glob("*.mp4"))) + len(list(ai_dir.glob("*.webm")))

    print(f"Human videos: {human_count}/50")
    print(f"AI videos: {ai_count}/50")

    # Check file sizes
    for video in human_dir.glob("*"):
        size_mb = video.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            print(f"WARNING: {video.name} exceeds 50MB ({size_mb:.1f}MB)")

    for video in ai_dir.glob("*"):
        size_mb = video.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            print(f"WARNING: {video.name} exceeds 50MB ({size_mb:.1f}MB)")

if __name__ == "__main__":
    validate_dataset()
```

---

## Updating Metadata

After downloading videos, update `metadata.json` with actual values:

```python
import json
import subprocess
from pathlib import Path

def get_video_info(video_path):
    """Get video duration and resolution using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    duration = float(data['format'].get('duration', 0))
    for stream in data.get('streams', []):
        if stream['codec_type'] == 'video':
            width = stream.get('width', 0)
            height = stream.get('height', 0)
            return duration, f"{width}x{height}"
    return duration, "unknown"

# Update metadata for each video
# ... implementation details
```

---

## Troubleshooting

### API Rate Limits
- Pexels: 200 requests/hour - add delays between downloads
- Pixabay: 100 requests/minute - add delays between downloads

### Large Files
Use ffmpeg to compress:
```bash
ffmpeg -i input.mp4 -vf scale=1280:720 -c:v libx264 -crf 23 output.mp4
```

### Long Videos
Extract a clip:
```bash
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:20 -c copy output.mp4
```

### Format Conversion
Convert to MP4:
```bash
ffmpeg -i input.webm -c:v libx264 -c:a aac output.mp4
```

---

## Legal Notes

- **Pexels License**: Free for commercial use, no attribution required
- **Pixabay License**: Free for commercial use, no attribution required
- **Public Domain**: No restrictions
- **Deepfake Datasets**: Research use only, academic access required
- **AI Generator Outputs**: Check individual platform terms of service

Always verify licensing terms before using videos in production systems.

---

## Contact

For questions about the Manuscript video benchmark dataset, please open an issue on the repository.
