#!/bin/bash

# Manuscript Benchmark Dataset Download Script
# Downloads all datasets needed for the benchmark evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="$BENCHMARK_DIR/datasets"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Manuscript Benchmark Dataset Downloader${NC}"
echo -e "${BLUE}========================================${NC}"

# Create directory structure
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p "$DATASETS_DIR"/{text,image,audio,video}/{human,ai_generated}
mkdir -p "$BENCHMARK_DIR/cache"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check requirements
echo -e "\n${YELLOW}Checking requirements...${NC}"
MISSING_DEPS=0

if ! command_exists curl; then
    echo -e "${RED}Error: curl is not installed${NC}"
    MISSING_DEPS=1
fi

if ! command_exists python3; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}Please install missing dependencies and try again${NC}"
    exit 1
fi

echo -e "${GREEN}All requirements satisfied${NC}"

# =============================================================================
# TEXT DATASETS
# =============================================================================
download_text_datasets() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Downloading Text Datasets${NC}"
    echo -e "${BLUE}========================================${NC}"

    TEXT_HUMAN_DIR="$DATASETS_DIR/text/human"
    TEXT_AI_DIR="$DATASETS_DIR/text/ai_generated"

    # Check if Python datasets library is available
    if python3 -c "import datasets" 2>/dev/null; then
        echo -e "${YELLOW}Downloading HC3 dataset via HuggingFace...${NC}"

        python3 << 'EOF'
import os
import json
from datasets import load_dataset

datasets_dir = os.environ.get('DATASETS_DIR', './datasets')
human_dir = f"{datasets_dir}/text/human"
ai_dir = f"{datasets_dir}/text/ai_generated"

print("Loading HC3 dataset...")
dataset = load_dataset("Hello-SimpleAI/HC3", "all")

human_samples = []
ai_samples = []

# Extract samples from the dataset
for split in ['train']:
    if split in dataset:
        for i, item in enumerate(dataset[split]):
            if len(human_samples) >= 50 and len(ai_samples) >= 50:
                break

            # Human answers
            if 'human_answers' in item and item['human_answers'] and len(human_samples) < 50:
                for j, answer in enumerate(item['human_answers'][:1]):
                    if len(answer) > 100 and len(human_samples) < 50:
                        human_samples.append({
                            'text': answer,
                            'question': item.get('question', ''),
                            'source': 'HC3',
                            'type': 'qa_response'
                        })

            # ChatGPT answers
            if 'chatgpt_answers' in item and item['chatgpt_answers'] and len(ai_samples) < 50:
                for j, answer in enumerate(item['chatgpt_answers'][:1]):
                    if len(answer) > 100 and len(ai_samples) < 50:
                        ai_samples.append({
                            'text': answer,
                            'question': item.get('question', ''),
                            'source': 'HC3',
                            'type': 'qa_response',
                            'generator': 'ChatGPT'
                        })

# Save human samples
print(f"Saving {len(human_samples)} human samples...")
human_metadata = []
for i, sample in enumerate(human_samples):
    filename = f"human_hc3_{i+1:03d}.txt"
    filepath = f"{human_dir}/{filename}"
    with open(filepath, 'w') as f:
        f.write(sample['text'])
    human_metadata.append({
        'filename': filename,
        'source': sample['source'],
        'type': sample['type'],
        'word_count': len(sample['text'].split()),
        'question': sample['question'][:100] + '...' if len(sample['question']) > 100 else sample['question']
    })

with open(f"{human_dir}/metadata.json", 'w') as f:
    json.dump(human_metadata, f, indent=2)

# Save AI samples
print(f"Saving {len(ai_samples)} AI samples...")
ai_metadata = []
for i, sample in enumerate(ai_samples):
    filename = f"ai_chatgpt_{i+1:03d}.txt"
    filepath = f"{ai_dir}/{filename}"
    with open(filepath, 'w') as f:
        f.write(sample['text'])
    ai_metadata.append({
        'filename': filename,
        'source': sample['source'],
        'type': sample['type'],
        'generator': sample['generator'],
        'word_count': len(sample['text'].split()),
        'question': sample['question'][:100] + '...' if len(sample['question']) > 100 else sample['question']
    })

with open(f"{ai_dir}/metadata.json", 'w') as f:
    json.dump(ai_metadata, f, indent=2)

print(f"Text dataset download complete: {len(human_samples)} human, {len(ai_samples)} AI samples")
EOF
        echo -e "${GREEN}Text datasets downloaded successfully${NC}"
    else
        echo -e "${YELLOW}Python 'datasets' library not found. Install with: pip install datasets${NC}"
        echo -e "${YELLOW}Creating placeholder text files...${NC}"

        # Create README for manual download
        cat > "$TEXT_HUMAN_DIR/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# Text Dataset Download Instructions

## HC3 Dataset (Recommended)

1. Install the datasets library:
   ```bash
   pip install datasets
   ```

2. Download using Python:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("Hello-SimpleAI/HC3", "all")
   ```

3. Or download manually from:
   https://huggingface.co/datasets/Hello-SimpleAI/HC3

## Alternative Sources

- **Defactify-Text**: https://arxiv.org/abs/2510.22874
- **Beemo Benchmark**: https://toloka.ai/ai-detection-benchmark
EOF
        cp "$TEXT_HUMAN_DIR/DOWNLOAD_INSTRUCTIONS.md" "$TEXT_AI_DIR/"
    fi
}

# =============================================================================
# IMAGE DATASETS
# =============================================================================
download_image_datasets() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Downloading Image Datasets${NC}"
    echo -e "${BLUE}========================================${NC}"

    IMAGE_HUMAN_DIR="$DATASETS_DIR/image/human"
    IMAGE_AI_DIR="$DATASETS_DIR/image/ai_generated"

    # Download sample images from Unsplash (using their source API)
    echo -e "${YELLOW}Downloading human images from Unsplash...${NC}"

    # Sample Unsplash image URLs (using unsplash.com/photos format)
    UNSPLASH_IDS=(
        "photo-1682687220063-4742bd7fd538"
        "photo-1506905925346-21bda4d32df4"
        "photo-1469474968028-56623f02e42e"
        "photo-1447752875215-b2761acb3c5d"
        "photo-1433086966358-54859d0ed716"
    )

    count=0
    for id in "${UNSPLASH_IDS[@]}"; do
        if [ $count -ge 5 ]; then break; fi
        filename="human_unsplash_$(printf '%03d' $((count+1))).jpg"
        echo "  Downloading $filename..."
        curl -sL "https://images.unsplash.com/${id}?w=800&q=80" -o "$IMAGE_HUMAN_DIR/$filename" 2>/dev/null || true
        ((count++))
    done

    # Create metadata
    cat > "$IMAGE_HUMAN_DIR/metadata.json" << 'EOF'
[
  {"filename": "human_unsplash_001.jpg", "source": "unsplash", "content": "nature landscape", "license": "Unsplash License"},
  {"filename": "human_unsplash_002.jpg", "source": "unsplash", "content": "mountain scene", "license": "Unsplash License"},
  {"filename": "human_unsplash_003.jpg", "source": "unsplash", "content": "forest path", "license": "Unsplash License"},
  {"filename": "human_unsplash_004.jpg", "source": "unsplash", "content": "waterfall", "license": "Unsplash License"},
  {"filename": "human_unsplash_005.jpg", "source": "unsplash", "content": "sunset view", "license": "Unsplash License"}
]
EOF

    # Create instructions for AI images
    cat > "$IMAGE_AI_DIR/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# AI Image Dataset Download Instructions

## GenImage Dataset (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/GenImage-Dataset/GenImage
   ```

2. Follow the download instructions in the repository

## Alternative Sources

### Lexica.art (Stable Diffusion)
- Website: https://lexica.art
- Contains millions of Stable Diffusion generated images
- Free to browse and download

### CIVITAI
- Website: https://civitai.com
- Large collection of AI-generated images
- Includes model information

### MS COCOAI Dataset
- Paper: https://arxiv.org/abs/2601.00553
- Contains DALL-E 3, Midjourney v6, Stable Diffusion 3 images

## Manual Collection Guidelines

When collecting AI images, ensure you:
1. Record the generator model (DALL-E 3, Midjourney v6, SD3, etc.)
2. Save the original prompt if available
3. Maintain original image quality (no re-compression)
4. Document the source URL
EOF

    echo -e "${GREEN}Image dataset structure created${NC}"
}

# =============================================================================
# AUDIO DATASETS
# =============================================================================
download_audio_datasets() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Downloading Audio Datasets${NC}"
    echo -e "${BLUE}========================================${NC}"

    AUDIO_HUMAN_DIR="$DATASETS_DIR/audio/human"
    AUDIO_AI_DIR="$DATASETS_DIR/audio/ai_generated"

    echo -e "${YELLOW}Note: Full audio datasets are large. Creating download instructions...${NC}"

    # Create download instructions for human audio
    cat > "$AUDIO_HUMAN_DIR/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# Human Audio Dataset Download Instructions

## LJSpeech Dataset (Recommended)

**Size:** ~2.6 GB (13,100 audio clips)
**License:** Public Domain

```bash
# Download
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# Extract
tar -xjf LJSpeech-1.1.tar.bz2

# Copy samples to this directory
cp LJSpeech-1.1/wavs/LJ001-*.wav ./
```

## LibriSpeech Dataset

**Size:** Varies by subset (100 hours to 1000 hours)
**License:** CC BY 4.0

```bash
# Download test-clean subset (~350 MB)
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
```

## Free Music Archive

- Website: https://freemusicarchive.org
- Filter by Creative Commons licenses
- Good for music samples
EOF

    # Create download instructions for AI audio
    cat > "$AUDIO_AI_DIR/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# AI Audio Dataset Download Instructions

## WaveFake Dataset (Recommended)

**Size:** ~10 GB
**License:** CC BY 4.0

```bash
# Download from Zenodo
wget https://zenodo.org/record/5642694/files/wavefake.zip
unzip wavefake.zip
```

Contains audio generated by:
- MelGAN
- Full-band MelGAN
- Multi-band MelGAN
- HiFi-GAN
- Parallel WaveGAN
- WaveGlow
- WaveRNN

## ElevenLabs Samples

ElevenLabs provides an AI speech classifier for their own generated content:
https://elevenlabs.io/ai-speech-classifier

To collect ElevenLabs samples:
1. Use the free tier to generate sample audio
2. Save with metadata about voice and settings

## ASVspoof Dataset

- Website: https://www.asvspoof.org/
- Requires registration
- Contains spoofed/synthetic speech samples

## Collection Guidelines

When collecting AI audio, record:
1. Generator tool (ElevenLabs, WaveFake, Suno, etc.)
2. Voice model used
3. Text prompt/script
4. Audio format and quality settings
EOF

    echo -e "${GREEN}Audio dataset instructions created${NC}"
}

# =============================================================================
# VIDEO DATASETS
# =============================================================================
download_video_datasets() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Downloading Video Datasets${NC}"
    echo -e "${BLUE}========================================${NC}"

    VIDEO_HUMAN_DIR="$DATASETS_DIR/video/human"
    VIDEO_AI_DIR="$DATASETS_DIR/video/ai_generated"

    echo -e "${YELLOW}Note: Video datasets are large. Creating download instructions...${NC}"

    # Create download instructions for human video
    cat > "$VIDEO_HUMAN_DIR/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# Human Video Dataset Download Instructions

## Pexels Videos (Recommended)

**License:** Pexels License (free for commercial use)

```bash
# Use Pexels API (requires free API key)
# https://www.pexels.com/api/

# Example with curl:
curl -H "Authorization: YOUR_API_KEY" \
  "https://api.pexels.com/videos/search?query=nature&per_page=10"
```

## Pixabay Videos

- Website: https://pixabay.com/videos/
- License: Pixabay License (free)
- No API key required for browsing

## Internet Archive

- Website: https://archive.org/details/movies
- Contains public domain videos
- Various formats and qualities

## Collection Guidelines

For authentic human videos:
1. Prefer original uploads (not re-encoded)
2. Include diverse content (vlogs, nature, events)
3. Maintain original metadata
4. Record source URL and license
EOF

    # Create download instructions for AI video
    cat > "$VIDEO_AI_DIR/DOWNLOAD_INSTRUCTIONS.md" << 'EOF'
# AI Video Dataset Download Instructions

## DeepfakeBench / DF40 (Recommended)

**License:** MIT

```bash
# Clone the repository
git clone https://github.com/SCLBD/DeepfakeBench

# Follow dataset preparation instructions
cd DeepfakeBench
python download_dataset.py
```

## FaceForensics++

**Note:** Requires access request

```bash
# Request access at:
# https://github.com/ondyari/FaceForensics

# After approval, use the download script provided
```

## Deepfake-Eval-2024

- Paper: https://arxiv.org/html/2503.02857v1
- Contains in-the-wild deepfakes from 2024
- Includes Sora-generated content samples

## Sora/Runway Public Samples

OpenAI and Runway have released demo videos:
- Sora: https://openai.com/sora (demo videos)
- Runway: https://runwayml.com/research

## Kaggle DFD Dataset

```bash
# Download from Kaggle (requires account)
kaggle datasets download -d sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
```

## Collection Guidelines

When collecting AI videos, record:
1. Generator (Sora, Runway, deepfake tool, etc.)
2. Manipulation type (face swap, lip sync, full generation)
3. Original source if applicable
4. Duration and resolution
EOF

    echo -e "${GREEN}Video dataset instructions created${NC}"
}

# =============================================================================
# MAIN
# =============================================================================

# Export DATASETS_DIR for Python scripts
export DATASETS_DIR

# Run all download functions
download_text_datasets
download_image_datasets
download_audio_datasets
download_video_datasets

# Create summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Download Summary${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}Directory structure created at: $DATASETS_DIR${NC}"
echo ""
echo "Dataset status:"
echo "  Text:   $(ls -1 "$DATASETS_DIR/text/human" 2>/dev/null | wc -l | tr -d ' ') human files, $(ls -1 "$DATASETS_DIR/text/ai_generated" 2>/dev/null | wc -l | tr -d ' ') AI files"
echo "  Image:  $(ls -1 "$DATASETS_DIR/image/human" 2>/dev/null | wc -l | tr -d ' ') human files, $(ls -1 "$DATASETS_DIR/image/ai_generated" 2>/dev/null | wc -l | tr -d ' ') AI files"
echo "  Audio:  See $DATASETS_DIR/audio/*/DOWNLOAD_INSTRUCTIONS.md"
echo "  Video:  See $DATASETS_DIR/video/*/DOWNLOAD_INSTRUCTIONS.md"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Install Python datasets library: pip install datasets"
echo "2. Re-run this script for full text dataset download"
echo "3. Follow DOWNLOAD_INSTRUCTIONS.md in each directory for full datasets"
echo "4. Run 'make benchmark-all' to evaluate with Manuscript"

echo -e "\n${GREEN}Done!${NC}"
