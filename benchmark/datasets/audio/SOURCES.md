# Manuscript Audio Dataset Sources

This document describes the sources used for the audio benchmark dataset and provides guidance for obtaining additional samples.

## Current Dataset Composition

### Human Audio (50 samples)

**Source: LibriSpeech**
- URL: https://www.openslr.org/12/
- License: Public Domain
- Description: LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, derived from audiobooks from the LibriVox project.
- Subset used: dev-clean (337MB compressed)
- Format: FLAC, 16kHz, 16-bit, mono

**Download Instructions:**
```bash
# Download dev-clean subset (smallest, ~337MB)
curl -O http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz

# For larger datasets:
# - train-clean-100.tar.gz (6.3GB)
# - train-clean-360.tar.gz (23GB)
# - test-clean.tar.gz (346MB)
```

### AI-Generated Audio (50 samples)

**Source: Hemg Deepfake Audio Dataset**
- URL: https://huggingface.co/datasets/Hemg/Deepfake-Audio-Dataset
- License: Research use
- Description: Dataset containing both real and AI-generated (deepfake) audio samples for detection research
- Total samples: 100 (50 fake, 50 real)
- Format: WAV, 44.1kHz, 16-bit
- Duration: ~10 seconds per sample

**Download Instructions:**
```python
from datasets import load_dataset
dataset = load_dataset('Hemg/Deepfake-Audio-Dataset')
```

Or download parquet directly:
```bash
curl -L "https://huggingface.co/datasets/Hemg/Deepfake-Audio-Dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet" -o dataset.parquet
```

---

## Additional AI Audio Sources

For expanding the AI-generated audio collection, consider these sources:

### 1. WaveFake Dataset
- URL: https://zenodo.org/records/5642694
- GitHub: https://github.com/RUB-SysSec/WaveFake
- Size: 28.9 GB
- Description: 104,885 generated audio clips from 6 different TTS architectures
- TTS Models included:
  - MelGAN
  - Full-Band MelGAN
  - Multi-Band MelGAN
  - HiFi-GAN
  - Parallel WaveGAN
  - WaveGlow
- Languages: English (LJSpeech), Japanese (JSUT)
- License: Research use

**Download:**
```bash
# Full dataset (28.9GB)
curl -L "https://zenodo.org/records/5642694/files/generated_audio.zip?download=1" -o wavefake.zip
```

### 2. ASVspoof 2021 Deepfake Track
- URL: https://zenodo.org/record/4835108
- Official site: https://www.asvspoof.org/index2021.html
- Size: ~34.5 GB
- Description: Evaluation set for deepfake speech detection
- Contains both bonafide (real) and spoofed speech

**Download:**
```bash
# Part 00 (8.6GB)
curl -L "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1" -o asvspoof_part00.tar.gz
```

### 3. Commercial TTS Services (Requires API Access)

These services can generate high-quality AI speech but require registration and may have usage costs:

#### ElevenLabs
- URL: https://elevenlabs.io/
- API: https://docs.elevenlabs.io/api-reference
- Features: Voice cloning, multiple voices, emotion control
- Free tier: Limited characters per month

#### OpenAI TTS
- URL: https://platform.openai.com/docs/guides/text-to-speech
- Models: tts-1, tts-1-hd
- Voices: alloy, echo, fable, onyx, nova, shimmer

#### Amazon Polly
- URL: https://aws.amazon.com/polly/
- Features: Multiple languages, neural voices
- Requires AWS account

### 4. AI Music Generation Services

For AI-generated music samples:

#### Suno
- URL: https://suno.com/
- Description: AI music generation from text prompts
- Note: Check terms of service for research use

#### Udio
- URL: https://udio.com/
- Description: AI music creation platform
- Note: Verify licensing for benchmark use

### 5. Open-Source TTS Models

Generate your own samples using:

#### Coqui TTS
- GitHub: https://github.com/coqui-ai/TTS
- Models: Tacotron2, VITS, YourTTS, etc.
- Installation: `pip install TTS`

#### StyleTTS
- GitHub: https://github.com/yl4579/StyleTTS2
- Demo: https://styletts2.github.io/
- Features: High-quality, expressive speech synthesis

#### Bark
- GitHub: https://github.com/suno-ai/bark
- Features: Multilingual, sound effects, music

---

## Dataset Guidelines

When adding new samples, ensure:

1. **Duration**: 5-60 seconds
2. **File size**: Under 10MB
3. **Format**: MP3, WAV, or FLAC
4. **Sample rate**: 16kHz minimum
5. **Content types**: Speech, music, or ambient sounds
6. **Licensing**: Verify rights for research/benchmark use

Update `metadata.json` with:
- source_url
- generator (model name for AI)
- duration_seconds
- format
- content_type
- file_name

---

## License Information

- **LibriSpeech**: Public Domain (derived from LibriVox)
- **Hemg Dataset**: Research use
- **WaveFake**: Research use (CC-BY license)
- **ASVspoof**: Open Data Commons Attribution License

Always verify licensing terms before using data for commercial purposes.
