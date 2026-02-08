---
title: Introduction
description: Learn about Manuscript, the open source AI content detector
---

# What is Manuscript?

**Manuscript** is an open-source, privacy-first AI content detector that runs entirely on your infrastructure. It can detect AI-generated content across four modalities:

- **Text** - Detect ChatGPT, Claude, Gemini, and other LLM-generated content
- **Images** - Identify DALL-E, Midjourney, Stable Diffusion generated images
- **Audio** - Spot ElevenLabs, Suno, and other AI voice/music generators
- **Video** - Detect deepfakes and AI-generated video content

## Why Manuscript?

Every other AI detection service requires you to **upload your content to their servers**. That's a dealbreaker for many organizations:

| Industry | Why On-Premise Matters |
|----------|----------------------|
| Healthcare | HIPAA compliance prohibits external data sharing |
| Legal | Attorney-client privilege can't survive third-party uploads |
| Finance | SOC2/PCI requirements restrict data sharing |
| Government | Air-gapped networks, classified environments |
| Education | Scale without per-seat licensing costs |

**Manuscript runs entirely on YOUR infrastructure. Your data never leaves your network.**

## How It Works

Unlike ML-based detectors that require GPUs and large models, Manuscript uses **statistical and forensic analysis**:

### Text Detection

Analyzes linguistic patterns that differentiate human and AI writing:
- Sentence length variance (humans vary more)
- Vocabulary richness and rare word usage
- Contraction patterns ("don't" vs "do not")
- Known AI phrases ("As an AI...", "It's important to note...")
- Hedging language and repetition patterns

### Image Detection

Examines forensic signals in image files:
- EXIF metadata presence and validity
- Camera make/model signatures
- Sensor noise characteristics
- Compression artifact patterns
- Color distribution analysis

### Audio Detection

Analyzes audio container and signal characteristics:
- File header and container metadata
- Encoding parameters and profiles
- FFT spectral analysis
- MFCC computation
- AI tool fingerprints (ElevenLabs, Suno markers)

### Video Detection

Combines multiple forensic approaches:
- Container metadata analysis
- Bitrate consistency checking
- Temporal pattern analysis
- Encoding signature detection

## Key Features

- **100% Offline** - No external API calls
- **Self-Hosted** - Deploy on your infrastructure
- **Open Source** - MIT licensed, fully auditable
- **Multi-Modal** - Text, image, audio, video
- **Fast** - Sub-10ms response times
- **Zero Dependencies** - Pure Go implementation

## Next Steps

- [Quick Start Guide](/manuscript/docs/quickstart/) - Get running in 30 seconds
- [Installation](/manuscript/docs/installation/) - Detailed setup instructions
- [Benchmarks](/manuscript/docs/benchmarks/) - See our performance metrics
