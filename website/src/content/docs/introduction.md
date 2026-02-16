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

### [Text Detection](/manuscript/docs/text-detection/)

Analyzes linguistic patterns that differentiate human and AI writing, including sentence variance, vocabulary richness, contraction usage, and known AI phrase detection.

### [Image Detection](/manuscript/docs/image-detection/)

Examines forensic signals in image files such as EXIF metadata, sensor noise, compression artifacts, and color distributions.

### [Audio Detection](/manuscript/docs/audio-detection/)

Analyzes audio container metadata, FFT spectral patterns, MFCC coefficients, and AI tool fingerprints.

### [Video Detection](/manuscript/docs/video-detection/)

Combines container metadata analysis, temporal pattern detection, encoding signatures, and audio track verification.

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
