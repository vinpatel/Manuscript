---
title: Methodology
description: How Manuscript's benchmark evaluation is conducted
---

# Benchmark Methodology

This document describes how Manuscript's detection accuracy is measured.

## Evaluation Criteria

We use standard machine learning metrics:

### Accuracy

The percentage of correctly classified samples:

```
Accuracy = (True Positives + True Negatives) / Total Samples
```

### Precision

How many detected AI samples are actually AI:

```
Precision = True Positives / (True Positives + False Positives)
```

### Recall

How many actual AI samples were detected:

```
Recall = True Positives / (True Positives + False Negatives)
```

### F1 Score

Harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

## Testing Protocol

For each content type:

```
1. Load dataset (50 human + 50 AI-generated samples)
2. Run Manuscript detection on each sample
3. Record: prediction, confidence, processing time
4. Calculate aggregate metrics
5. Analyze failure cases
```

## Dataset Composition

### Text (100 samples)

| Category | Count | Source |
|----------|-------|--------|
| Human - Essays | 15 | Verified authors |
| Human - Articles | 15 | NYT/News |
| Human - Creative | 10 | Short stories, poetry |
| Human - Technical | 10 | Documentation |
| AI - GPT-4 | 15 | OpenAI |
| AI - Claude | 15 | Anthropic |
| AI - Gemini | 10 | Google |
| AI - Llama-3 | 10 | Meta |

### Image (100 samples)

| Category | Count | Source |
|----------|-------|--------|
| Human - Photos | 25 | COCO/Unsplash |
| Human - Artwork | 15 | Digital art |
| Human - Screenshots | 10 | UI captures |
| AI - DALL-E 3 | 15 | OpenAI |
| AI - Midjourney v6 | 15 | Midjourney |
| AI - Stable Diffusion 3 | 15 | Stability AI |
| AI - FLUX | 5 | Black Forest |

### Audio (100 samples)

| Category | Count | Source |
|----------|-------|--------|
| Human - Speech | 20 | LibriSpeech |
| Human - Podcast | 15 | Various |
| Human - Music | 15 | CC-licensed |
| AI - ElevenLabs | 20 | TTS synthesis |
| AI - WaveFake | 15 | Vocoders |
| AI - Music AI | 15 | Suno/Udio |

### Video (100 samples)

| Category | Count | Source |
|----------|-------|--------|
| Human - UGC | 25 | YouTube/Vimeo |
| Human - Professional | 15 | Stock footage |
| Human - Mobile | 10 | Smartphone |
| AI - Deepfake | 20 | DeepfakeBench |
| AI - Sora | 10 | OpenAI |
| AI - Runway | 10 | Text-to-video |
| AI - Other | 10 | Various tools |

## Signal Weights

Each detection signal contributes to the final score with different weights:

### Text Detection Weights

| Signal | Weight |
|--------|--------|
| AI Phrase Detection | 0.20 |
| Vocabulary Richness | 0.20 |
| Sentence Variance | 0.15 |
| Contractions Usage | 0.10 |
| Punctuation Variety | 0.10 |
| Burstiness | 0.10 |
| Repetition Penalty | 0.10 |
| Word Length Variance | 0.05 |

### Image Detection Weights

| Signal | Weight |
|--------|--------|
| Metadata Score | 0.25 |
| Color Distribution | 0.20 |
| Edge Consistency | 0.15 |
| Noise Pattern | 0.15 |
| Compression Analysis | 0.15 |
| Symmetry Detection | 0.10 |

### Audio Detection Weights

| Signal | Weight |
|--------|--------|
| Metadata Score | 0.25 |
| Pattern Analysis | 0.20 |
| Format Analysis | 0.15 |
| Quality Indicators | 0.15 |
| AI Signatures | 0.15 |
| Noise Profile | 0.10 |

### Video Detection Weights

| Signal | Weight |
|--------|--------|
| Metadata Score | 0.25 |
| Container Analysis | 0.20 |
| Audio Presence | 0.15 |
| Temporal Pattern | 0.15 |
| Encoding Signature | 0.15 |
| Bitrate Consistency | 0.10 |

## Confidence Calculation

The final confidence score is calculated as:

```
confidence = Σ(signal_value × signal_weight) / Σ(signal_weight)
```

A threshold of 0.5 determines the verdict:
- confidence >= 0.5 → "ai"
- confidence < 0.5 → "human"

## Reproducibility

All benchmark results can be reproduced:

```bash
git clone https://github.com/vinpatel/manuscript
cd manuscript
make download-benchmark-data
make benchmark-all
```

Results are saved to `benchmark/results/` with timestamps.

## Limitations

### Dataset Biases

- Text samples may over-represent certain writing styles
- Image samples are web-sourced (stripped metadata)
- Audio samples are high-quality studio recordings
- Video samples require API downloads

### Detection Limitations

- Short content (<100 words text, <5s audio/video) is less reliable
- Heavily edited content may evade detection
- New AI models may not be in signature databases
- Language support is primarily English

## Continuous Improvement

We regularly update:

1. **Signature databases** with new AI tool markers
2. **Training datasets** with fresh samples
3. **Detection algorithms** based on failure analysis
4. **Benchmark reports** with each release
