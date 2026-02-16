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

Each content type is tested with 100 samples (50 human + 50 AI-generated), drawn from diverse sources and generators. For full dataset details and download instructions, see the [Datasets](/manuscript/docs/datasets/) page.

## Signal Weights

Each detection signal contributes to the final score with different weights. For the complete weight breakdown per content type, see the individual detection pages:

- [Text Detection](/manuscript/docs/text-detection/) - 8 signals including AI phrase detection, vocabulary richness, sentence variance
- [Image Detection](/manuscript/docs/image-detection/) - 6 signals including metadata, color distribution, noise pattern
- [Audio Detection](/manuscript/docs/audio-detection/) - 6 signals including metadata, spectral analysis, MFCC
- [Video Detection](/manuscript/docs/video-detection/) - 6 signals including metadata, container analysis, temporal patterns

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

For detection-specific limitations, see each content type's dedicated page.

## Continuous Improvement

We regularly update:

1. **Signature databases** with new AI tool markers
2. **Training datasets** with fresh samples
3. **Detection algorithms** based on failure analysis
4. **Benchmark reports** with each release
