---
title: Benchmark Overview
description: Comprehensive evaluation of Manuscript's AI detection accuracy
---

# Benchmark Overview

This page summarizes Manuscript's detection performance across all content types.

## Executive Summary

| Content Type | Dataset Size | Accuracy | Precision | Recall | F1 Score |
|-------------|-------------|----------|-----------|--------|----------|
| **Text** | 100 | **90.00%** | 100.00% | 80.00% | 88.89% |
| **Image** | 100 | 50.00% | 50.00% | 100.00% | 66.67% |
| **Audio** | 100 | **46.00%** | 47.92% | 92.00% | 63.01% |
| **Video** | 100 | *Pending* | - | - | - |

*Benchmark run: January 2026 with Manuscript v0.2.0*

## Version Improvements

### v0.1.0 → v0.2.0

| Content Type | Metric | v0.1.0 | v0.2.0 | Change |
|-------------|--------|--------|--------|--------|
| **Audio** | Accuracy | 38.00% | 46.00% | +8.00% |
| | Recall | 76.00% | 92.00% | +16.00% |
| | F1 Score | 55.07% | 63.01% | +7.94% |
| Text | Accuracy | 90.00% | 90.00% | No change |
| Image | Accuracy | 50.00% | 50.00% | No change |

The v0.2.0 enhancements (FFT spectral analysis, MFCC computation, temporal consistency) significantly improved audio detection.

## Text Detection

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 90.00% |
| **Precision** | 100.00% |
| **Recall** | 80.00% |
| **F1 Score** | 88.89% |

### Confusion Matrix

|  | Predicted Human | Predicted AI |
|--|-----------------|--------------|
| **Actual Human** | 50 | 0 |
| **Actual AI** | 10 | 40 |

### Analysis

- Text detection shows excellent precision (100%) - zero false positives
- 80% recall indicates 10 AI samples were misclassified as human
- 90% accuracy exceeds the baseline of 58-65% from academic benchmarks
- Primary challenges: Some AI content is indistinguishable from human

## Image Detection

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 50.00% |
| **Precision** | 50.00% |
| **Recall** | 100.00% |
| **F1 Score** | 66.67% |

### Confusion Matrix

|  | Predicted Human | Predicted AI |
|--|-----------------|--------------|
| **Actual Human** | 0 | 50 |
| **Actual AI** | 0 | 50 |

### Analysis

- Image detection correctly identified all 50 AI-generated images (100% recall)
- However, all 50 human images were also flagged as AI (0% specificity)
- **Root cause:** Downloaded images have EXIF metadata stripped
- The detector needs tuning to reduce false positives on web-sourced images

### Industry Comparison

| Detector | Accuracy |
|----------|----------|
| Hive Moderation | 98-99.9% |
| AI or Not | 88.89% |
| **Manuscript** | 50.00% |

## Audio Detection

### Results

| Metric | v0.1.0 | v0.2.0 |
|--------|--------|--------|
| **Accuracy** | 38.00% | **46.00%** |
| **Precision** | 43.18% | 47.92% |
| **Recall** | 76.00% | **92.00%** |
| **F1 Score** | 55.07% | **63.01%** |

### Confusion Matrix (v0.2.0)

|  | Predicted Human | Predicted AI |
|--|-----------------|--------------|
| **Actual Human** | 0 | 50 |
| **Actual AI** | 4 | 46 |

### Analysis

- v0.2.0 improvements increased AI audio detection significantly (+16% recall)
- FFT spectral analysis and MFCC computation catch 8 more AI samples
- False positive rate unchanged - clean human audio still triggers false positives
- **Root cause:** LibriSpeech audiobook recordings are "too clean" and resemble synthesized audio

### Industry Comparison

| Detector | Accuracy |
|----------|----------|
| ElevenLabs Classifier | >99% (unlaundered) |
| ElevenLabs Classifier | >90% (laundered) |
| **Manuscript v0.2.0** | 46.00% |

## Video Detection

*Video benchmark is pending - requires video file downloads via API keys.*

### Expected Challenges

Based on industry benchmarks:
- Off-the-shelf detectors show 21.3% lower accuracy on Sora-like videos
- Target accuracy: >75%
- Primary challenges: New diffusion video models, compression

## Dataset Sources

### Text

| Source | Samples | Description |
|--------|---------|-------------|
| HC3 | 37,000+ | Human vs ChatGPT responses |
| Defactify-Text | 58,000+ | Articles + LLM versions |
| HATC-2025 | 50,000+ | Benchmark samples |

### Image

| Source | Samples | Description |
|--------|---------|-------------|
| MS COCOAI | 96,000 | Real + SD3/DALL-E/Midjourney |
| GenImage | 1M+ | Multi-generator dataset |
| AIGIBench | 6,000+ | Latest generators |

### Audio

| Source | Samples | Description |
|--------|---------|-------------|
| WaveFake | 117,985 | 7 vocoder architectures |
| LibriSpeech | 1000+ hrs | Real audiobook speech |
| ASVspoof | 180,000+ | Spoofing detection |

### Video

| Source | Samples | Description |
|--------|---------|-------------|
| Deepfake-Eval-2024 | 44+ hrs | In-the-wild deepfakes (includes Sora) |
| DeepfakeBench | Large | 40 deepfake techniques |
| FaceForensics++ | 1.8M+ | Face manipulation |

## Unique Value Proposition

Despite accuracy differences from commercial solutions:

1. **Privacy-First:** No data leaves your infrastructure
2. **Multi-Modal:** Single tool for all content types
3. **No API Costs:** No per-request charges
4. **Transparent:** Open-source, auditable algorithms
5. **Customizable:** Adjustable detection weights

## Running the Benchmark

```bash
# Clone the repository
git clone https://github.com/vinpatel/manuscript
cd manuscript

# Download benchmark datasets
make download-benchmark-data

# Run the full benchmark suite
make benchmark-all

# Generate report
make benchmark-report
```

## Citation

```bibtex
@misc{manuscript2025benchmark,
  title={Manuscript Benchmark: Multi-Modal AI Content Detection Evaluation},
  author={Manuscript Contributors},
  year={2025},
  url={https://github.com/manuscript/manuscript/benchmark}
}
```
