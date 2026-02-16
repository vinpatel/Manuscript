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

## Detailed Results by Content Type

For detailed results, confusion matrices, analysis, and industry comparisons for each content type, see:

- [Text Detection](/manuscript/docs/text-detection/) - 90% accuracy, 100% precision, zero false positives
- [Image Detection](/manuscript/docs/image-detection/) - 100% recall, known false positive issue with web-sourced images
- [Audio Detection](/manuscript/docs/audio-detection/) - Significant v0.2.0 improvements via FFT and MFCC analysis
- [Video Detection](/manuscript/docs/video-detection/) - Benchmark pending, requires video file downloads

## Dataset Sources

For full dataset details, licensing, and download instructions, see the [Datasets](/manuscript/docs/datasets/) page.

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
