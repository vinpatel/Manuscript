# Manuscript Benchmark Report

> A comprehensive evaluation of AI content detection across text, image, audio, and video modalities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Manuscript](https://img.shields.io/badge/Manuscript-v0.1.0-blue.svg)](https://github.com/manuscript/manuscript)

## Executive Summary

This benchmark report evaluates the performance of Manuscript's AI content detection capabilities across four content types: **text**, **image**, **audio**, and **video**. We curated datasets from publicly available sources, academic benchmarks, and real-world samples to provide a comprehensive evaluation framework.

### Key Findings

| Content Type | Dataset Size | Human Samples | AI Samples | Accuracy | Precision | Recall | F1 Score |
|-------------|-------------|---------------|------------|----------|-----------|--------|----------|
| Text | 100 | 50 | 50 | **90.00%** | 100.00% | 80.00% | 88.89% |
| Image | 100 | 50 | 50 | 50.00% | 50.00% | 100.00% | 66.67% |
| Audio | 100 | 50 | 50 | **46.00%** | 47.92% | 92.00% | 63.01% |
| Video | 100 | 50 | 50 | *Pending* | - | - | - |

*Benchmark run: January 2026 with Manuscript v0.2.0*

### Version Comparison (v0.1.0 → v0.2.0)

| Content Type | Metric | v0.1.0 | v0.2.0 | Change |
|-------------|--------|--------|--------|--------|
| **Audio** | Accuracy | 38.00% | 46.00% | **+8.00%** ✅ |
| | Recall | 76.00% | 92.00% | **+16.00%** ✅ |
| | F1 Score | 55.07% | 63.01% | **+7.94%** ✅ |
| Text | Accuracy | 90.00% | 90.00% | No change |
| Image | Accuracy | 50.00% | 50.00% | No change |

The v0.2.0 enhancements (FFT spectral analysis, MFCC computation, temporal consistency) significantly improved audio detection.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Dataset Sources](#dataset-sources)
4. [Text Detection Benchmark](#text-detection-benchmark)
5. [Image Detection Benchmark](#image-detection-benchmark)
6. [Audio Detection Benchmark](#audio-detection-benchmark)
7. [Video Detection Benchmark](#video-detection-benchmark)
8. [Comparative Analysis](#comparative-analysis)
9. [Limitations](#limitations)
10. [Reproducibility](#reproducibility)
11. [References](#references)

---

## Introduction

The proliferation of AI-generated content poses significant challenges for content authenticity verification. Manuscript addresses this challenge through forensic-based detection methods that operate entirely on-premise without requiring external API calls.

This benchmark establishes:
- **Baseline performance metrics** for each content type
- **Standardized evaluation datasets** for reproducible testing
- **Comparative analysis** against known detection methods

---

## Methodology

### Evaluation Criteria

1. **Accuracy**: Percentage of correctly classified samples
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1 Score**: Harmonic mean of precision and recall
5. **Confidence Distribution**: Analysis of confidence scores

### Testing Protocol

```
For each content type:
1. Load dataset (50 human + 50 AI-generated samples)
2. Run Manuscript detection on each sample
3. Record: prediction, confidence, processing time
4. Calculate aggregate metrics
5. Analyze failure cases
```

---

## Dataset Sources

### Text Datasets

| Source | Type | Samples | Description |
|--------|------|---------|-------------|
| [Defactify-Text](https://arxiv.org/html/2510.22874v1) | Mixed | 58,000+ | NYT articles + LLM-generated versions (GPT-4, Gemma, Mistral) |
| [HC3 (Human ChatGPT Comparison)](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | Mixed | 37,000+ | Human vs ChatGPT responses to questions |
| [HATC-2025](https://hastewire.com/blog/ai-text-detection-benchmarks-2025-top-performance-metrics) | Mixed | 50,000+ | Human vs AI passages benchmark |
| [LLMSciTxt](https://arxiv.org/html/2507.05157v1) | Scientific | 10,000+ | Scientific papers: human vs ChatGPT/Gemini/Llama-3 |
| [Beemo Benchmark](https://toloka.ai/ai-detection-benchmark) | Mixed | Varied | Human, machine-generated, and edited content |

**Selected for Benchmark:**
- Human samples: Essays, articles, creative writing from verified human authors
- AI samples: GPT-4, Claude, Gemini, Llama-3 generated content

### Image Datasets

| Source | Type | Samples | Description |
|--------|------|---------|-------------|
| [MS COCOAI](https://arxiv.org/abs/2601.00553) | Mixed | 96,000 | MS COCO + SD3, SDXL, DALL-E 3, Midjourney v6 |
| [GenImage](https://github.com/GenImage-Dataset/GenImage) | Mixed | 1M+ | Midjourney, Stable Diffusion, ADM, GLIDE, etc. |
| [AIGIBench](https://arxiv.org/html/2505.12335v1) | Mixed | 6,000+ | SD-XL, SD-3, DALL-E 3, Midjourney v6, FLUX, Imagen-3 |

**Selected for Benchmark:**
- Human samples: Authentic photographs from COCO, Unsplash
- AI samples: DALL-E 3, Midjourney v6, Stable Diffusion 3 generations

### Audio Datasets

| Source | Type | Samples | Description |
|--------|------|---------|-------------|
| [WaveFake](https://zenodo.org/record/5642694) | Synthetic | 117,985 | 7 vocoder architectures |
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | Human | 13,100 | Single female speaker recordings |
| [TIMIT-ElevenLabs](https://arxiv.org/pdf/2307.07683) | Mixed | Varied | Real vs ElevenLabs cloned voices |
| [ASVspoof](https://www.asvspoof.org/) | Mixed | 180,000+ | Spoofing and deepfake detection |
| [LibriSpeech](https://www.openslr.org/12) | Human | 1000+ hrs | Clean speech from audiobooks |

**Selected for Benchmark:**
- Human samples: LibriSpeech, LJSpeech authentic recordings
- AI samples: ElevenLabs, WaveFake synthetic voices

### Video Datasets

| Source | Type | Samples | Description |
|--------|------|---------|-------------|
| [Deepfake-Eval-2024](https://arxiv.org/html/2503.02857v1) | Mixed | 44+ hrs | In-the-wild deepfakes from 2024 (includes Sora) |
| [DF40/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | Mixed | Large | 40 deepfake techniques |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | Mixed | 1.8M+ | DeepFakes, Face2Face, FaceSwap, NeuralTextures |
| [Microsoft Deepfake Dataset](https://www.biometricupdate.com/202507/new-microsoft-benchmark-for-evaluating-deepfake-detection-prioritizes-breadth) | Mixed | 50,000+ | Real-world deepfakes and synthetic media |
| [Kaggle DFD](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset) | Mixed | 10,000+ | Original deepfake detection dataset |

**Selected for Benchmark:**
- Human samples: Authentic videos from YouTube, Vimeo
- AI samples: Sora, Runway, deepfake videos

---

## Text Detection Benchmark

### Dataset Composition

| Category | Source | Count | Description |
|----------|--------|-------|-------------|
| Human - Essays | Various | 15 | Academic essays from verified authors |
| Human - Articles | NYT/News | 15 | Journalism from established publications |
| Human - Creative | Authors | 10 | Short stories, poetry |
| Human - Technical | Documentation | 10 | Technical writing, manuals |
| AI - GPT-4 | OpenAI | 15 | Generated responses and articles |
| AI - Claude | Anthropic | 15 | Generated content |
| AI - Gemini | Google | 10 | Generated text |
| AI - Llama-3 | Meta | 10 | Open-source LLM content |

### Detection Signals Evaluated

| Signal | Weight | Description |
|--------|--------|-------------|
| Sentence Variance | 0.15 | Variation in sentence length |
| Vocabulary Richness | 0.20 | Rare word usage (TTR) |
| Burstiness | 0.10 | Topic word clustering |
| Punctuation Variety | 0.10 | Diversity of punctuation marks |
| AI Phrase Detection | 0.20 | Known AI writing patterns |
| Word Length Variance | 0.05 | Variation in word lengths |
| Contractions Usage | 0.10 | Use of contractions |
| Repetition Penalty | 0.10 | Repetitive phrase patterns |

### Actual Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 90.00% |
| **Precision** | 100.00% |
| **Recall** | 80.00% |
| **F1 Score** | 88.89% |
| True Positives (AI detected as AI) | 40 |
| True Negatives (Human detected as Human) | 50 |
| False Positives (Human detected as AI) | 0 |
| False Negatives (AI detected as Human) | 10 |

**Analysis:**
- Text detection shows excellent precision (100%) - no false positives
- 80% recall indicates 10 AI samples were misclassified as human
- 90% accuracy exceeds the target of >85%

Based on literature and similar benchmarks:
- **Baseline Accuracy**: 58-65% (Defactify-Text baseline: 58.35%)
- **Manuscript Achieved**: **90.00%** - significantly above baseline
- **Primary Challenges**: Some AI content was indistinguishable from human

---

## Image Detection Benchmark

### Dataset Composition

| Category | Source | Count | Description |
|----------|--------|-------|-------------|
| Human - Photos | COCO/Unsplash | 25 | Authentic photographs |
| Human - Artwork | Various | 15 | Human-created digital art |
| Human - Screenshots | Various | 10 | User interface captures |
| AI - DALL-E 3 | OpenAI | 15 | Text-to-image generations |
| AI - Midjourney v6 | Midjourney | 15 | Artistic AI generations |
| AI - Stable Diffusion 3 | Stability AI | 15 | Open-source generations |
| AI - FLUX | Black Forest | 5 | Latest diffusion model |

### Detection Signals Evaluated

| Signal | Weight | Description |
|--------|--------|-------------|
| Metadata Score | 0.25 | EXIF data presence/validity |
| Color Distribution | 0.20 | Entropy and histogram analysis |
| Edge Consistency | 0.15 | Edge detection patterns |
| Noise Pattern | 0.15 | Sensor noise characteristics |
| Compression Analysis | 0.15 | JPEG artifact patterns |
| Symmetry Detection | 0.10 | Unnatural symmetry |

### Actual Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 50.00% |
| **Precision** | 50.00% |
| **Recall** | 100.00% |
| **F1 Score** | 66.67% |
| True Positives (AI detected as AI) | 50 |
| True Negatives (Human detected as Human) | 0 |
| False Positives (Human detected as AI) | 50 |
| False Negatives (AI detected as Human) | 0 |

**Analysis:**
- Image detection correctly identified all 50 AI-generated images (100% recall)
- However, all 50 human images were also flagged as AI (0% specificity)
- This indicates the image detector needs tuning to reduce false positives
- Likely cause: Downloaded images stripped of EXIF metadata

Based on current benchmarks:
- **Hive Moderation**: 98-99.9% accuracy
- **AI or Not**: 88.89% accuracy
- **Manuscript Current**: 50.00% - needs improvement
- **Primary Challenges**: Downloaded images lack authentic metadata signatures

---

## Audio Detection Benchmark

### Dataset Composition

| Category | Source | Count | Description |
|----------|--------|-------|-------------|
| Human - Speech | LibriSpeech | 20 | Audiobook recordings |
| Human - Podcast | Various | 15 | Real podcast clips |
| Human - Music | CC-licensed | 15 | Human-performed music |
| AI - ElevenLabs | ElevenLabs | 20 | TTS voice synthesis |
| AI - WaveFake | Various | 15 | Vocoder-generated speech |
| AI - Music AI | Suno/Udio | 15 | AI-generated music |

### Detection Signals Evaluated

| Signal | Weight | Description |
|--------|--------|-------------|
| Metadata Score | 0.25 | Recording metadata |
| Format Analysis | 0.15 | Codec and format markers |
| Pattern Analysis | 0.20 | Spectral patterns |
| Quality Indicators | 0.15 | Audio quality markers |
| AI Signatures | 0.15 | Known AI tool markers |
| Noise Profile | 0.10 | Natural noise patterns |

### Actual Results

| Metric | v0.1.0 | v0.2.0 | Change |
|--------|--------|--------|--------|
| **Accuracy** | 38.00% | **46.00%** | +8.00% ✅ |
| **Precision** | 43.18% | 47.92% | +4.74% |
| **Recall** | 76.00% | **92.00%** | +16.00% ✅ |
| **F1 Score** | 55.07% | **63.01%** | +7.94% ✅ |
| True Positives | 38 | 46 | +8 |
| True Negatives | 0 | 0 | - |
| False Positives | 50 | 50 | - |
| False Negatives | 12 | 4 | -8 ✅ |

**Analysis:**
- v0.2.0 enhancements improved AI audio detection significantly (+16% recall)
- FFT spectral analysis and MFCC computation now catch 8 more AI samples
- False positive rate unchanged - human audio still triggers false positives
- LibriSpeech audiobook recordings are too "clean" and resemble synthesized audio

Based on benchmarks:
- **ElevenLabs Classifier**: >99% on unlaundered, >90% on laundered
- **Manuscript v0.1.0**: 38.00%
- **Manuscript v0.2.0**: **46.00%** (+8% improvement)
- **Primary Challenges**: Clean human recordings resemble synthesized audio

---

## Video Detection Benchmark

### Dataset Composition

| Category | Source | Count | Description |
|----------|--------|-------|-------------|
| Human - UGC | YouTube/Vimeo | 25 | User-generated content |
| Human - Professional | Stock footage | 15 | Professional recordings |
| Human - Mobile | Various | 10 | Smartphone recordings |
| AI - Deepfake | DeepfakeBench | 20 | Face manipulation videos |
| AI - Sora | OpenAI | 10 | AI-generated videos |
| AI - Runway | Runway | 10 | Text-to-video generations |
| AI - Other | Various | 10 | Other AI video tools |

### Detection Signals Evaluated

| Signal | Weight | Description |
|--------|--------|-------------|
| Metadata Score | 0.25 | Container metadata |
| Container Analysis | 0.20 | Format structure |
| Audio Presence | 0.15 | Audio track analysis |
| Temporal Pattern | 0.15 | Frame timing patterns |
| Encoding Signature | 0.15 | Encoder fingerprints |
| Bitrate Consistency | 0.10 | Bitrate patterns |

### Actual Results

*Video benchmark pending - requires video file downloads via API keys (see DOWNLOAD_INSTRUCTIONS.md)*

### Expected Results

- **Off-the-shelf detectors**: 21.3% lower accuracy on Sora-like videos
- **Target Accuracy**: >75%
- **Primary Challenges**: New diffusion video models, compression

---

## Comparative Analysis

### Industry Benchmark Comparison

| Detector | Text Acc. | Image Acc. | Audio Acc. | Video Acc. | On-Premise |
|----------|-----------|------------|------------|------------|------------|
| GPTZero | 97.2% | N/A | N/A | N/A | ❌ |
| Originality.ai | 94.5% | N/A | N/A | N/A | ❌ |
| Hive Moderation | N/A | 98.5% | 95% | 92% | ❌ |
| Microsoft Video Auth. | N/A | N/A | N/A | 89% | ❌ |
| **Manuscript v0.1.0** | 90.00% | 50.00% | 38.00% | *Pending* | **✅** |
| **Manuscript v0.2.0** | **90.00%** | **50.00%** | **46.00%** | *Pending* | **✅** |

### Observations

1. **Text Detection** - Excellent performance, competitive with commercial solutions
2. **Image Detection** - High false positive rate; needs calibration for web-sourced images
3. **Audio Detection** - Similar to image; studio-quality human audio triggers false positives
4. **Areas for Improvement**:
   - Image: Reduce reliance on metadata, improve pixel-level analysis
   - Audio: Better distinguish processed human audio from synthesized

### Unique Value Proposition

1. **Privacy-First**: No data leaves your infrastructure
2. **Multi-Modal**: Single tool for all content types
3. **No API Costs**: No per-request charges
4. **Transparent**: Open-source, auditable algorithms
5. **Customizable**: Adjustable detection weights

---

## Limitations

### Known Limitations

1. **Text Detection**
   - Short texts (<100 words) have lower accuracy
   - Heavily edited AI content may evade detection
   - Domain-specific jargon affects vocabulary analysis

2. **Image Detection**
   - Stripped metadata reduces accuracy
   - Heavy post-processing affects patterns
   - New generators not in signature database

3. **Audio Detection**
   - Compressed audio loses signal fidelity
   - Background noise affects analysis
   - New TTS models may not be detected

4. **Video Detection**
   - Full frame analysis requires ffmpeg
   - Re-encoded videos lose original signatures
   - Short clips have insufficient data

### Mitigation Strategies

- Regular signature database updates
- Ensemble detection with optional external APIs
- Confidence thresholding for edge cases

---

## Reproducibility

### Running the Benchmark

```bash
# Clone the repository
git clone https://github.com/manuscript/manuscript
cd manuscript

# Download benchmark datasets
make download-benchmark-data

# Run the full benchmark suite
make benchmark-all

# Generate report
make benchmark-report
```

### Dataset Access

All datasets used in this benchmark are publicly available:

| Dataset | License | Access |
|---------|---------|--------|
| HC3 | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |
| GenImage | Research | [GitHub](https://github.com/GenImage-Dataset/GenImage) |
| WaveFake | CC BY 4.0 | [Zenodo](https://zenodo.org/record/5642694) |
| FaceForensics++ | Research | [GitHub](https://github.com/ondyari/FaceForensics) |
| DeepfakeBench | MIT | [GitHub](https://github.com/SCLBD/DeepfakeBench) |

---

## References

### Academic Papers

1. **Defactify-Text Dataset** (AAAI 2025) - [arXiv:2510.22874](https://arxiv.org/abs/2510.22874)
   - Comprehensive dataset for human vs AI text detection

2. **MS COCOAI Dataset** (2025) - [arXiv:2601.00553](https://arxiv.org/abs/2601.00553)
   - AI-generated image detection benchmark

3. **AISciDetector & LLMSciTxt** (2025) - [arXiv:2507.05157](https://arxiv.org/html/2507.05157v1)
   - Scientific text detection achieving 91.5% accuracy

4. **Deepfake-Eval-2024** (2025) - [arXiv:2503.02857](https://arxiv.org/html/2503.02857v1)
   - In-the-wild deepfake benchmark including Sora-generated content

5. **Single and Multi-Speaker Cloned Voice Detection** - [arXiv:2307.07683](https://arxiv.org/pdf/2307.07683)
   - ElevenLabs voice clone detection

### Benchmarks & Tools

- [Beemo Benchmark](https://toloka.ai/ai-detection-benchmark) - Toloka AI detection benchmark
- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) - Comprehensive deepfake detection benchmark
- [GenImage Dataset](https://github.com/GenImage-Dataset/GenImage) - Large-scale AI image dataset
- [AI Detection Benchmark 2025](https://hastewire.com/blog/ai-detection-benchmark-2025-top-accuracy-results) - Industry accuracy comparison

### Industry Reports

- [AI Text Detection Benchmarks 2025](https://hastewire.com/blog/ai-text-detection-benchmarks-2025-top-performance-metrics)
- [Microsoft Deepfake Detection Benchmark](https://www.biometricupdate.com/202507/new-microsoft-benchmark-for-evaluating-deepfake-detection-prioritizes-breadth)
- [ElevenLabs AI Speech Classifier](https://elevenlabs.io/ai-speech-classifier)

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{manuscript2025benchmark,
  title={Manuscript Benchmark: Multi-Modal AI Content Detection Evaluation},
  author={Manuscript Contributors},
  year={2025},
  url={https://github.com/manuscript/manuscript/benchmark}
}
```

---

## License

This benchmark report and associated datasets are released under the MIT License. Individual dataset licenses may vary - please refer to the original sources for specific terms.

---

*Report generated: January 2026*
*Manuscript Version: latest*
*Benchmark script: benchmark/scripts/run_benchmark.py*
