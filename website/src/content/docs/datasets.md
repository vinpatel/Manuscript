---
title: Datasets
description: Dataset sources used for Manuscript benchmarks
---

# Benchmark Datasets

All datasets used in Manuscript benchmarks are publicly available.

## Text Datasets

### HC3 (Human ChatGPT Comparison)

- **Source:** [HuggingFace](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- **License:** Apache 2.0
- **Size:** 37,000+ QA pairs
- **Content:** Human vs ChatGPT responses to questions

### Defactify-Text

- **Source:** [arXiv:2510.22874](https://arxiv.org/abs/2510.22874)
- **License:** Research
- **Size:** 58,000+ articles
- **Content:** NYT articles + LLM-generated versions (GPT-4, Gemma, Mistral)

### HATC-2025

- **Source:** [Hastewire](https://hastewire.com/blog/ai-text-detection-benchmarks-2025-top-performance-metrics)
- **License:** Research
- **Size:** 50,000+ samples
- **Content:** Human vs AI passages benchmark

### LLMSciTxt

- **Source:** [arXiv:2507.05157](https://arxiv.org/html/2507.05157v1)
- **License:** Research
- **Size:** 10,000+ papers
- **Content:** Scientific papers: human vs ChatGPT/Gemini/Llama-3

### Beemo Benchmark

- **Source:** [Toloka](https://toloka.ai/ai-detection-benchmark)
- **License:** Research
- **Size:** Varied
- **Content:** Human, machine-generated, and edited content

## Image Datasets

### MS COCOAI

- **Source:** [arXiv:2601.00553](https://arxiv.org/abs/2601.00553)
- **License:** Research
- **Size:** 96,000 pairs
- **Content:** MS COCO + SD3, SDXL, DALL-E 3, Midjourney v6

### GenImage

- **Source:** [GitHub](https://github.com/GenImage-Dataset/GenImage)
- **License:** Research
- **Size:** 1M+ images
- **Content:** Midjourney, Stable Diffusion, ADM, GLIDE, etc.

### AIGIBench

- **Source:** [arXiv:2505.12335](https://arxiv.org/html/2505.12335v1)
- **License:** Research
- **Size:** 6,000+ samples
- **Content:** SD-XL, SD-3, DALL-E 3, Midjourney v6, FLUX, Imagen-3

### Human Image Sources

- **Unsplash:** Free high-resolution photos
- **Pexels:** Free stock photos
- **COCO:** Common Objects in Context dataset

## Audio Datasets

### WaveFake

- **Source:** [Zenodo](https://zenodo.org/record/5642694)
- **License:** CC BY 4.0
- **Size:** 117,985 samples
- **Content:** 7 vocoder architectures

### LibriSpeech

- **Source:** [OpenSLR](https://www.openslr.org/12)
- **License:** CC BY 4.0
- **Size:** 1000+ hours
- **Content:** Clean speech from audiobooks

### LJSpeech

- **Source:** [Keith Ito](https://keithito.com/LJ-Speech-Dataset/)
- **License:** Public Domain
- **Size:** 13,100 clips
- **Content:** Single female speaker recordings

### ASVspoof

- **Source:** [asvspoof.org](https://www.asvspoof.org/)
- **License:** Research
- **Size:** 180,000+ samples
- **Content:** Spoofing and deepfake detection

### TIMIT-ElevenLabs

- **Source:** [arXiv:2307.07683](https://arxiv.org/pdf/2307.07683)
- **License:** Research
- **Size:** Varied
- **Content:** Real vs ElevenLabs cloned voices

## Video Datasets

### Deepfake-Eval-2024

- **Source:** [arXiv:2503.02857](https://arxiv.org/html/2503.02857v1)
- **License:** Research
- **Size:** 44+ hours
- **Content:** In-the-wild deepfakes from 2024 (includes Sora)

### DF40/DeepfakeBench

- **Source:** [GitHub](https://github.com/SCLBD/DeepfakeBench)
- **License:** MIT
- **Size:** Large
- **Content:** 40 deepfake techniques

### FaceForensics++

- **Source:** [GitHub](https://github.com/ondyari/FaceForensics)
- **License:** Research
- **Size:** 1.8M+ images
- **Content:** DeepFakes, Face2Face, FaceSwap, NeuralTextures

### Microsoft Deepfake Dataset

- **Source:** [Microsoft](https://www.biometricupdate.com/202507/new-microsoft-benchmark-for-evaluating-deepfake-detection-prioritizes-breadth)
- **License:** Research
- **Size:** 50,000+ samples
- **Content:** Real-world deepfakes and synthetic media

### Kaggle DFD

- **Source:** [Kaggle](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)
- **License:** Research
- **Size:** 10,000+ samples
- **Content:** Original deepfake detection dataset

## Downloading Datasets

### Automatic Download

```bash
make download-benchmark-data
```

### Manual Download

See `benchmark/DATASET_SOURCES.md` for detailed instructions on accessing each dataset.

## License Compliance

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| HC3 | Apache 2.0 | Yes |
| WaveFake | CC BY 4.0 | Yes (with attribution) |
| LibriSpeech | CC BY 4.0 | Yes (with attribution) |
| LJSpeech | Public Domain | Yes |
| DeepfakeBench | MIT | Yes |
| Others | Research | Academic only |

## Contributing Datasets

We welcome contributions of:
- Labeled human vs AI content
- New AI generator samples
- Edge case examples
- Multi-language content

Submit via [GitHub Issues](https://github.com/vinpatel/manuscript/issues/new?template=accuracy_report.md).
