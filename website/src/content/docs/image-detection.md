---
title: Image Detection
description: How Manuscript detects AI-generated images
---

# Image Detection

Manuscript analyzes forensic signals in image files to detect AI-generated content.

## Detection Signals

### EXIF Metadata Analysis

Real photographs contain EXIF data from the camera:

| Metadata | Real Photo | AI-Generated |
|----------|-----------|--------------|
| Camera make | Apple, Canon, Sony | Missing or generic |
| Camera model | iPhone 15, EOS R5 | Missing |
| GPS coordinates | Present (often) | Never |
| Date/time | Consistent | Missing or fake |
| Exposure settings | Physical values | Missing |

**Weight:** 0.25

### Sensor Noise Pattern

Real cameras produce characteristic noise patterns. AI images lack authentic sensor noise.

**Weight:** 0.15

### Color Distribution

Human photographs have natural color histograms. AI often produces artificially smooth distributions.

**Weight:** 0.20

### Compression Artifacts

JPEG compression leaves specific patterns. AI images may have inconsistent compression artifacts.

**Weight:** 0.15

### Edge Consistency

AI images often have unnatural edges, especially around fine details like hair, fingers, and text.

**Weight:** 0.15

### Symmetry Detection

AI-generated faces often exhibit unnatural symmetry that real faces don't have.

**Weight:** 0.10

## Example Analysis

**Real photograph:**

```
File: vacation_photo.jpg
EXIF: Canon EOS R5, 1/250s, f/4, ISO 400
GPS: 40.7128° N, 74.0060° W
Date: 2024-12-25 14:30:00
```

**Signals detected:**
- Valid EXIF metadata ✓
- Authentic camera signature ✓
- Natural noise pattern ✓
- Consistent compression ✓

**Verdict:** Human (confidence: 0.89)

---

**AI-generated image:**

```
File: portrait.png
EXIF: None
GPS: None
Date: None
```

**Signals detected:**
- No EXIF metadata ✗
- No camera signature ✗
- Too clean (no sensor noise) ✗
- Symmetric face detected ✗

**Verdict:** AI (confidence: 0.85)

## API Usage

Upload images via `POST /verify` using multipart form data or base64-encoded JSON. See the [API Reference](/manuscript/docs/api/endpoints/) for full request/response details.

## Current Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 50.00% | Needs improvement |
| Precision | 50.00% | |
| Recall | 100.00% | All AI images detected |
| F1 Score | 66.67% | |

### Known Issues

The current accuracy is limited because:

1. **Web-sourced images** often have EXIF metadata stripped
2. **Downloaded photos** lose camera signatures
3. **Social media compression** removes forensic markers

This is a known limitation. See [Improvement Stories](https://github.com/vinpatel/manuscript/blob/main/IMPROVEMENT_STORIES.md) for planned enhancements.

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- GIF (.gif) - first frame
- BMP (.bmp)
- TIFF (.tif, .tiff)

## Best Practices

1. **Original files:** Detection works best on original camera files
2. **Uncompressed:** Avoid heavily compressed images
3. **Full resolution:** Don't resize before analysis
4. **Multiple samples:** For important decisions, analyze multiple images

## Planned Improvements

- DCT coefficient analysis for JPEG forensics
- Full EXIF parsing with validation
- PNG chunk analysis
- CLIP-based deep analysis (optional)
