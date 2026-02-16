---
title: Video Detection
description: How Manuscript detects deepfakes and AI-generated video
---

# Video Detection

Manuscript analyzes video files to detect deepfakes, face swaps, and fully AI-generated video content.

## Detection Signals

### Container Metadata

Video containers contain rich metadata:

| Metadata | Real Video | AI-Generated |
|----------|-----------|--------------|
| Encoder | Standard codecs (H.264, H.265) | May use non-standard |
| Creation software | Premiere, Final Cut, etc. | AI tool markers |
| Frame rate | Standard (24/30/60fps) | May be unusual |
| Bitrate pattern | Variable (VBR) | Often constant |

**Weight:** 0.25

### Temporal Pattern Analysis

AI-generated videos often have temporal inconsistencies:

- **Frame timing:** Irregular intervals
- **Motion blur:** Incorrect blur patterns
- **Flickering:** Subtle frame-to-frame inconsistencies

**Weight:** 0.15

### Encoding Signature

Identifies markers from AI video tools:

- Sora
- Runway Gen-2
- Pika Labs
- Stable Video Diffusion
- DeepFake tools

**Weight:** 0.15

### Bitrate Consistency

Real videos have natural bitrate variation. AI videos may have:

- Too-consistent bitrate
- Unusual compression patterns
- Incorrect keyframe intervals

**Weight:** 0.10

### Audio Track Analysis

If audio is present:

- Lip sync consistency
- Audio-video timing
- Audio authenticity (uses audio detection)

**Weight:** 0.20

### Visual Forensics

When frame analysis is enabled:

- Face consistency across frames
- Background stability
- Edge artifacts
- Reflection/shadow consistency

**Weight:** 0.15

## API Usage

Upload video files via `POST /verify` using multipart form data. See the [API Reference](/manuscript/docs/api/endpoints/) for full request/response details.

## Current Status

**Video benchmark is pending** - requires video file downloads via API keys.

### Expected Performance

Based on related benchmarks:
- **Target Accuracy:** >75%
- **Off-the-shelf detectors:** 21.3% lower accuracy on Sora-like videos
- **Primary Challenges:** New diffusion video models, compression

## Supported Formats

- MP4 (.mp4)
- WebM (.webm)
- MOV (.mov)
- AVI (.avi)
- MKV (.mkv)

## Limitations

1. **Full frame analysis** requires ffmpeg for extraction
2. **Re-encoded videos** lose original signatures
3. **Short clips** (<3 seconds) have insufficient data
4. **New AI models** may not be in signature database

## Best Practices

1. **Original files:** Use source video when possible
2. **Minimum length:** At least 3 seconds for reliable detection
3. **Include audio:** Audio track improves detection
4. **Avoid re-encoding:** Each re-encode removes markers

## Planned Improvements

- Full frame extraction with ffmpeg integration
- Face consistency analysis across frames
- Lip sync verification
- Temporal coherence scoring
- Real-time streaming detection

## Deepfake-Specific Detection

For face-swap deepfakes, Manuscript looks for:

1. **Blending boundaries** around face edges
2. **Skin tone inconsistencies**
3. **Eye/teeth artifacts**
4. **Temporal face jittering**
5. **Lighting inconsistencies**

## Resources

- [Deepfake-Eval-2024 Dataset](https://arxiv.org/html/2503.02857v1)
- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
