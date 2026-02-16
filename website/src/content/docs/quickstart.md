---
title: Quick Start
description: Get Manuscript running in under 30 seconds
---

# Quick Start

Get Manuscript running and detecting AI content in under 30 seconds.

## Option 1: Docker (Recommended)

The fastest way to get started:

```bash
docker run -p 8080:8080 manuscript/manuscript
```

That's it! The API is now running at `http://localhost:8080`.

## Option 2: Go Install

If you have Go 1.21+ installed:

```bash
go install github.com/vinpatel/manuscript/cmd/api@latest
manuscript
```

## Option 3: Build from Source

```bash
git clone https://github.com/vinpatel/manuscript.git
cd manuscript
make run
```

## Verify It's Working

Check the health endpoint:

```bash
curl http://localhost:8080/health
```

You should see:

```json
{"status": "healthy", "version": "0.2.0"}
```

## Detect AI Content

### Text Detection

```bash
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content to analyze here..."}'
```

**Response:**

```json
{
  "id": "hm_abc123",
  "verdict": "human",
  "confidence": 0.87,
  "content_type": "text",
  "signals": {
    "sentence_variance": 0.42,
    "vocabulary_richness": 0.78,
    "contraction_ratio": 0.15
  },
  "processing_time_ms": 8
}
```

### Other Content Types

The same `/verify` endpoint handles images, audio, and video via file upload:

```bash
curl -X POST http://localhost:8080/verify -F "image=@photo.jpg"
curl -X POST http://localhost:8080/verify -F "audio=@recording.mp3"
curl -X POST http://localhost:8080/verify -F "video=@clip.mp4"
```

See the [API Reference](/manuscript/docs/api/endpoints/) for full request/response details, batch processing, and query parameters.

## Next Steps

- [Installation Guide](/manuscript/docs/installation/) - Production deployment
- [API Reference](/manuscript/docs/api/endpoints/) - Full API documentation
- [Text Detection](/manuscript/docs/text-detection/) - How text detection works
