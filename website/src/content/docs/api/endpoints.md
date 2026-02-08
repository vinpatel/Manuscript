---
title: API Endpoints
description: Complete API reference for Manuscript
---

# API Endpoints

Complete reference for all Manuscript API endpoints.

## Base URL

```
http://localhost:8080
```

## Authentication

Manuscript does not require authentication by default. Add authentication via reverse proxy if needed.

---

## POST /verify

Analyze content for AI generation.

### Request

#### Text Content

```bash
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content to analyze..."}'
```

#### Image Upload

```bash
curl -X POST http://localhost:8080/verify \
  -F "image=@photo.jpg"
```

#### Audio Upload

```bash
curl -X POST http://localhost:8080/verify \
  -F "audio=@recording.mp3"
```

#### Video Upload

```bash
curl -X POST http://localhost:8080/verify \
  -F "video=@clip.mp4"
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detailed` | boolean | `false` | Include full signal breakdown |

### Response

```json
{
  "id": "hm_abc123def456",
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

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique analysis identifier |
| `verdict` | string | `"human"` or `"ai"` |
| `confidence` | float | 0.0 to 1.0 confidence score |
| `content_type` | string | `"text"`, `"image"`, `"audio"`, or `"video"` |
| `signals` | object | Detection signal breakdown |
| `processing_time_ms` | integer | Processing time in milliseconds |

### Error Responses

```json
{
  "error": "content_too_large",
  "message": "Text exceeds maximum length of 100000 characters",
  "status": 400
}
```

| Status | Error | Description |
|--------|-------|-------------|
| 400 | `invalid_content` | Missing or invalid content |
| 400 | `content_too_large` | Exceeds size limits |
| 400 | `unsupported_format` | Unknown file format |
| 500 | `processing_error` | Internal processing failed |

---

## GET /verify/{id}

Retrieve a previous analysis by ID.

### Request

```bash
curl http://localhost:8080/verify/hm_abc123def456
```

### Response

Same format as POST /verify response.

### Error Responses

| Status | Error | Description |
|--------|-------|-------------|
| 404 | `not_found` | Analysis ID not found |

---

## POST /batch

Analyze multiple items in a single request.

### Request

```bash
curl -X POST http://localhost:8080/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"text": "First text to analyze..."},
      {"text": "Second text to analyze..."},
      {"text": "Third text to analyze..."}
    ]
  }'
```

### Response

```json
{
  "results": [
    {
      "id": "hm_batch_001",
      "verdict": "human",
      "confidence": 0.89,
      "content_type": "text"
    },
    {
      "id": "hm_batch_002",
      "verdict": "ai",
      "confidence": 0.92,
      "content_type": "text"
    },
    {
      "id": "hm_batch_003",
      "verdict": "human",
      "confidence": 0.76,
      "content_type": "text"
    }
  ],
  "summary": {
    "total": 3,
    "human": 2,
    "ai": 1,
    "processing_time_ms": 45
  }
}
```

### Limits

- Maximum 100 items per batch
- Maximum 10MB total request size

---

## GET /health

Health check endpoint.

### Request

```bash
curl http://localhost:8080/health
```

### Response

```json
{
  "status": "healthy",
  "version": "0.2.0",
  "uptime_seconds": 3600
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` or `"unhealthy"` |
| `version` | string | Manuscript version |
| `uptime_seconds` | integer | Time since start |

---

## GET /metrics

Prometheus metrics endpoint.

### Request

```bash
curl http://localhost:8080/metrics
```

### Response

```
# HELP manuscript_requests_total Total requests by type
# TYPE manuscript_requests_total counter
manuscript_requests_total{type="text"} 1523
manuscript_requests_total{type="image"} 456
manuscript_requests_total{type="audio"} 89
manuscript_requests_total{type="video"} 12

# HELP manuscript_request_duration_seconds Request latency
# TYPE manuscript_request_duration_seconds histogram
manuscript_request_duration_seconds_bucket{le="0.01"} 1890
manuscript_request_duration_seconds_bucket{le="0.05"} 2034
manuscript_request_duration_seconds_bucket{le="0.1"} 2067
manuscript_request_duration_seconds_bucket{le="+Inf"} 2080

# HELP manuscript_active_requests Currently processing requests
# TYPE manuscript_active_requests gauge
manuscript_active_requests 3
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `manuscript_requests_total` | Counter | Total requests by content type |
| `manuscript_request_duration_seconds` | Histogram | Request latency distribution |
| `manuscript_active_requests` | Gauge | Current in-flight requests |
| `manuscript_detection_confidence` | Histogram | Confidence score distribution |

---

## Content Type Headers

### Request Content Types

| Content Type | Usage |
|--------------|-------|
| `application/json` | JSON body with `text` or `image_base64` |
| `multipart/form-data` | File uploads |

### Response Content Type

All responses are `application/json`.

---

## Rate Limiting

Manuscript does not implement rate limiting by default. Configure rate limiting at your reverse proxy level.

Example nginx configuration:

```nginx
limit_req_zone $binary_remote_addr zone=manuscript:10m rate=100r/s;

location /verify {
    limit_req zone=manuscript burst=200 nodelay;
    proxy_pass http://manuscript:8080;
}
```

---

## SDK Examples

### Python

```python
import requests

def verify_text(text):
    response = requests.post(
        "http://localhost:8080/verify",
        json={"text": text}
    )
    return response.json()

result = verify_text("Your content here...")
print(f"Verdict: {result['verdict']} ({result['confidence']:.0%})")
```

### JavaScript

```javascript
async function verifyText(text) {
  const response = await fetch("http://localhost:8080/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return response.json();
}

const result = await verifyText("Your content here...");
console.log(`Verdict: ${result.verdict} (${Math.round(result.confidence * 100)}%)`);
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
)

type VerifyRequest struct {
    Text string `json:"text"`
}

type VerifyResponse struct {
    ID         string  `json:"id"`
    Verdict    string  `json:"verdict"`
    Confidence float64 `json:"confidence"`
}

func verifyText(text string) (*VerifyResponse, error) {
    body, _ := json.Marshal(VerifyRequest{Text: text})
    resp, err := http.Post(
        "http://localhost:8080/verify",
        "application/json",
        bytes.NewReader(body),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result VerifyResponse
    json.NewDecoder(resp.Body).Decode(&result)
    return &result, nil
}
```
