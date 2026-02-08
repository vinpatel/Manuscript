
<div align="center">

# 🔍 Manuscript

### The Open Source AI Content Detector That Respects Your Privacy

**Detect AI-generated text, images, audio & video—100% offline, self-hosted, zero external calls.**

[![GitHub Stars](https://img.shields.io/github/stars/vinpatel/manuscript?style=for-the-badge&logo=github&color=FFD700)](https://github.com/vinpatel/manuscript/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8?style=for-the-badge&logo=go)](https://go.dev)
[![Docker Pulls](https://img.shields.io/docker/pulls/manuscript/manuscript?style=for-the-badge&logo=docker)](https://hub.docker.com/r/manuscript/manuscript)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge)](CONTRIBUTING.md)

<br />

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-api-reference) • [🎯 Use Cases](#-use-cases) • [💬 Discussions](https://github.com/vinpatel/manuscript/discussions)

<br />


</div>

---

## 🤔 The Problem

Every AI detection service requires you to **upload your content to their servers**. That's a dealbreaker for:

- 🏥 **Healthcare** — HIPAA compliance prohibits sending patient data externally
- ⚖️ **Law Firms** — Attorney-client privilege can't survive third-party uploads  
- 🏦 **Finance** — SOC2/PCI requirements restrict data sharing
- 🛡️ **Government** — Air-gapped networks, classified environments
- 🎓 **Universities** — 100K+ students = $100K+ annual licensing

**Manuscript runs entirely on YOUR infrastructure. Your data never leaves your network.**

---

## ⚡ Quick Start

Get running in under 30 seconds:

```bash
# Option 1: Docker (Recommended)
docker run -p 8080:8080 manuscript/manuscript

# Option 2: Go Install
go install github.com/vinpatel/manuscript/cmd/api@latest
manuscript

# Option 3: Build from Source
git clone https://github.com/vinpatel/manuscript.git
cd manuscript && make run
```

Then detect AI content:

```bash
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here"}'
```

**Response:**
```json
{
  "id": "hm_abc123",
  "verdict": "human",
  "confidence": 0.87,
  "signals": {
    "sentence_variance": 0.42,
    "vocabulary_richness": 0.78,
    "contraction_ratio": 0.15
  }
}
```

---

## 🎯 Use Cases

<table>
<tr>
<td width="50%" valign="top">

### 👨‍💻 For Developers

| Integration | What It Does |
|------------|--------------|
| **Content Platforms** | Filter AI spam from UGC |
| **Hiring Tools** | Verify candidate samples |
| **EdTech** | Academic integrity checks |
| **Social Apps** | Flag synthetic profiles |
| **CMS Plugins** | WordPress, Ghost, Strapi |
| **CI/CD** | Lint content like code |
| **Extensions** | Detect AI on any webpage |

</td>
<td width="50%" valign="top">

### 🏢 For Organizations

| Industry | Why Manuscript |
|----------|---------------|
| **Enterprise** | Compliance (HIPAA, GDPR, SOC2) |
| **Government** | Air-gapped, classified envs |
| **Legal** | Protect client privilege |
| **Healthcare** | Patient data stays on-prem |
| **Finance** | Regulatory restrictions |
| **Education** | Scale without per-seat costs |
| **Media** | Own the tool, don't rent it |

</td>
</tr>
</table>

---

## 🏆 Why Manuscript?

<table>
<tr>
<th align="left" width="200">Feature</th>
<th align="center" width="150">Manuscript</th>
<th align="center" width="150">GPTZero</th>
<th align="center" width="150">Originality.ai</th>
<th align="center" width="150">Turnitin</th>
</tr>
<tr>
<td><b>Self-hosted</b></td>
<td align="center">✅</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>
<tr>
<td><b>Works Offline</b></td>
<td align="center">✅</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>
<tr>
<td><b>Open Source</b></td>
<td align="center">✅ MIT</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>
<tr>
<td><b>Zero Cost</b></td>
<td align="center">✅ Forever</td>
<td align="center">❌ $$$</td>
<td align="center">❌ $$$</td>
<td align="center">❌ $$$$</td>
</tr>
<tr>
<td><b>Privacy-First</b></td>
<td align="center">✅ No data leaves</td>
<td align="center">⚠️ Cloud-only</td>
<td align="center">⚠️ Cloud-only</td>
<td align="center">⚠️ Cloud-only</td>
</tr>
<tr>
<td><b>Multi-Modal</b></td>
<td align="center">✅ Text/Image/Audio/Video</td>
<td align="center">⚠️ Text only</td>
<td align="center">⚠️ Text only</td>
<td align="center">⚠️ Text only</td>
</tr>
<tr>
<td><b>API Limits</b></td>
<td align="center">∞ Unlimited</td>
<td align="center">⚠️ Tiered</td>
<td align="center">⚠️ Per-check</td>
<td align="center">⚠️ Per-student</td>
</tr>
</table>

---

## 🔬 How It Works

Manuscript uses **statistical and forensic analysis**—no ML models, no GPU required, instant results.

<details>
<summary><b>📝 Text Detection Signals</b></summary>

| Signal | Human Writing | AI Writing |
|--------|--------------|------------|
| Sentence length variance | High (varied rhythm) | Low (uniform) |
| Vocabulary richness | Diverse, personal words | "Safe" common words |
| Contractions | "don't", "I'm", "we'll" | "do not", "I am" |
| Punctuation variety | !?;:—... | Mostly periods |
| AI phrases | Rare | "As an AI...", "It's important to note..." |
| Hedging language | Natural uncertainty | Excessive qualifiers |
| Repetition patterns | Organic callbacks | Mechanical repetition |

</details>

<details>
<summary><b>🖼️ Image Detection Signals</b></summary>

| Signal | Real Photo | AI-Generated |
|--------|-----------|--------------|
| EXIF metadata | Present (camera, GPS, date) | Missing or fake |
| Camera make | Apple, Canon, Sony, etc. | None or generic |
| Sensor noise | Natural grain patterns | Too clean or uniform |
| Compression artifacts | JPEG-consistent | Inconsistent patterns |
| Color distribution | Natural histogram | Artificial smoothing |

</details>

<details>
<summary><b>🎵 Audio/Video Detection</b></summary>

Analyzes format metadata, encoder signatures, and AI tool markers in:
- File headers and container metadata
- Encoding parameters and profiles
- Generation tool fingerprints (e.g., ElevenLabs, Suno markers)
- Temporal consistency patterns

</details>

---

## 📖 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/verify` | POST | Analyze content for AI generation |
| `/verify/{id}` | GET | Retrieve analysis by ID |
| `/batch` | POST | Analyze multiple items |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### Detailed Analysis

```bash
curl -X POST "http://localhost:8080/verify?detailed=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here"}'
```

**Response with full signal breakdown:**
```json
{
  "id": "hm_xyz789",
  "verdict": "ai",
  "confidence": 0.92,
  "content_type": "text",
  "signals": {
    "sentence_variance": 0.12,
    "vocabulary_richness": 0.34,
    "contraction_ratio": 0.02,
    "ai_phrases_detected": ["It's important to note", "Additionally"],
    "hedging_score": 0.78
  },
  "processing_time_ms": 23
}
```

### Image Analysis

```bash
curl -X POST http://localhost:8080/verify \
  -F "image=@photo.jpg"
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `ENV` | `development` | Environment mode |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `LOG_FORMAT` | `json` | Log format (json/text) |
| `METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_TEXT_LENGTH` | `100000` | Max text chars |
| `MAX_IMAGE_SIZE` | `10MB` | Max image upload |

### Docker Compose

```yaml
version: '3.8'
services:
  manuscript:
    image: manuscript/manuscript:latest
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=info
      - METRICS_ENABLED=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: manuscript
spec:
  replicas: 3
  selector:
    matchLabels:
      app: manuscript
  template:
    spec:
      containers:
      - name: manuscript
        image: manuscript/manuscript:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
```

---

## 📊 Performance

Benchmarked on AWS c5.xlarge (4 vCPU, 8GB RAM):

| Metric | Manuscript | Alternative A | Alternative B |
|--------|-----------|---------------|---------------|
| **Requests/sec** | 12,400 | N/A (cloud) | N/A (cloud) |
| **Latency (p50)** | 8ms | 180ms | 240ms |
| **Latency (p99)** | 23ms | 890ms | 1200ms |
| **Memory usage** | 45MB | N/A | N/A |
| **Cold start** | 150ms | N/A | N/A |

---

## 🗺️ Roadmap

- [x] Text detection (statistical analysis)
- [x] Image detection (EXIF + forensics)
- [x] Audio detection (metadata analysis)
- [x] Video detection (container analysis)
- [x] Docker support
- [x] Prometheus metrics
- [ ] 🔜 Browser extension
- [ ] 🔜 VS Code extension  
- [ ] 🔜 WordPress plugin
- [ ] 🔜 Python SDK
- [ ] 🔜 JavaScript SDK
- [ ] 🔜 Webhook notifications
- [ ] 🔜 Admin dashboard

See our [project board](https://github.com/vinpatel/manuscript/projects/1) for the full roadmap.

---

## 🤝 Contributing

We love contributions! Here's how to get involved:

1. **Star this repo** ⭐ — It helps more than you think!
2. **Report bugs** — [Open an issue](https://github.com/vinpatel/manuscript/issues/new?template=bug_report.md)
3. **Suggest features** — [Start a discussion](https://github.com/vinpatel/manuscript/discussions/new?category=ideas)
4. **Submit PRs** — See [CONTRIBUTING.md](CONTRIBUTING.md)

### Good First Issues

Looking to contribute? Check out issues labeled [`good first issue`](https://github.com/vinpatel/manuscript/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

### Help Us Improve Accuracy

Found a false positive or false negative? [Submit a sample](https://github.com/vinpatel/manuscript/issues/new?template=accuracy_report.md) to help improve detection accuracy!

---

## 💬 Community

- 💬 [GitHub Discussions](https://github.com/vinpatel/manuscript/discussions) — Ask questions, share ideas
- 🐛 [Issue Tracker](https://github.com/vinpatel/manuscript/issues) — Report bugs
- 🐦 [Twitter/X](https://twitter.com/manuscript) — Updates and announcements
- 📧 [Email](mailto:hello@manuscript.dev) — Business inquiries

---

## 📜 License

MIT License — use it however you want. See [LICENSE](LICENSE) for details.

---

<div align="center">

### ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vinpatel/manuscript&type=Date)](https://star-history.com/#vinpatel/manuscript&Date)

---

**If Manuscript helps you, please consider giving it a ⭐**

It takes 2 seconds and helps the project reach more people who need privacy-first AI detection.

<br />

Made with ❤️ by [Vin Patel](https://vinpatel.com) and [contributors](https://github.com/vinpatel/manuscript/graphs/contributors)

[⬆ Back to Top](#-manuscript)

</div>
