---
title: Installation
description: Detailed installation and deployment guide for Manuscript
---

# Installation

This guide covers production deployment of Manuscript.

## Requirements

- Go 1.21+ (for building from source)
- Docker (for containerized deployment)
- 128MB RAM minimum
- Any x86_64 or ARM64 architecture

## Docker Deployment

### Basic Deployment

```bash
docker run -d \
  --name manuscript \
  -p 8080:8080 \
  --restart unless-stopped \
  manuscript/manuscript:latest
```

### Docker Compose

Create a `docker-compose.yml`:

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
    restart: unless-stopped
```

Then run:

```bash
docker-compose up -d
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: manuscript
  labels:
    app: manuscript
spec:
  replicas: 3
  selector:
    matchLabels:
      app: manuscript
  template:
    metadata:
      labels:
        app: manuscript
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
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: manuscript
spec:
  selector:
    app: manuscript
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/vinpatel/manuscript.git
cd manuscript

# Build the binary
make build

# Or build with version info
make build-release

# Run
./bin/manuscript
```

## Configuration

Manuscript is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `ENV` | `development` | Environment mode |
| `LOG_LEVEL` | `info` | Logging verbosity (debug/info/warn/error) |
| `LOG_FORMAT` | `json` | Log format (json/text) |
| `METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_TEXT_LENGTH` | `100000` | Maximum text length (chars) |
| `MAX_IMAGE_SIZE` | `10MB` | Maximum image upload size |

### Example with Custom Config

```bash
docker run -d \
  -p 8080:8080 \
  -e LOG_LEVEL=debug \
  -e METRICS_ENABLED=true \
  -e MAX_TEXT_LENGTH=50000 \
  manuscript/manuscript:latest
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

Available metrics:
- `manuscript_requests_total` - Total requests by type
- `manuscript_request_duration_seconds` - Request latency histogram
- `manuscript_detection_accuracy` - Detection accuracy gauge
- `manuscript_active_requests` - Currently processing requests

### Grafana Dashboard

Import the provided Grafana dashboard from `monitoring/grafana-dashboard.json`.

## Security Considerations

1. **Network Isolation** - Run behind a reverse proxy
2. **TLS Termination** - Use nginx/traefik for HTTPS
3. **Rate Limiting** - Configure at reverse proxy level
4. **Authentication** - Add auth middleware as needed

## Next Steps

- [API Reference](/manuscript/docs/api/endpoints/) - Full API documentation
- [Benchmarks](/manuscript/docs/benchmarks/) - Performance metrics
