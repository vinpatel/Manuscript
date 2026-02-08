---
title: Audio Detection
description: How Manuscript detects AI-generated audio
---

# Audio Detection

Manuscript analyzes audio files to detect synthetic voices and AI-generated music.

## Detection Signals

### Container Metadata

Real recordings contain metadata about the recording device and software.

| Metadata | Real Recording | AI-Generated |
|----------|---------------|--------------|
| Encoder | Authentic DAW/recorder | TTS engine markers |
| Sample rate | Standard (44.1/48kHz) | Often non-standard |
| Bit depth | 16/24-bit | Variable |
| Channel layout | Stereo/mono consistent | May be inconsistent |

**Weight:** 0.25

### Spectral Analysis (FFT)

v0.2.0 introduced FFT spectral analysis to detect AI audio patterns:

- **Frequency distribution:** AI audio often has unnatural frequency cutoffs
- **Harmonic structure:** Synthetic voices may lack natural harmonics
- **Background noise:** Real recordings have ambient noise patterns

**Weight:** 0.20

### MFCC Computation

Mel-Frequency Cepstral Coefficients help distinguish natural speech from synthesis:

- **Formant patterns:** Human formants vary naturally
- **Transition smoothness:** AI may have unnatural transitions
- **Dynamic range:** Real speech has more variation

**Weight:** 0.20

### AI Signature Detection

Manuscript detects markers from known AI audio tools:

- ElevenLabs voice cloning
- Suno music generation
- Bark text-to-speech
- VALL-E clones
- Other TTS engines

**Weight:** 0.15

### Temporal Consistency

Analyzes timing patterns:

- **Breath patterns:** Real speakers breathe naturally
- **Pacing:** AI may have too-consistent timing
- **Pauses:** Natural pauses vs mechanical silence

**Weight:** 0.10

### Noise Profile

Real recordings have characteristic noise:

- Room ambience
- Microphone noise floor
- Environmental sounds

AI-generated audio is often "too clean."

**Weight:** 0.10

## v0.2.0 Improvements

| Metric | v0.1.0 | v0.2.0 | Change |
|--------|--------|--------|--------|
| Accuracy | 38.00% | 46.00% | +8.00% |
| Recall | 76.00% | 92.00% | +16.00% |
| F1 Score | 55.07% | 63.01% | +7.94% |

Improvements came from:
1. FFT spectral analysis
2. MFCC computation
3. Temporal consistency checking

## API Usage

### Upload Audio File

```bash
curl -X POST http://localhost:8080/verify \
  -F "audio=@recording.mp3"
```

### Response

```json
{
  "id": "hm_audio456",
  "verdict": "ai",
  "confidence": 0.82,
  "content_type": "audio",
  "signals": {
    "metadata_score": 0.30,
    "spectral_analysis": 0.75,
    "mfcc_score": 0.68,
    "ai_signatures": ["elevenlabs_marker"],
    "noise_profile": 0.25,
    "temporal_consistency": 0.70
  },
  "processing_time_ms": 120
}
```

## Current Benchmarks

| Metric | Value |
|--------|-------|
| Accuracy | 46.00% |
| Precision | 47.92% |
| Recall | 92.00% |
| F1 Score | 63.01% |

### Known Issues

1. **Studio recordings:** Clean human recordings resemble AI audio
2. **Heavy processing:** EQ/compression can mask natural markers
3. **New TTS models:** Latest models may not be in signature database

## Supported Formats

- MP3 (.mp3)
- WAV (.wav)
- FLAC (.flac)
- OGG (.ogg)
- M4A (.m4a)
- AAC (.aac)

## Best Practices

1. **Original files:** Use unprocessed recordings when possible
2. **Minimum length:** At least 5 seconds for reliable detection
3. **Avoid compression:** Heavy compression removes markers
4. **Multiple samples:** Analyze several clips from the same source

## Planned Improvements

- Full audio waveform decoding
- Enhanced MFCC analysis
- Vocoder fingerprinting
- Real-time streaming detection
