# Manuscript Improvement Stories

Generated from algorithm analysis comparing Manuscript with state-of-the-art AI detection research.

---

## Tier 1: Improvements Within Current Architecture (No ML Required)

These stories maintain Manuscript's core value proposition: no ML models, no GPU, fully on-premise.

---

### Story 1.1: Implement Proper Text Tokenization

**Title**: Add BPE-like tokenization for improved vocabulary analysis

**As a** Manuscript user analyzing text content
**I want** the text analyzer to use proper tokenization instead of simple whitespace splitting
**So that** vocabulary metrics are more accurate and consistent with how LLMs process text

**Background**:
Current implementation (`text_analyzer.go:164-189`) uses simple regex-based word splitting. This misses subword patterns and doesn't align with how LLMs tokenize text.

**Acceptance Criteria**:
- [ ] Implement a simple BPE-like tokenizer or integrate a lightweight Go tokenization library
- [ ] Update `calculateVocabularyRichness()` to use token-based analysis
- [ ] Add token-level statistics (average tokens per word, rare token ratio)
- [ ] Maintain backwards compatibility with existing API response format
- [ ] Add unit tests comparing old vs new tokenization accuracy

**Technical Notes**:
- Consider using `github.com/tiktoken-go/tokenizer` for GPT-compatible tokenization
- Alternative: implement simple character-level n-gram analysis
- Keep tokenizer initialization lazy to avoid startup cost

**Files to Modify**:
- `internal/service/text_analyzer.go`
- `internal/service/text_analyzer_test.go`

**Estimated Complexity**: Medium

---

### Story 1.2: Add N-gram Statistical Analysis for Text

**Title**: Implement bigram and trigram frequency analysis

**As a** Manuscript user
**I want** the text analyzer to compute n-gram statistics
**So that** AI-generated text with unnaturally uniform n-gram distributions can be detected

**Background**:
Research shows AI text has more uniform n-gram distributions than human text. Humans repeat certain phrases and have personal writing patterns; AI optimizes for "average" text.

**Acceptance Criteria**:
- [ ] Add bigram frequency calculation
- [ ] Add trigram frequency calculation
- [ ] Compute n-gram entropy as a signal
- [ ] Add n-gram repetition detection (same trigram appearing multiple times)
- [ ] Integrate as new weighted signal in `TextSignals` struct
- [ ] Add configurable weight (suggest 0.10)

**Technical Notes**:
```go
type TextSignals struct {
    // ... existing signals
    NGramUniformity float64 // NEW: High uniformity = AI-like
}
```

**Files to Modify**:
- `internal/service/text_analyzer.go`
- `internal/service/text_analyzer_test.go`

**Estimated Complexity**: Low-Medium

---

### Story 1.3: Add Readability Score Analysis

**Title**: Implement Flesch-Kincaid and Gunning Fog readability metrics

**As a** Manuscript user
**I want** readability scores computed for analyzed text
**So that** unnaturally consistent readability levels (common in AI) can be flagged

**Background**:
AI-generated text often maintains a very consistent reading level throughout. Human writing varies in complexity based on topic, emotion, and emphasis.

**Acceptance Criteria**:
- [ ] Implement Flesch-Kincaid Grade Level calculation
- [ ] Implement Gunning Fog Index calculation
- [ ] Compute variance of readability across paragraphs/sections
- [ ] Low variance in readability = higher AI score
- [ ] Add to detailed response when `?detailed=true`

**Technical Notes**:
```go
// Flesch-Kincaid Grade Level formula
func fleschKincaidGrade(words, sentences, syllables int) float64 {
    return 0.39*(float64(words)/float64(sentences)) +
           11.8*(float64(syllables)/float64(words)) - 15.59
}
```

**Files to Modify**:
- `internal/service/text_analyzer.go`
- `internal/service/text_analyzer_test.go`

**Estimated Complexity**: Low

---

### Story 1.4: Expand AI Phrase Detection Database

**Title**: Expand AI phrase patterns from ~35 to 500+ with model-specific patterns

**As a** Manuscript user
**I want** a comprehensive database of AI-specific phrases and patterns
**So that** detection catches more subtle AI tells across different models

**Background**:
Current implementation (`text_analyzer.go:369-444`) has ~35 patterns. Different AI models have distinct phrase preferences (Claude uses "I'd be happy to", GPT uses "Certainly!", etc.).

**Acceptance Criteria**:
- [ ] Expand phrase database to 500+ patterns
- [ ] Categorize patterns by AI model family (GPT, Claude, Llama, Gemini)
- [ ] Add weighted scoring (some phrases are stronger signals than others)
- [ ] Add phrase context matching (phrase at start of response vs middle)
- [ ] Make phrase database configurable/extensible via JSON file
- [ ] Add versioning to phrase database for updates

**Technical Notes**:
```go
type AIPhrase struct {
    Pattern    string   `json:"pattern"`
    Weight     float64  `json:"weight"`      // 0.1-1.0
    Models     []string `json:"models"`      // ["gpt", "claude", "all"]
    Position   string   `json:"position"`    // "start", "any", "end"
    Category   string   `json:"category"`    // "hedging", "formal", "filler"
}
```

**Suggested New Patterns**:
- Claude: "I'd be happy to", "I should note", "I appreciate", "That said,"
- GPT: "Certainly!", "Absolutely!", "Great question!", "Here's"
- General: "In conclusion,", "To summarize,", "It's worth noting"
- Formal: "pertaining to", "in regards to", "aforementioned"

**Files to Modify**:
- `internal/service/text_analyzer.go`
- NEW: `internal/service/ai_phrases.json` (or embed in Go)
- `internal/service/text_analyzer_test.go`

**Estimated Complexity**: Medium

---

### Story 1.5: Implement DCT Coefficient Analysis for JPEG Images

**Title**: Add frequency-domain analysis for JPEG images without full decoding

**As a** Manuscript user analyzing images
**I want** the image analyzer to examine DCT coefficients in JPEG files
**So that** AI-generated images with abnormal frequency patterns are detected

**Background**:
JPEG stores image data as DCT (Discrete Cosine Transform) coefficients. AI-generated images often have distinct DCT coefficient distributions that differ from camera-captured photos. This can be analyzed without fully decoding the image.

**Acceptance Criteria**:
- [ ] Parse JPEG to extract quantization tables
- [ ] Analyze DCT coefficient distribution patterns
- [ ] Detect double-compression artifacts
- [ ] Compare against expected camera vs AI patterns
- [ ] Add as new signal in `ImageSignals` struct

**Technical Notes**:
```go
func (a *ImageAnalyzer) analyzeDCTCoefficients(jpegData []byte) float64 {
    // Parse JPEG markers to find DQT (Define Quantization Table)
    // Analyze coefficient histogram
    // AI images often have unusual high-frequency coefficient patterns
}
```

**Research Reference**: AI images often lack the natural DCT coefficient decay pattern seen in real photos.

**Files to Modify**:
- `internal/service/image_analyzer.go`
- `internal/service/image_analyzer_test.go`

**Estimated Complexity**: High

---

### Story 1.6: Implement Full EXIF Parsing for Images

**Title**: Add comprehensive EXIF metadata extraction and analysis

**As a** Manuscript user
**I want** complete EXIF parsing including all camera and software metadata
**So that** image provenance can be thoroughly validated

**Background**:
Current implementation does basic EXIF detection (`image_analyzer.go:280-335`) but doesn't parse all fields. Full EXIF contains timestamps, GPS, camera settings, software history that can reveal AI generation.

**Acceptance Criteria**:
- [ ] Parse all standard EXIF tags (camera make/model, exposure, ISO, etc.)
- [ ] Parse EXIF software/processing history
- [ ] Detect timestamp inconsistencies
- [ ] Check for AI watermark fields (C2PA, metadata injections)
- [ ] Validate GPS coordinate plausibility
- [ ] Add IPTC/XMP metadata parsing

**Technical Notes**:
- Use `github.com/rwcarlsen/goexif/exif` or similar
- Check for C2PA (Coalition for Content Provenance and Authenticity) metadata
- Look for Adobe XMP `ai:generatorName` field

**Files to Modify**:
- `internal/service/image_analyzer.go`
- `go.mod` (add EXIF library if needed)
- `internal/service/image_analyzer_test.go`

**Estimated Complexity**: Medium

---

### Story 1.7: Add PNG Chunk Analysis for Images

**Title**: Implement PNG ancillary chunk parsing for AI detection

**As a** Manuscript user analyzing PNG images
**I want** the analyzer to parse PNG chunks for metadata and AI markers
**So that** AI-generated PNGs can be detected through their metadata

**Background**:
PNG files contain chunks that can include text metadata, software info, and AI generation markers. Many AI tools embed metadata in `tEXt`, `iTXt`, or `eXIf` chunks.

**Acceptance Criteria**:
- [ ] Parse PNG chunk structure
- [ ] Extract `tEXt` and `iTXt` chunks for AI markers
- [ ] Check for Stable Diffusion parameters in metadata
- [ ] Detect ComfyUI/Automatic1111 metadata formats
- [ ] Extract and analyze embedded ICC profiles

**Technical Notes**:
```go
// Stable Diffusion often embeds parameters like:
// "parameters: masterpiece, best quality, ..."
// "Negative prompt: ..."
// "Steps: 20, Sampler: Euler, CFG scale: 7"
```

**Files to Modify**:
- `internal/service/image_analyzer.go`
- `internal/service/image_analyzer_test.go`

**Estimated Complexity**: Medium

---

## Tier 2: Add Optional Lightweight Processing

These stories add optional audio/video decoding for deeper analysis while keeping ML optional.

---

### Story 2.1: Add Audio Waveform Decoding

**Title**: Implement audio decoding to PCM for spectral analysis

**As a** Manuscript user analyzing audio
**I want** optional audio decoding to enable real spectral analysis
**So that** synthetic speech characteristics can be properly detected

**Background**:
Current audio analysis (`audio_analyzer.go`) operates on compressed bytes, which cannot detect the acoustic features that distinguish synthetic speech. Decoding to PCM enables proper analysis.

**Acceptance Criteria**:
- [ ] Add MP3 decoding to PCM (use `github.com/hajimehoshi/go-mp3` or similar)
- [ ] Add WAV PCM extraction
- [ ] Add FLAC decoding support
- [ ] Make decoding optional (config flag)
- [ ] Implement basic waveform statistics (RMS, zero-crossing rate)
- [ ] Add spectral centroid calculation

**Technical Notes**:
- Keep existing container analysis as fallback when decoding disabled
- Decoding increases processing time ~10-50x
- Consider streaming decode for large files

**Files to Modify**:
- `internal/service/audio_analyzer.go`
- `internal/config/config.go` (add `ENABLE_AUDIO_DECODE` flag)
- `go.mod`
- `internal/service/audio_analyzer_test.go`

**Estimated Complexity**: High

---

### Story 2.2: Implement Basic FFT Spectral Analysis for Audio

**Title**: Add FFT-based spectral envelope analysis for decoded audio

**As a** Manuscript user
**I want** frequency-domain analysis of audio content
**So that** synthetic speech with unnatural spectral characteristics can be detected

**Background**:
Synthetic speech often has telltale spectral characteristics: unnaturally smooth formants, missing harmonics, or artificial spectral envelope.

**Acceptance Criteria**:
- [ ] Implement FFT on decoded PCM audio
- [ ] Calculate spectral envelope
- [ ] Detect formant frequencies
- [ ] Compute spectral flatness (synthetic audio often has unnatural flatness)
- [ ] Add harmonic-to-noise ratio estimation
- [ ] Integrate as new weighted signal

**Technical Notes**:
```go
// Use github.com/mjibson/go-dsp/fft or implement Cooley-Tukey
func computeSpectralEnvelope(samples []float64, sampleRate int) []float64 {
    // Window the signal (Hann window)
    // Compute FFT
    // Calculate magnitude spectrum
    // Smooth to get envelope
}
```

**Dependencies**: Requires Story 2.1 (audio decoding)

**Files to Modify**:
- `internal/service/audio_analyzer.go`
- `go.mod` (add FFT library)
- `internal/service/audio_analyzer_test.go`

**Estimated Complexity**: High

---

### Story 2.3: Add MFCC Computation for Audio

**Title**: Implement Mel-Frequency Cepstral Coefficients extraction

**As a** Manuscript user
**I want** MFCC features computed from audio
**So that** the most discriminative features for speech analysis are available

**Background**:
MFCCs are the standard features for speech analysis, used by all state-of-the-art audio deepfake detectors. They capture the spectral envelope in a perceptually meaningful way.

**Acceptance Criteria**:
- [ ] Implement mel filterbank
- [ ] Compute MFCCs (typically 13-20 coefficients)
- [ ] Calculate delta and delta-delta coefficients
- [ ] Compute statistics over MFCCs (mean, variance, skewness)
- [ ] Use MFCC statistics as detection signals

**Technical Notes**:
```go
type MFCCConfig struct {
    NumCoeffs    int     // typically 13
    NumFilters   int     // typically 26
    FrameSize    int     // typically 25ms
    FrameStride  int     // typically 10ms
    PreEmphasis  float64 // typically 0.97
}
```

**Dependencies**: Requires Story 2.1 (audio decoding)

**Files to Modify**:
- `internal/service/audio_analyzer.go`
- `internal/service/audio_analyzer_test.go`

**Estimated Complexity**: High

---

### Story 2.4: Add Video Frame Extraction

**Title**: Implement keyframe extraction from video containers

**As a** Manuscript user analyzing video
**I want** the analyzer to extract and analyze video frames
**So that** per-frame forensic analysis can detect AI-generated video

**Background**:
Current video analysis only examines container metadata. Real AI video detection requires analyzing actual frames for artifacts, temporal inconsistencies, and facial analysis.

**Acceptance Criteria**:
- [ ] Extract I-frames (keyframes) from MP4/WebM without full decode
- [ ] Apply existing image forensics to extracted frames
- [ ] Compute inter-frame consistency metrics
- [ ] Sample frames at configurable intervals
- [ ] Make frame extraction optional (config flag)

**Technical Notes**:
- I-frames can be extracted without full decoding by parsing NAL units
- Consider using `github.com/3d0c/gmf` for frame extraction
- Limit to N frames (e.g., 10) to control processing time

**Files to Modify**:
- `internal/service/video_analyzer.go`
- `internal/config/config.go`
- `go.mod`
- `internal/service/video_analyzer_test.go`

**Estimated Complexity**: Very High

---

### Story 2.5: Add Temporal Consistency Analysis for Video

**Title**: Implement frame-to-frame consistency checking

**As a** Manuscript user
**I want** analysis of temporal consistency between video frames
**So that** AI-generated videos with frame-to-frame artifacts can be detected

**Background**:
AI-generated videos often have temporal inconsistencies: flickering, morphing artifacts, inconsistent lighting, or objects that appear/disappear. These are detectable by comparing adjacent frames.

**Acceptance Criteria**:
- [ ] Compare adjacent extracted frames
- [ ] Compute frame difference statistics
- [ ] Detect unusual flickering patterns
- [ ] Check for lighting consistency
- [ ] Detect "morphing" artifacts common in AI video
- [ ] Add temporal consistency score as new signal

**Dependencies**: Requires Story 2.4 (frame extraction)

**Files to Modify**:
- `internal/service/video_analyzer.go`
- `internal/service/video_analyzer_test.go`

**Estimated Complexity**: High

---

## Tier 3: Add Optional ML Components (Maintaining On-Premise)

These stories add optional ML inference while maintaining on-premise operation.

---

### Story 3.1: Add ONNX Runtime Integration

**Title**: Integrate ONNX Runtime for optional ML model inference

**As a** Manuscript operator
**I want** optional ONNX model inference capability
**So that** ML models can be used on-premise without external API calls

**Background**:
ONNX Runtime allows running ML models locally. This enables Manuscript to optionally use pre-trained detection models while maintaining its privacy-first, on-premise architecture.

**Acceptance Criteria**:
- [ ] Add ONNX Runtime Go bindings
- [ ] Create model loading infrastructure
- [ ] Add configuration for model paths
- [ ] Implement graceful fallback when models unavailable
- [ ] Add model versioning support
- [ ] Document model requirements

**Technical Notes**:
- Use `github.com/yalue/onnxruntime_go`
- Models stored in configurable directory
- CPU inference only (no GPU requirement)

**Files to Modify**:
- NEW: `internal/ml/onnx_runtime.go`
- `internal/config/config.go`
- `go.mod`
- `docker-compose.yml` (add volume for models)

**Estimated Complexity**: High

---

### Story 3.2: Implement Local Perplexity Calculation for Text

**Title**: Add small GPT-2 model for actual perplexity computation

**As a** Manuscript user
**I want** optional true perplexity calculation using a local language model
**So that** text detection accuracy matches research-grade detectors

**Background**:
Current implementation approximates perplexity through statistical proxies. True perplexity requires computing log probabilities from a language model. GPT-2 (124M) is small enough to run on CPU.

**Acceptance Criteria**:
- [ ] Integrate GPT-2 124M ONNX model (~500MB)
- [ ] Implement token probability extraction
- [ ] Compute perplexity per token and aggregate
- [ ] Implement simplified DetectGPT (perplexity curvature)
- [ ] Add as high-weight signal when model available
- [ ] Maintain fallback to statistical analysis

**Technical Notes**:
```go
func (d *TextDetector) computePerplexity(text string) (float64, error) {
    // Tokenize text
    // Run through GPT-2 model
    // Extract log probabilities
    // Compute perplexity = exp(-1/N * sum(log(p)))
}
```

**Dependencies**: Requires Story 3.1 (ONNX Runtime)

**Files to Modify**:
- `internal/service/text_detector.go`
- NEW: `internal/ml/perplexity.go`
- `internal/service/text_detector_test.go`

**Estimated Complexity**: Very High

---

### Story 3.3: Add CLIP Image Encoder for Image Detection

**Title**: Integrate CLIP ViT model for image feature extraction

**As a** Manuscript user
**I want** optional CLIP-based image analysis
**So that** image detection matches state-of-the-art accuracy

**Background**:
Research shows CLIP-based detectors achieve 97%+ accuracy and are robust to adversarial attacks. CLIP extracts high-level semantic features that capture AI generation artifacts.

**Acceptance Criteria**:
- [ ] Integrate CLIP ViT-B/32 ONNX model (~350MB)
- [ ] Implement image preprocessing pipeline
- [ ] Extract CLIP embeddings
- [ ] Train/use simple classifier on embeddings (or use cosine similarity to known AI/human centroids)
- [ ] Add as high-weight signal when model available

**Technical Notes**:
```go
func (d *ImageDetector) computeCLIPScore(imageData []byte) (float64, error) {
    // Decode image
    // Resize to 224x224
    // Normalize with CLIP stats
    // Run through CLIP vision encoder
    // Compare embedding to AI/human centroids
}
```

**Dependencies**: Requires Story 3.1 (ONNX Runtime)

**Files to Modify**:
- `internal/service/image_detector.go` (in `media_detectors.go`)
- NEW: `internal/ml/clip.go`
- `internal/service/image_detector_test.go`

**Estimated Complexity**: Very High

---

### Story 3.4: Add Audio Classification Model

**Title**: Integrate speech deepfake detection model

**As a** Manuscript user
**I want** optional ML-based audio deepfake detection
**So that** synthetic speech is detected with high accuracy

**Background**:
State-of-the-art audio deepfake detection requires ML models trained on synthetic speech datasets. A small classifier on MFCC features can achieve 85-95% accuracy.

**Acceptance Criteria**:
- [ ] Train or obtain small audio classifier (XGBoost or small CNN)
- [ ] Implement MFCC → model pipeline
- [ ] Export model to ONNX format
- [ ] Integrate with existing audio analyzer
- [ ] Document training data and methodology

**Dependencies**:
- Requires Story 2.3 (MFCC computation)
- Requires Story 3.1 (ONNX Runtime)

**Files to Modify**:
- `internal/service/audio_detector.go`
- NEW: `internal/ml/audio_classifier.go`

**Estimated Complexity**: Very High

---

## Tier 4: Code Quality and Bug Fixes

---

### Story 4.1: Fix Sentence Variance Threshold

**Title**: Adjust sentence length CV threshold based on research

**As a** Manuscript user
**I want** accurate sentence variance scoring
**So that** human text isn't incorrectly flagged as AI

**Background**:
Current threshold (`text_analyzer.go:225`) uses CV of 0.8 as the normalization point. Research suggests CV > 0.5 indicates human-like variance, making 0.8 too aggressive.

**Acceptance Criteria**:
- [ ] Change CV normalization from 0.8 to 0.5
- [ ] Add configurable threshold
- [ ] Update tests with research-backed thresholds
- [ ] Document the research basis for thresholds

**Current Code**:
```go
aiScore := 1.0 - math.Min(cv/0.8, 1.0)  // Line 225
```

**Fixed Code**:
```go
aiScore := 1.0 - math.Min(cv/0.5, 1.0)  // More aligned with research
```

**Files to Modify**:
- `internal/service/text_analyzer.go`
- `internal/service/text_analyzer_test.go`

**Estimated Complexity**: Low

---

### Story 4.2: Fix Image Color Distribution Analysis

**Title**: Implement actual color histogram instead of byte entropy

**As a** Manuscript user
**I want** color distribution analysis based on actual decoded colors
**So that** the image forensics are meaningful

**Background**:
Current implementation (`image_analyzer.go:340-386`) computes entropy of raw bytes, which doesn't reflect actual color distribution in the image. This needs to either decode the image or at least analyze relevant color data sections.

**Acceptance Criteria**:
- [ ] For JPEG: Focus entropy analysis on DCT coefficient sections only
- [ ] For PNG: Decode IDAT chunks to analyze actual pixel data
- [ ] Add option to fully decode for proper RGB histogram
- [ ] Update scoring to reflect actual color distribution metrics

**Files to Modify**:
- `internal/service/image_analyzer.go`
- `internal/service/image_analyzer_test.go`

**Estimated Complexity**: Medium

---

### Story 4.3: Improve Audio Pattern Analysis

**Title**: Replace byte entropy with meaningful audio metrics

**As a** Manuscript user
**I want** audio pattern analysis that reflects actual audio characteristics
**So that** the analysis provides meaningful detection signals

**Background**:
Current `analyzePatterns()` (`audio_analyzer.go:475-517`) computes entropy variance of compressed audio bytes, which doesn't correlate with audio characteristics.

**Acceptance Criteria**:
- [ ] When decoding disabled: Focus on frame-level analysis for MP3 (frame headers contain bitrate info)
- [ ] When decoding enabled: Use actual spectral features
- [ ] Remove or deprecate meaningless byte entropy analysis
- [ ] Document limitations of container-only analysis

**Files to Modify**:
- `internal/service/audio_analyzer.go`
- `internal/service/audio_analyzer_test.go`

**Estimated Complexity**: Medium

---

### Story 4.4: Improve Video Temporal Pattern Analysis

**Title**: Replace arbitrary byte comparison with frame-based analysis

**As a** Manuscript user
**I want** video temporal analysis based on actual frame data
**So that** the detection reflects real video characteristics

**Background**:
Current `analyzeTemporalPattern()` (`video_analyzer.go:454-498`) compares arbitrary byte regions, which doesn't reflect actual video content.

**Acceptance Criteria**:
- [ ] Improve to analyze I-frame positions and sizes
- [ ] Check GOP (Group of Pictures) structure consistency
- [ ] Analyze bitrate variation between frames
- [ ] When frame extraction enabled: Use actual frame comparison

**Files to Modify**:
- `internal/service/video_analyzer.go`
- `internal/service/video_analyzer_test.go`

**Estimated Complexity**: Medium

---

### Story 4.5: Add Comprehensive Test Suite with Ground Truth Data

**Title**: Create test fixtures with known AI and human content

**As a** Manuscript developer
**I want** a comprehensive test suite with labeled AI/human content
**So that** detection accuracy can be measured and regressions prevented

**Acceptance Criteria**:
- [ ] Create test fixtures directory with labeled samples
- [ ] Add text samples: 50 human, 50 AI (from various models)
- [ ] Add image samples: 20 real photos, 20 AI-generated
- [ ] Add audio samples: 10 real recordings, 10 synthetic
- [ ] Add video samples: 5 real, 5 AI-generated
- [ ] Create accuracy benchmark tests
- [ ] Add CI job to run accuracy benchmarks
- [ ] Document expected accuracy ranges

**Files to Modify**:
- NEW: `testdata/text/human/*.txt`
- NEW: `testdata/text/ai/*.txt`
- NEW: `testdata/images/human/*.jpg`
- NEW: `testdata/images/ai/*.png`
- NEW: `internal/service/accuracy_test.go`
- `.github/workflows/ci.yml`

**Estimated Complexity**: Medium

---

### Story 4.6: Add Configurable Signal Weights

**Title**: Make detection signal weights configurable via environment or API

**As a** Manuscript operator
**I want** to tune signal weights without code changes
**So that** detection can be optimized for specific use cases

**Acceptance Criteria**:
- [ ] Add weight configuration via environment variables
- [ ] Add weight configuration via config file
- [ ] Add API endpoint to view/modify weights (optional)
- [ ] Add validation for weight ranges
- [ ] Document tuning guidance

**Example Configuration**:
```yaml
text_weights:
  sentence_variance: 0.15
  vocabulary_richness: 0.20
  burstiness: 0.10
  punctuation_variety: 0.10
  ai_phrases: 0.20
  word_length_variance: 0.05
  contractions: 0.10
  repetition: 0.10
```

**Files to Modify**:
- `internal/config/config.go`
- `internal/service/text_analyzer.go`
- `internal/service/image_analyzer.go`
- `internal/service/audio_analyzer.go`
- `internal/service/video_analyzer.go`

**Estimated Complexity**: Medium

---

## Priority Matrix

| Story | Impact | Effort | Priority |
|-------|--------|--------|----------|
| 4.1 Fix Sentence Variance | Medium | Low | P1 - Quick Win |
| 1.4 Expand AI Phrases | High | Medium | P1 - High Value |
| 1.3 Readability Scores | Medium | Low | P1 - Quick Win |
| 1.2 N-gram Analysis | Medium | Low-Med | P2 |
| 1.6 Full EXIF Parsing | Medium | Medium | P2 |
| 4.5 Test Suite | High | Medium | P2 - Foundation |
| 1.5 DCT Analysis | High | High | P2 |
| 4.6 Configurable Weights | Medium | Medium | P2 |
| 1.1 Tokenization | Medium | Medium | P3 |
| 1.7 PNG Chunk Analysis | Medium | Medium | P3 |
| 4.2 Fix Color Analysis | Medium | Medium | P3 |
| 4.3 Fix Audio Patterns | Low | Medium | P3 |
| 4.4 Fix Video Patterns | Low | Medium | P3 |
| 2.1 Audio Decoding | High | High | P3 |
| 2.2 FFT Analysis | High | High | P3 |
| 2.3 MFCC | High | High | P3 |
| 2.4 Video Frames | High | Very High | P4 |
| 2.5 Temporal Analysis | Medium | High | P4 |
| 3.1 ONNX Runtime | High | High | P4 - Enables ML |
| 3.2 Local Perplexity | Very High | Very High | P4 |
| 3.3 CLIP Encoder | Very High | Very High | P4 |
| 3.4 Audio Classifier | High | Very High | P4 |

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- Story 4.1: Fix Sentence Variance
- Story 1.3: Readability Scores
- Story 1.4: Expand AI Phrases (initial 200 patterns)

### Phase 2: Core Improvements (2-4 weeks)
- Story 4.5: Test Suite
- Story 1.2: N-gram Analysis
- Story 1.6: Full EXIF Parsing
- Story 4.6: Configurable Weights

### Phase 3: Advanced Heuristics (4-6 weeks)
- Story 1.5: DCT Analysis
- Story 1.7: PNG Chunks
- Story 1.1: Tokenization
- Story 4.2-4.4: Bug Fixes

### Phase 4: Audio/Video Deep Analysis (6-10 weeks)
- Story 2.1: Audio Decoding
- Story 2.2: FFT Analysis
- Story 2.3: MFCC
- Story 2.4-2.5: Video Frames

### Phase 5: ML Integration (10+ weeks)
- Story 3.1: ONNX Runtime
- Story 3.2: Local Perplexity
- Story 3.3: CLIP Encoder
- Story 3.4: Audio Classifier
