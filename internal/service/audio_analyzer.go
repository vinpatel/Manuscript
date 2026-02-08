package service

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/cmplx"
)

// =============================================================================
// Manuscript Audio Detection Algorithm
// =============================================================================
//
// Audio AI detection focuses on identifying synthetic speech and AI-generated music.
//
// Key signals:
//   1. Format and codec metadata
//   2. Audio characteristics (sampling rate, channels, bitrate)
//   3. Spectral patterns (via byte analysis as proxy)
//   4. Silence/noise patterns
//   5. Known AI audio signatures
//   6. Recording device markers
//
// Note: Full spectral analysis requires decoding. Our analysis works on
// container metadata and compressed byte patterns.
//
// =============================================================================

// AudioAnalyzer performs forensic analysis on audio files.
type AudioAnalyzer struct {
	weights AudioAnalyzerWeights
}

// AudioAnalyzerWeights controls signal importance.
type AudioAnalyzerWeights struct {
	MetadataScore     float64
	FormatAnalysis    float64
	PatternAnalysis   float64
	QualityIndicators float64
	AISignatures      float64
	NoiseProfile      float64
}

// DefaultAudioWeights returns tuned weights.
func DefaultAudioWeights() AudioAnalyzerWeights {
	return AudioAnalyzerWeights{
		MetadataScore:     0.25,
		FormatAnalysis:    0.15,
		PatternAnalysis:   0.20,
		QualityIndicators: 0.15,
		AISignatures:      0.15,
		NoiseProfile:      0.10,
	}
}

// NewAudioAnalyzer creates a new analyzer.
func NewAudioAnalyzer() *AudioAnalyzer {
	return &AudioAnalyzer{
		weights: DefaultAudioWeights(),
	}
}

// AudioAnalysisResult contains analysis results.
type AudioAnalysisResult struct {
	AIScore  float64
	Signals  AudioSignals
	Metadata AudioMetadata
	Stats    AudioStats
}

// AudioSignals contains individual signal scores.
type AudioSignals struct {
	MetadataScore     float64 // Missing/fake metadata = AI-like
	FormatAnalysis    float64 // Unusual format = suspicious
	PatternAnalysis   float64 // Unusual patterns = AI-like
	QualityIndicators float64 // Unusual quality = suspicious
	AISignatures      float64 // Known AI markers = AI
	NoiseProfile      float64 // Unnatural noise = AI-like
}

// AudioMetadata contains extracted metadata.
type AudioMetadata struct {
	Format       string // mp3, wav, flac, ogg, etc.
	SampleRate   int    // Hz
	Channels     int    // 1=mono, 2=stereo
	BitDepth     int    // bits per sample
	Bitrate      int    // kbps (for compressed formats)
	Duration     float64 // seconds (estimated)
	HasID3       bool
	Artist       string
	Title        string
	EncoderName  string
	IsAIMarked   bool
	HasRecording bool // Markers of real recording
}

// AudioStats contains audio statistics.
type AudioStats struct {
	FileSize        int64
	EstimatedFrames int
	SilenceRatio    float64 // Proportion of silence
	PeakAmplitude   float64 // Normalized 0-1
	DynamicRange    float64 // Difference between loud/quiet

	// Waveform statistics (populated when PCM data available)
	HasWaveformStats  bool
	RMSAmplitude      float64 // Root Mean Square amplitude
	ZeroCrossingRate  float64 // Rate of zero crossings per sample
	SpectralCentroid  float64 // Weighted mean of frequencies (Hz)
	SpectralRolloff   float64 // Frequency below which 85% of energy lies
	Crest             float64 // Peak to RMS ratio (dynamics indicator)
}

// WaveformData represents decoded PCM audio data.
type WaveformData struct {
	Samples    []float64 // Normalized samples (-1 to 1)
	SampleRate int
	Channels   int
	BitDepth   int
}

// Analyze performs forensic analysis on audio data.
func (a *AudioAnalyzer) Analyze(data []byte) AudioAnalysisResult {
	result := AudioAnalysisResult{}
	result.Stats.FileSize = int64(len(data))

	// Detect format
	format := a.detectAudioFormat(data)
	result.Metadata.Format = format

	// Extract metadata based on format
	switch format {
	case "mp3":
		result.Metadata, result.Stats = a.analyzeMP3(data)
	case "wav":
		result.Metadata, result.Stats = a.analyzeWAV(data)
	case "flac":
		result.Metadata, result.Stats = a.analyzeFLAC(data)
	case "ogg":
		result.Metadata, result.Stats = a.analyzeOGG(data)
	case "m4a", "aac":
		result.Metadata, result.Stats = a.analyzeM4A(data)
	default:
		result.Metadata.Format = format
	}

	// Calculate signals
	result.Signals.MetadataScore = a.analyzeMetadata(result.Metadata)
	result.Signals.FormatAnalysis = a.analyzeFormat(result.Metadata, data)
	result.Signals.PatternAnalysis = a.analyzePatterns(data, format)
	result.Signals.QualityIndicators = a.analyzeQuality(result.Metadata, result.Stats)
	result.Signals.AISignatures = a.detectAISignatures(data, result.Metadata)
	result.Signals.NoiseProfile = a.analyzeNoiseProfile(data, format)

	// Calculate weighted score
	result.AIScore = a.calculateWeightedScore(result.Signals)

	return result
}

// detectAudioFormat identifies audio format from magic bytes.
func (a *AudioAnalyzer) detectAudioFormat(data []byte) string {
	if len(data) < 12 {
		return "unknown"
	}

	// MP3: FF FB, FF FA, FF F3, FF F2 (frame sync) or ID3 tag
	if data[0] == 0xFF && (data[1]&0xE0) == 0xE0 {
		return "mp3"
	}
	if data[0] == 'I' && data[1] == 'D' && data[2] == '3' {
		return "mp3"
	}

	// WAV: RIFF....WAVE
	if bytes.Equal(data[0:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WAVE")) {
		return "wav"
	}

	// FLAC: fLaC
	if bytes.Equal(data[0:4], []byte("fLaC")) {
		return "flac"
	}

	// OGG: OggS
	if bytes.Equal(data[0:4], []byte("OggS")) {
		return "ogg"
	}

	// M4A/AAC: ftyp M4A or similar
	if len(data) >= 8 && bytes.Equal(data[4:8], []byte("ftyp")) {
		if len(data) >= 12 {
			brand := string(data[8:12])
			if brand == "M4A " || brand == "mp42" || brand == "isom" {
				return "m4a"
			}
		}
		return "m4a"
	}

	// AAC ADTS: sync word 0xFFF
	if data[0] == 0xFF && (data[1]&0xF0) == 0xF0 {
		return "aac"
	}

	return "unknown"
}

// analyzeMP3 extracts metadata from MP3 files.
func (a *AudioAnalyzer) analyzeMP3(data []byte) (AudioMetadata, AudioStats) {
	meta := AudioMetadata{Format: "mp3"}
	stats := AudioStats{FileSize: int64(len(data))}

	// Check for ID3v2 tag at start
	if len(data) > 10 && data[0] == 'I' && data[1] == 'D' && data[2] == '3' {
		meta.HasID3 = true

		// ID3v2 size is syncsafe integer at bytes 6-9
		id3Size := int(data[6])<<21 | int(data[7])<<14 | int(data[8])<<7 | int(data[9])

		// Look for AI markers in ID3 tags
		if id3Size > 0 && id3Size+10 < len(data) {
			id3Data := string(data[10 : 10+id3Size])
			if containsAIAudioMarker(id3Data) {
				meta.IsAIMarked = true
			}
			meta.EncoderName = extractAudioEncoder(id3Data)

			// Check for recording indicators
			if containsRecordingMarker(id3Data) {
				meta.HasRecording = true
			}
		}
	}

	// Find first MP3 frame to get audio params
	for i := 0; i < len(data)-4; i++ {
		if data[i] == 0xFF && (data[i+1]&0xE0) == 0xE0 {
			// Found frame sync
			header := binary.BigEndian.Uint32(data[i : i+4])

			// Extract bitrate index and sample rate index
			version := (header >> 19) & 0x03
			layer := (header >> 17) & 0x03
			bitrateIdx := (header >> 12) & 0x0F
			srIdx := (header >> 10) & 0x03
			channelMode := (header >> 6) & 0x03

			// Set sample rate (simplified - MPEG1 Layer 3)
			sampleRates := []int{44100, 48000, 32000, 0}
			if srIdx < 3 {
				meta.SampleRate = sampleRates[srIdx]
			}

			// Set channels
			if channelMode == 3 {
				meta.Channels = 1 // Mono
			} else {
				meta.Channels = 2 // Stereo
			}

			// Bitrate table for MPEG1 Layer 3
			bitrates := []int{0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0}
			if bitrateIdx < 15 {
				meta.Bitrate = bitrates[bitrateIdx]
			}

			// Avoid unused variable warnings
			_ = version
			_ = layer

			break
		}
	}

	// Check ID3v1 tag at end
	if len(data) >= 128 {
		if bytes.Equal(data[len(data)-128:len(data)-125], []byte("TAG")) {
			meta.HasID3 = true
		}
	}

	return meta, stats
}

// analyzeWAV extracts metadata from WAV files.
func (a *AudioAnalyzer) analyzeWAV(data []byte) (AudioMetadata, AudioStats) {
	meta := AudioMetadata{Format: "wav"}
	stats := AudioStats{FileSize: int64(len(data))}

	if len(data) < 44 {
		return meta, stats
	}

	// Parse WAV header
	var audioFormat uint16
	var dataOffset, dataSize int

	// fmt chunk starts at byte 12
	if bytes.Equal(data[12:16], []byte("fmt ")) {
		// Audio format at 20-21 (1 = PCM)
		audioFormat = binary.LittleEndian.Uint16(data[20:22])
		if audioFormat == 1 {
			meta.EncoderName = "PCM"
		}

		// Channels at 22-23
		meta.Channels = int(binary.LittleEndian.Uint16(data[22:24]))

		// Sample rate at 24-27
		meta.SampleRate = int(binary.LittleEndian.Uint32(data[24:28]))

		// Bits per sample at 34-35
		meta.BitDepth = int(binary.LittleEndian.Uint16(data[34:36]))
	}

	// Find data chunk and metadata chunks
	for i := 12; i < len(data)-8; i++ {
		chunkID := string(data[i : i+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[i+4 : i+8]))

		if chunkSize <= 0 || i+8+chunkSize > len(data) {
			break
		}

		switch chunkID {
		case "data":
			dataOffset = i + 8
			dataSize = chunkSize
		case "LIST":
			listData := string(data[i+8 : i+8+chunkSize])
			if containsAIAudioMarker(listData) {
				meta.IsAIMarked = true
			}
			if containsRecordingMarker(listData) {
				meta.HasRecording = true
			}
		}

		i += 8 + chunkSize - 1 // -1 because loop increments
	}

	// Extract PCM data and compute waveform statistics if available
	if audioFormat == 1 && dataOffset > 0 && dataSize > 0 && meta.BitDepth > 0 {
		waveform := a.extractWAVPCM(data[dataOffset:dataOffset+dataSize], meta.BitDepth, meta.Channels)
		if len(waveform.Samples) > 0 {
			waveform.SampleRate = meta.SampleRate
			stats = a.computeWaveformStats(waveform, stats)

			// Estimate duration
			if meta.SampleRate > 0 && meta.Channels > 0 {
				totalSamples := len(waveform.Samples)
				meta.Duration = float64(totalSamples) / float64(meta.SampleRate*meta.Channels)
			}
		}
	}

	return meta, stats
}

// extractWAVPCM extracts PCM samples from WAV data chunk.
func (a *AudioAnalyzer) extractWAVPCM(data []byte, bitDepth, channels int) WaveformData {
	waveform := WaveformData{
		BitDepth: bitDepth,
		Channels: channels,
	}

	bytesPerSample := bitDepth / 8
	if bytesPerSample <= 0 || bytesPerSample > 4 {
		return waveform
	}

	numSamples := len(data) / bytesPerSample
	// Limit to prevent excessive memory usage
	maxSamples := 10000000 // 10M samples max
	if numSamples > maxSamples {
		numSamples = maxSamples
	}

	waveform.Samples = make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		offset := i * bytesPerSample
		if offset+bytesPerSample > len(data) {
			break
		}

		var sample float64
		switch bitDepth {
		case 8:
			// 8-bit unsigned
			sample = (float64(data[offset]) - 128) / 128.0
		case 16:
			// 16-bit signed little-endian
			val := int16(binary.LittleEndian.Uint16(data[offset : offset+2]))
			sample = float64(val) / 32768.0
		case 24:
			// 24-bit signed little-endian
			val := int32(data[offset]) | int32(data[offset+1])<<8 | int32(data[offset+2])<<16
			if val&0x800000 != 0 {
				val |= ^0xFFFFFF // Sign extend
			}
			sample = float64(val) / 8388608.0
		case 32:
			// 32-bit signed or float
			val := int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
			sample = float64(val) / 2147483648.0
		}

		waveform.Samples[i] = sample
	}

	return waveform
}

// computeWaveformStats calculates audio statistics from waveform data.
func (a *AudioAnalyzer) computeWaveformStats(waveform WaveformData, stats AudioStats) AudioStats {
	if len(waveform.Samples) == 0 {
		return stats
	}

	stats.HasWaveformStats = true

	// Calculate RMS amplitude
	sumSquares := 0.0
	peak := 0.0
	zeroCrossings := 0
	var prevSample float64

	for i, sample := range waveform.Samples {
		sumSquares += sample * sample

		absSample := math.Abs(sample)
		if absSample > peak {
			peak = absSample
		}

		// Count zero crossings
		if i > 0 && ((prevSample >= 0 && sample < 0) || (prevSample < 0 && sample >= 0)) {
			zeroCrossings++
		}
		prevSample = sample
	}

	stats.RMSAmplitude = math.Sqrt(sumSquares / float64(len(waveform.Samples)))
	stats.PeakAmplitude = peak
	stats.ZeroCrossingRate = float64(zeroCrossings) / float64(len(waveform.Samples))

	// Crest factor (peak to RMS ratio) - high for impulsive sounds
	if stats.RMSAmplitude > 0 {
		stats.Crest = peak / stats.RMSAmplitude
	}

	// Compute spectral centroid using simple DFT approximation
	stats.SpectralCentroid = a.computeSpectralCentroid(waveform)
	stats.SpectralRolloff = a.computeSpectralRolloff(waveform)

	// Estimate silence ratio
	silenceThreshold := 0.01
	silentSamples := 0
	for _, sample := range waveform.Samples {
		if math.Abs(sample) < silenceThreshold {
			silentSamples++
		}
	}
	stats.SilenceRatio = float64(silentSamples) / float64(len(waveform.Samples))

	// Dynamic range (simplified)
	if stats.RMSAmplitude > 0 {
		stats.DynamicRange = 20 * math.Log10(peak/stats.RMSAmplitude)
	}

	return stats
}

// computeSpectralCentroid calculates the spectral centroid using time-domain approximation.
// True spectral centroid requires FFT, but ZCR correlates with perceived brightness.
func (a *AudioAnalyzer) computeSpectralCentroid(waveform WaveformData) float64 {
	if len(waveform.Samples) < 1000 || waveform.SampleRate == 0 {
		return 0
	}

	// Use zero-crossing rate as proxy for spectral centroid
	// Higher ZCR generally correlates with higher frequency content
	zeroCrossings := 0
	for i := 1; i < len(waveform.Samples); i++ {
		if (waveform.Samples[i-1] >= 0 && waveform.Samples[i] < 0) ||
			(waveform.Samples[i-1] < 0 && waveform.Samples[i] >= 0) {
			zeroCrossings++
		}
	}

	// Approximate frequency = ZCR * SampleRate / 2
	zcr := float64(zeroCrossings) / float64(len(waveform.Samples))
	approximateFreq := zcr * float64(waveform.SampleRate) / 2.0

	return approximateFreq
}

// computeSpectralRolloff estimates the frequency below which 85% of energy lies.
// This is an approximation based on energy distribution in time domain.
func (a *AudioAnalyzer) computeSpectralRolloff(waveform WaveformData) float64 {
	if len(waveform.Samples) < 1000 || waveform.SampleRate == 0 {
		return 0
	}

	// Simple energy-based estimation
	// Count high-frequency energy using derivative
	totalEnergy := 0.0
	highFreqEnergy := 0.0

	for i := 1; i < len(waveform.Samples); i++ {
		sample := waveform.Samples[i]
		totalEnergy += sample * sample

		// High-frequency energy from sample differences (like high-pass filter)
		diff := waveform.Samples[i] - waveform.Samples[i-1]
		highFreqEnergy += diff * diff
	}

	if totalEnergy == 0 {
		return 0
	}

	// Estimate rolloff based on high/low frequency ratio
	highFreqRatio := highFreqEnergy / totalEnergy

	// Map ratio to approximate frequency
	// Higher ratio = more high frequency content
	nyquist := float64(waveform.SampleRate) / 2.0
	rolloff := nyquist * (0.2 + 0.6*highFreqRatio) // Scale between 20-80% of Nyquist

	return math.Min(rolloff, nyquist)
}

// =============================================================================
// FFT-Based Spectral Analysis
// =============================================================================

// SpectralAnalysis contains FFT-based spectral features.
type SpectralAnalysis struct {
	SpectralCentroid  float64   // Center of mass of spectrum (Hz)
	SpectralFlatness  float64   // Geometric/arithmetic mean ratio (0=tonal, 1=noisy)
	SpectralRolloff   float64   // Frequency below which 85% of energy (Hz)
	SpectralFlux      float64   // Spectral change over time
	HarmonicRatio     float64   // Ratio of harmonic to non-harmonic energy
	FormantFreqs      []float64 // Detected formant frequencies (Hz)
	Bandwidth         float64   // Spectral bandwidth (Hz)
	MagnitudeSpectrum []float64 // Magnitude spectrum (for MFCC computation)
}

// AnalyzeSpectrum performs FFT-based spectral analysis on audio.
func (a *AudioAnalyzer) AnalyzeSpectrum(waveform WaveformData) SpectralAnalysis {
	result := SpectralAnalysis{}

	if len(waveform.Samples) < 512 || waveform.SampleRate == 0 {
		return result
	}

	// Use 4096-sample FFT for good frequency resolution
	fftSize := 4096
	if len(waveform.Samples) < fftSize {
		// Use smaller power of 2
		fftSize = 1
		for fftSize < len(waveform.Samples) {
			fftSize *= 2
		}
		fftSize /= 2
	}

	if fftSize < 256 {
		return result
	}

	// Apply Hann window to reduce spectral leakage
	windowed := a.applyHannWindow(waveform.Samples[:fftSize])

	// Compute FFT
	spectrum := a.fft(windowed)

	// Calculate magnitude spectrum (only positive frequencies)
	numBins := fftSize / 2
	magnitudes := make([]float64, numBins)
	for i := 0; i < numBins; i++ {
		magnitudes[i] = cmplx.Abs(spectrum[i])
	}
	result.MagnitudeSpectrum = magnitudes

	// Frequency resolution
	freqRes := float64(waveform.SampleRate) / float64(fftSize)

	// Compute spectral features
	result.SpectralCentroid = a.computeFFTSpectralCentroid(magnitudes, freqRes)
	result.SpectralFlatness = a.computeSpectralFlatness(magnitudes)
	result.SpectralRolloff = a.computeFFTSpectralRolloff(magnitudes, freqRes, 0.85)
	result.Bandwidth = a.computeSpectralBandwidth(magnitudes, freqRes, result.SpectralCentroid)
	result.HarmonicRatio = a.estimateHarmonicRatio(magnitudes, freqRes)
	result.FormantFreqs = a.detectFormants(magnitudes, freqRes, waveform.SampleRate)

	// Compute spectral flux if we have enough data for multiple frames
	if len(waveform.Samples) >= fftSize*2 {
		result.SpectralFlux = a.computeSpectralFlux(waveform, fftSize)
	}

	return result
}

// applyHannWindow applies a Hann window to reduce spectral leakage.
func (a *AudioAnalyzer) applyHannWindow(samples []float64) []float64 {
	n := len(samples)
	windowed := make([]float64, n)

	for i := 0; i < n; i++ {
		// Hann window: 0.5 * (1 - cos(2*pi*i/(n-1)))
		w := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
		windowed[i] = samples[i] * w
	}

	return windowed
}

// fft implements the Cooley-Tukey FFT algorithm.
func (a *AudioAnalyzer) fft(x []float64) []complex128 {
	n := len(x)

	// Convert to complex
	c := make([]complex128, n)
	for i, v := range x {
		c[i] = complex(v, 0)
	}

	return a.fftRecursive(c)
}

// fftRecursive performs FFT using Cooley-Tukey algorithm.
func (a *AudioAnalyzer) fftRecursive(x []complex128) []complex128 {
	n := len(x)

	if n <= 1 {
		return x
	}

	// Divide: separate even and odd indices
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)
	for i := 0; i < n/2; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	// Conquer: recursive FFT
	evenFFT := a.fftRecursive(even)
	oddFFT := a.fftRecursive(odd)

	// Combine
	result := make([]complex128, n)
	for k := 0; k < n/2; k++ {
		// Twiddle factor: e^(-2*pi*i*k/n)
		angle := -2 * math.Pi * float64(k) / float64(n)
		twiddle := complex(math.Cos(angle), math.Sin(angle))

		result[k] = evenFFT[k] + twiddle*oddFFT[k]
		result[k+n/2] = evenFFT[k] - twiddle*oddFFT[k]
	}

	return result
}

// computeFFTSpectralCentroid calculates the spectral centroid from FFT magnitudes.
func (a *AudioAnalyzer) computeFFTSpectralCentroid(magnitudes []float64, freqRes float64) float64 {
	weightedSum := 0.0
	totalMag := 0.0

	for i, mag := range magnitudes {
		freq := float64(i) * freqRes
		weightedSum += freq * mag
		totalMag += mag
	}

	if totalMag == 0 {
		return 0
	}

	return weightedSum / totalMag
}

// computeSpectralFlatness calculates the ratio of geometric to arithmetic mean.
// Values close to 1 indicate noise-like signal; close to 0 indicates tonal.
func (a *AudioAnalyzer) computeSpectralFlatness(magnitudes []float64) float64 {
	if len(magnitudes) == 0 {
		return 0
	}

	// Avoid log of zero
	epsilon := 1e-10
	logSum := 0.0
	arithmeticSum := 0.0

	for _, mag := range magnitudes {
		if mag < epsilon {
			mag = epsilon
		}
		logSum += math.Log(mag)
		arithmeticSum += mag
	}

	n := float64(len(magnitudes))
	geometricMean := math.Exp(logSum / n)
	arithmeticMean := arithmeticSum / n

	if arithmeticMean == 0 {
		return 0
	}

	return geometricMean / arithmeticMean
}

// computeFFTSpectralRolloff finds frequency below which given percentage of energy lies.
func (a *AudioAnalyzer) computeFFTSpectralRolloff(magnitudes []float64, freqRes float64, threshold float64) float64 {
	totalEnergy := 0.0
	for _, mag := range magnitudes {
		totalEnergy += mag * mag
	}

	if totalEnergy == 0 {
		return 0
	}

	targetEnergy := totalEnergy * threshold
	cumulativeEnergy := 0.0

	for i, mag := range magnitudes {
		cumulativeEnergy += mag * mag
		if cumulativeEnergy >= targetEnergy {
			return float64(i) * freqRes
		}
	}

	return float64(len(magnitudes)-1) * freqRes
}

// computeSpectralBandwidth calculates the variance of spectrum around centroid.
func (a *AudioAnalyzer) computeSpectralBandwidth(magnitudes []float64, freqRes float64, centroid float64) float64 {
	if len(magnitudes) == 0 || centroid == 0 {
		return 0
	}

	weightedSum := 0.0
	totalMag := 0.0

	for i, mag := range magnitudes {
		freq := float64(i) * freqRes
		diff := freq - centroid
		weightedSum += diff * diff * mag
		totalMag += mag
	}

	if totalMag == 0 {
		return 0
	}

	return math.Sqrt(weightedSum / totalMag)
}

// estimateHarmonicRatio estimates the ratio of harmonic to non-harmonic energy.
// High ratio indicates more tonal/harmonic content (typical for real speech).
func (a *AudioAnalyzer) estimateHarmonicRatio(magnitudes []float64, freqRes float64) float64 {
	if len(magnitudes) < 50 {
		return 0
	}

	// Find fundamental frequency (strongest low-frequency peak)
	// Look in typical voice range: 80-400 Hz
	minBin := int(80.0 / freqRes)
	maxBin := int(400.0 / freqRes)

	if minBin >= len(magnitudes) || maxBin >= len(magnitudes) {
		return 0
	}

	// Find fundamental frequency
	f0Bin := minBin
	f0Mag := magnitudes[minBin]
	for i := minBin; i <= maxBin && i < len(magnitudes); i++ {
		if magnitudes[i] > f0Mag {
			f0Mag = magnitudes[i]
			f0Bin = i
		}
	}

	if f0Mag == 0 {
		return 0
	}

	// Sum energy at harmonics (2f0, 3f0, 4f0, ...)
	harmonicEnergy := f0Mag * f0Mag
	totalEnergy := 0.0
	for _, mag := range magnitudes {
		totalEnergy += mag * mag
	}

	// Check harmonics up to 10th
	for h := 2; h <= 10; h++ {
		harmonicBin := f0Bin * h
		if harmonicBin >= len(magnitudes) {
			break
		}

		// Look in a small window around expected harmonic
		window := 3
		start := harmonicBin - window
		end := harmonicBin + window
		if start < 0 {
			start = 0
		}
		if end >= len(magnitudes) {
			end = len(magnitudes) - 1
		}

		// Find peak in window
		peakMag := 0.0
		for i := start; i <= end; i++ {
			if magnitudes[i] > peakMag {
				peakMag = magnitudes[i]
			}
		}
		harmonicEnergy += peakMag * peakMag
	}

	if totalEnergy == 0 {
		return 0
	}

	return harmonicEnergy / totalEnergy
}

// detectFormants finds formant frequencies (resonant peaks in spectrum).
// Formants are key for speech - AI often has unnatural formant patterns.
func (a *AudioAnalyzer) detectFormants(magnitudes []float64, freqRes float64, sampleRate int) []float64 {
	var formants []float64

	if len(magnitudes) < 100 {
		return formants
	}

	// Compute spectral envelope using peak picking
	// Look in typical formant range: 200-4000 Hz
	minBin := int(200.0 / freqRes)
	maxBin := int(4000.0 / freqRes)

	if minBin >= len(magnitudes) {
		return formants
	}
	if maxBin >= len(magnitudes) {
		maxBin = len(magnitudes) - 1
	}

	// Find local peaks (potential formants)
	for i := minBin + 2; i < maxBin-2; i++ {
		// Check if this is a local maximum
		if magnitudes[i] > magnitudes[i-1] && magnitudes[i] > magnitudes[i+1] &&
			magnitudes[i] > magnitudes[i-2] && magnitudes[i] > magnitudes[i+2] {

			// Must be significant peak (above average)
			avgMag := 0.0
			for j := minBin; j <= maxBin; j++ {
				avgMag += magnitudes[j]
			}
			avgMag /= float64(maxBin - minBin + 1)

			if magnitudes[i] > avgMag*1.5 {
				freq := float64(i) * freqRes
				formants = append(formants, freq)

				// Skip nearby bins to avoid detecting same formant twice
				i += 5
			}
		}

		// Limit to first 5 formants
		if len(formants) >= 5 {
			break
		}
	}

	return formants
}

// computeSpectralFlux measures spectral change between frames.
func (a *AudioAnalyzer) computeSpectralFlux(waveform WaveformData, fftSize int) float64 {
	if len(waveform.Samples) < fftSize*2 {
		return 0
	}

	// Compute FFT for two consecutive frames
	hopSize := fftSize / 2
	numFrames := (len(waveform.Samples) - fftSize) / hopSize

	if numFrames < 2 {
		return 0
	}

	// Limit number of frames analyzed
	if numFrames > 50 {
		numFrames = 50
	}

	totalFlux := 0.0
	var prevMagnitudes []float64

	for f := 0; f < numFrames; f++ {
		start := f * hopSize
		end := start + fftSize
		if end > len(waveform.Samples) {
			break
		}

		// Window and FFT
		windowed := a.applyHannWindow(waveform.Samples[start:end])
		spectrum := a.fft(windowed)

		// Compute magnitudes
		numBins := fftSize / 2
		magnitudes := make([]float64, numBins)
		for i := 0; i < numBins; i++ {
			magnitudes[i] = cmplx.Abs(spectrum[i])
		}

		// Compute flux (half-wave rectified difference)
		if prevMagnitudes != nil {
			flux := 0.0
			for i := range magnitudes {
				diff := magnitudes[i] - prevMagnitudes[i]
				if diff > 0 {
					flux += diff * diff
				}
			}
			totalFlux += math.Sqrt(flux)
		}

		prevMagnitudes = magnitudes
	}

	// Normalize by number of comparisons
	if numFrames > 1 {
		return totalFlux / float64(numFrames-1)
	}

	return 0
}

// =============================================================================
// MFCC (Mel-Frequency Cepstral Coefficients) Computation
// =============================================================================

// MFCCConfig contains MFCC computation parameters.
type MFCCConfig struct {
	NumCoeffs   int     // Number of MFCC coefficients (typically 13)
	NumFilters  int     // Number of mel filterbank channels (typically 26)
	FrameSize   int     // Frame size in samples
	FrameStride int     // Hop size in samples
	PreEmphasis float64 // Pre-emphasis coefficient (typically 0.97)
	LowFreq     float64 // Lowest frequency for filterbank (Hz)
	HighFreq    float64 // Highest frequency for filterbank (Hz, 0 = Nyquist)
}

// DefaultMFCCConfig returns standard MFCC parameters.
func DefaultMFCCConfig(sampleRate int) MFCCConfig {
	// 25ms frame, 10ms hop
	frameSize := sampleRate * 25 / 1000
	frameStride := sampleRate * 10 / 1000

	// Round to power of 2 for FFT
	fftSize := 1
	for fftSize < frameSize {
		fftSize *= 2
	}

	return MFCCConfig{
		NumCoeffs:   13,
		NumFilters:  26,
		FrameSize:   fftSize,
		FrameStride: frameStride,
		PreEmphasis: 0.97,
		LowFreq:     0,
		HighFreq:    0, // 0 = Nyquist
	}
}

// MFCCResult contains computed MFCC features and statistics.
type MFCCResult struct {
	Coefficients [][]float64 // MFCCs for each frame [frame][coeff]
	Delta        [][]float64 // First-order derivatives
	DeltaDelta   [][]float64 // Second-order derivatives

	// Statistics over all frames (useful for classification)
	Mean     []float64 // Mean of each coefficient
	Variance []float64 // Variance of each coefficient
	Skewness []float64 // Skewness of each coefficient
	Kurtosis []float64 // Kurtosis of each coefficient
}

// ComputeMFCC extracts MFCC features from audio waveform.
func (a *AudioAnalyzer) ComputeMFCC(waveform WaveformData, config MFCCConfig) MFCCResult {
	result := MFCCResult{}

	if len(waveform.Samples) < config.FrameSize || waveform.SampleRate == 0 {
		return result
	}

	// Set high frequency to Nyquist if not specified
	highFreq := config.HighFreq
	if highFreq <= 0 {
		highFreq = float64(waveform.SampleRate) / 2
	}

	// Build mel filterbank
	filterbank := a.buildMelFilterbank(config.NumFilters, config.FrameSize, waveform.SampleRate, config.LowFreq, highFreq)

	// Pre-emphasis
	emphasized := a.applyPreEmphasis(waveform.Samples, config.PreEmphasis)

	// Process frames
	numFrames := (len(emphasized) - config.FrameSize) / config.FrameStride
	if numFrames < 1 {
		return result
	}

	// Limit frames for performance
	if numFrames > 1000 {
		numFrames = 1000
	}

	result.Coefficients = make([][]float64, numFrames)

	for f := 0; f < numFrames; f++ {
		start := f * config.FrameStride
		end := start + config.FrameSize
		if end > len(emphasized) {
			break
		}

		frame := emphasized[start:end]

		// Apply window
		windowed := a.applyHannWindow(frame)

		// Compute power spectrum
		spectrum := a.fft(windowed)
		powerSpectrum := make([]float64, config.FrameSize/2)
		for i := range powerSpectrum {
			mag := cmplx.Abs(spectrum[i])
			powerSpectrum[i] = mag * mag
		}

		// Apply mel filterbank
		melEnergies := a.applyFilterbank(powerSpectrum, filterbank)

		// Log compression
		for i := range melEnergies {
			if melEnergies[i] < 1e-10 {
				melEnergies[i] = 1e-10
			}
			melEnergies[i] = math.Log(melEnergies[i])
		}

		// DCT to get MFCCs
		mfccs := a.dct(melEnergies, config.NumCoeffs)
		result.Coefficients[f] = mfccs
	}

	// Compute delta and delta-delta coefficients
	result.Delta = a.computeDeltas(result.Coefficients)
	result.DeltaDelta = a.computeDeltas(result.Delta)

	// Compute statistics
	result.Mean, result.Variance, result.Skewness, result.Kurtosis = a.computeMFCCStats(result.Coefficients)

	return result
}

// hzToMel converts frequency in Hz to mel scale.
func hzToMel(hz float64) float64 {
	return 2595 * math.Log10(1+hz/700)
}

// melToHz converts mel scale to Hz.
func melToHz(mel float64) float64 {
	return 700 * (math.Pow(10, mel/2595) - 1)
}

// buildMelFilterbank creates triangular mel-spaced filterbank.
func (a *AudioAnalyzer) buildMelFilterbank(numFilters, fftSize, sampleRate int, lowFreq, highFreq float64) [][]float64 {
	numBins := fftSize / 2

	// Convert frequency bounds to mel scale
	melLow := hzToMel(lowFreq)
	melHigh := hzToMel(highFreq)

	// Create equally spaced mel points
	melPoints := make([]float64, numFilters+2)
	melStep := (melHigh - melLow) / float64(numFilters+1)
	for i := range melPoints {
		melPoints[i] = melLow + float64(i)*melStep
	}

	// Convert back to Hz and then to FFT bins
	binPoints := make([]int, numFilters+2)
	freqRes := float64(sampleRate) / float64(fftSize)
	for i, mel := range melPoints {
		hz := melToHz(mel)
		binPoints[i] = int(hz / freqRes)
		if binPoints[i] >= numBins {
			binPoints[i] = numBins - 1
		}
	}

	// Create triangular filters
	filterbank := make([][]float64, numFilters)
	for f := 0; f < numFilters; f++ {
		filterbank[f] = make([]float64, numBins)

		left := binPoints[f]
		center := binPoints[f+1]
		right := binPoints[f+2]

		// Rising edge
		for b := left; b < center; b++ {
			if center > left {
				filterbank[f][b] = float64(b-left) / float64(center-left)
			}
		}

		// Falling edge
		for b := center; b < right; b++ {
			if right > center {
				filterbank[f][b] = float64(right-b) / float64(right-center)
			}
		}
	}

	return filterbank
}

// applyPreEmphasis applies first-order high-pass filter.
func (a *AudioAnalyzer) applyPreEmphasis(samples []float64, coeff float64) []float64 {
	emphasized := make([]float64, len(samples))
	emphasized[0] = samples[0]
	for i := 1; i < len(samples); i++ {
		emphasized[i] = samples[i] - coeff*samples[i-1]
	}
	return emphasized
}

// applyFilterbank multiplies power spectrum by mel filterbank.
func (a *AudioAnalyzer) applyFilterbank(powerSpectrum []float64, filterbank [][]float64) []float64 {
	numFilters := len(filterbank)
	melEnergies := make([]float64, numFilters)

	for f := 0; f < numFilters; f++ {
		energy := 0.0
		for b, power := range powerSpectrum {
			if b < len(filterbank[f]) {
				energy += power * filterbank[f][b]
			}
		}
		melEnergies[f] = energy
	}

	return melEnergies
}

// dct computes Discrete Cosine Transform (Type-II) for MFCC.
func (a *AudioAnalyzer) dct(input []float64, numCoeffs int) []float64 {
	n := len(input)
	output := make([]float64, numCoeffs)

	// DCT-II: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1)/(2N))
	for k := 0; k < numCoeffs; k++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			angle := math.Pi * float64(k) * (2*float64(i) + 1) / (2 * float64(n))
			sum += input[i] * math.Cos(angle)
		}

		// Normalization
		if k == 0 {
			output[k] = sum * math.Sqrt(1.0/float64(n))
		} else {
			output[k] = sum * math.Sqrt(2.0/float64(n))
		}
	}

	return output
}

// computeDeltas calculates first-order derivatives of MFCCs.
func (a *AudioAnalyzer) computeDeltas(coefficients [][]float64) [][]float64 {
	numFrames := len(coefficients)
	if numFrames < 3 {
		return nil
	}

	numCoeffs := len(coefficients[0])
	deltas := make([][]float64, numFrames)

	// Use context of 2 frames on each side
	context := 2
	denominator := 0.0
	for t := 1; t <= context; t++ {
		denominator += float64(t * t)
	}
	denominator *= 2

	for f := 0; f < numFrames; f++ {
		deltas[f] = make([]float64, numCoeffs)

		for c := 0; c < numCoeffs; c++ {
			numerator := 0.0
			for t := 1; t <= context; t++ {
				prevFrame := f - t
				nextFrame := f + t

				// Clamp to valid range
				if prevFrame < 0 {
					prevFrame = 0
				}
				if nextFrame >= numFrames {
					nextFrame = numFrames - 1
				}

				numerator += float64(t) * (coefficients[nextFrame][c] - coefficients[prevFrame][c])
			}

			if denominator > 0 {
				deltas[f][c] = numerator / denominator
			}
		}
	}

	return deltas
}

// computeMFCCStats calculates statistics over MFCC frames.
func (a *AudioAnalyzer) computeMFCCStats(coefficients [][]float64) (mean, variance, skewness, kurtosis []float64) {
	numFrames := len(coefficients)
	if numFrames == 0 {
		return
	}

	numCoeffs := len(coefficients[0])
	mean = make([]float64, numCoeffs)
	variance = make([]float64, numCoeffs)
	skewness = make([]float64, numCoeffs)
	kurtosis = make([]float64, numCoeffs)

	// Compute mean
	for c := 0; c < numCoeffs; c++ {
		sum := 0.0
		for f := 0; f < numFrames; f++ {
			sum += coefficients[f][c]
		}
		mean[c] = sum / float64(numFrames)
	}

	// Compute variance, skewness, kurtosis
	for c := 0; c < numCoeffs; c++ {
		m2 := 0.0 // Sum of squared deviations
		m3 := 0.0 // Sum of cubed deviations
		m4 := 0.0 // Sum of 4th power deviations

		for f := 0; f < numFrames; f++ {
			diff := coefficients[f][c] - mean[c]
			m2 += diff * diff
			m3 += diff * diff * diff
			m4 += diff * diff * diff * diff
		}

		variance[c] = m2 / float64(numFrames)

		stdDev := math.Sqrt(variance[c])
		if stdDev > 0 {
			skewness[c] = (m3 / float64(numFrames)) / (stdDev * stdDev * stdDev)
			kurtosis[c] = (m4/float64(numFrames))/(variance[c]*variance[c]) - 3 // Excess kurtosis
		}
	}

	return
}

// AnalyzeMFCCForAI uses MFCC features to detect AI-generated audio.
// Returns a score from 0 (human) to 1 (AI).
func (a *AudioAnalyzer) AnalyzeMFCCForAI(mfcc MFCCResult) float64 {
	if len(mfcc.Coefficients) == 0 || len(mfcc.Mean) == 0 {
		return 0.5 // Neutral score when no data
	}

	score := 0.5

	// AI-generated audio often has:
	// 1. Lower MFCC variance (too consistent)
	// 2. Less variation in delta coefficients (less natural dynamics)
	// 3. Unusual skewness patterns

	// Check variance of first few MFCCs (excluding c0 which is energy)
	if len(mfcc.Variance) > 1 {
		avgVariance := 0.0
		for i := 1; i < min(6, len(mfcc.Variance)); i++ {
			avgVariance += mfcc.Variance[i]
		}
		avgVariance /= float64(min(5, len(mfcc.Variance)-1))

		// Very low variance is suspicious (AI is often too consistent)
		if avgVariance < 10 {
			score += 0.1
		}
		// Very high variance might indicate natural speech
		if avgVariance > 50 {
			score -= 0.1
		}
	}

	// Check delta variance (temporal dynamics)
	if len(mfcc.Delta) > 0 {
		deltaVariance := 0.0
		numCoeffs := len(mfcc.Delta[0])
		for c := 1; c < min(6, numCoeffs); c++ {
			varSum := 0.0
			mean := 0.0
			for f := range mfcc.Delta {
				mean += mfcc.Delta[f][c]
			}
			mean /= float64(len(mfcc.Delta))
			for f := range mfcc.Delta {
				diff := mfcc.Delta[f][c] - mean
				varSum += diff * diff
			}
			deltaVariance += varSum / float64(len(mfcc.Delta))
		}
		deltaVariance /= float64(min(5, numCoeffs-1))

		// Low delta variance = less natural dynamics
		if deltaVariance < 1 {
			score += 0.1
		}
	}

	// Check for unusual kurtosis (AI often has flatter distributions)
	if len(mfcc.Kurtosis) > 1 {
		avgKurtosis := 0.0
		for i := 1; i < min(6, len(mfcc.Kurtosis)); i++ {
			avgKurtosis += math.Abs(mfcc.Kurtosis[i])
		}
		avgKurtosis /= float64(min(5, len(mfcc.Kurtosis)-1))

		// Very low kurtosis might indicate synthetic audio
		if avgKurtosis < 0.5 {
			score += 0.05
		}
	}

	return math.Max(0, math.Min(1, score))
}

// analyzeFLAC extracts metadata from FLAC files.
func (a *AudioAnalyzer) analyzeFLAC(data []byte) (AudioMetadata, AudioStats) {
	meta := AudioMetadata{Format: "flac"}
	stats := AudioStats{FileSize: int64(len(data))}

	if len(data) < 42 {
		return meta, stats
	}

	// FLAC STREAMINFO block starts at byte 4
	// Sample rate at bytes 18-20 (20 bits)
	if len(data) >= 22 {
		sr := (int(data[18]) << 12) | (int(data[19]) << 4) | (int(data[20]) >> 4)
		meta.SampleRate = sr
	}

	// Channels at byte 20 (3 bits) + 1
	if len(data) >= 21 {
		meta.Channels = int((data[20]>>1)&0x07) + 1
	}

	// Bits per sample at bytes 20-21 (5 bits) + 1
	if len(data) >= 22 {
		bps := int((data[20]&0x01)<<4) | int((data[21]>>4)&0x0F)
		meta.BitDepth = bps + 1
	}

	// Look for Vorbis comment block for metadata
	for i := 4; i < len(data)-4 && i < 10000; {
		blockType := data[i] & 0x7F
		isLast := (data[i] & 0x80) != 0
		blockSize := int(data[i+1])<<16 | int(data[i+2])<<8 | int(data[i+3])

		if blockType == 4 { // Vorbis comment
			if i+4+blockSize <= len(data) {
				commentData := string(data[i+4 : i+4+blockSize])
				if containsAIAudioMarker(commentData) {
					meta.IsAIMarked = true
				}
				meta.EncoderName = extractAudioEncoder(commentData)
			}
		}

		i += 4 + blockSize
		if isLast {
			break
		}
	}

	return meta, stats
}

// analyzeOGG extracts metadata from OGG files.
func (a *AudioAnalyzer) analyzeOGG(data []byte) (AudioMetadata, AudioStats) {
	meta := AudioMetadata{Format: "ogg"}
	stats := AudioStats{FileSize: int64(len(data))}

	// Look for Vorbis identification header
	if bytes.Contains(data[:min(len(data), 1000)], []byte("vorbis")) {
		meta.EncoderName = "Vorbis"
	}

	// Look for Opus
	if bytes.Contains(data[:min(len(data), 1000)], []byte("OpusHead")) {
		meta.Format = "opus"
		meta.EncoderName = "Opus"
	}

	// Check for AI markers in comments
	if len(data) > 500 {
		headerData := string(data[:min(len(data), 5000)])
		if containsAIAudioMarker(headerData) {
			meta.IsAIMarked = true
		}
	}

	return meta, stats
}

// analyzeM4A extracts metadata from M4A/AAC files.
func (a *AudioAnalyzer) analyzeM4A(data []byte) (AudioMetadata, AudioStats) {
	meta := AudioMetadata{Format: "m4a"}
	stats := AudioStats{FileSize: int64(len(data))}

	// M4A uses MP4 container - look for metadata atoms
	searchData := string(data[:min(len(data), 10000)])

	if containsAIAudioMarker(searchData) {
		meta.IsAIMarked = true
	}

	meta.EncoderName = extractAudioEncoder(searchData)

	return meta, stats
}

// analyzeMetadata scores based on metadata presence.
func (a *AudioAnalyzer) analyzeMetadata(meta AudioMetadata) float64 {
	score := 0.5

	// AI markers are strong signal
	if meta.IsAIMarked {
		score += 0.35
	}

	// Recording markers suggest real audio
	if meta.HasRecording {
		score -= 0.2
	}

	// ID3 tags suggest real music file
	if meta.HasID3 {
		score -= 0.1
	}

	// Known AI audio tools
	aiTools := []string{
		"elevenlabs", "eleven labs", "murf", "play.ht",
		"resemble", "descript", "synthesia", "wellsaid",
		"amazon polly", "google tts", "azure speech",
		"suno", "udio", "musicgen", "riffusion",
	}

	encoderLower := bytes.ToLower([]byte(meta.EncoderName))
	for _, tool := range aiTools {
		if bytes.Contains(encoderLower, []byte(tool)) {
			score += 0.3
			break
		}
	}

	return math.Max(0, math.Min(1, score))
}

// analyzeFormat checks format-specific indicators.
func (a *AudioAnalyzer) analyzeFormat(meta AudioMetadata, data []byte) float64 {
	score := 0.5

	// Standard sample rates suggest real audio
	standardRates := map[int]bool{44100: true, 48000: true, 96000: true, 22050: true, 16000: true}
	if meta.SampleRate > 0 && !standardRates[meta.SampleRate] {
		score += 0.1 // Unusual sample rate
	}

	// Very high quality might indicate AI (sometimes over-engineered)
	if meta.SampleRate >= 96000 && meta.BitDepth >= 24 {
		score += 0.1
	}

	// Mono audio is more common in AI voice
	if meta.Channels == 1 {
		score += 0.1
	}

	return score
}

// analyzePatterns looks for unusual patterns in audio data.
// Uses format-specific frame analysis instead of raw byte entropy.
func (a *AudioAnalyzer) analyzePatterns(data []byte, format string) float64 {
	if len(data) < 5000 {
		return 0.5
	}

	switch format {
	case "mp3":
		return a.analyzeMP3Patterns(data)
	case "wav":
		return a.analyzeWAVPatterns(data)
	case "flac":
		return a.analyzeFLACPatterns(data)
	default:
		return a.analyzeGenericAudioPatterns(data)
	}
}

// analyzeMP3Patterns performs frame-level MP3 analysis.
// MP3 frames have headers with meaningful audio metadata.
func (a *AudioAnalyzer) analyzeMP3Patterns(data []byte) float64 {
	score := 0.5

	// Find and analyze MP3 frame headers
	frames := a.extractMP3Frames(data)
	if len(frames) < 10 {
		return 0.5
	}

	// Analyze bitrate consistency
	bitrateVariance := a.calculateBitrateVariance(frames)
	if bitrateVariance < 0.01 {
		// Constant bitrate (CBR) - more common in AI-generated
		score += 0.1
	} else if bitrateVariance > 0.3 {
		// High VBR variance - more natural for real recordings
		score -= 0.1
	}

	// Analyze frame padding patterns
	paddingScore := a.analyzeFramePadding(frames)
	score += paddingScore

	// Check for perfect frame alignment (AI often has clean boundaries)
	alignmentScore := a.analyzeFrameAlignment(frames)
	score += alignmentScore

	// Detect silence frames
	silenceRatio := a.detectSilenceFrames(frames)
	if silenceRatio > 0.3 {
		// High silence ratio can indicate AI-generated with gaps
		score += 0.1
	} else if silenceRatio < 0.05 {
		// Very little silence - natural for continuous speech/music
		score -= 0.05
	}

	return math.Max(0, math.Min(1, score))
}

// MP3Frame represents a parsed MP3 frame header.
type MP3Frame struct {
	Offset     int
	Bitrate    int
	SampleRate int
	Padding    bool
	ChannelMode int
	FrameSize  int
}

// extractMP3Frames finds and parses MP3 frame headers.
func (a *AudioAnalyzer) extractMP3Frames(data []byte) []MP3Frame {
	var frames []MP3Frame

	// Bitrate table for MPEG1 Layer 3
	bitrateTable := []int{0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0}
	// Sample rate table for MPEG1
	sampleRateTable := []int{44100, 48000, 32000, 0}

	maxFrames := 500 // Limit to prevent excessive processing

	for i := 0; i < len(data)-4 && len(frames) < maxFrames; i++ {
		// Look for frame sync (0xFF followed by 0xE0-0xFF)
		if data[i] != 0xFF || (data[i+1]&0xE0) != 0xE0 {
			continue
		}

		header := binary.BigEndian.Uint32(data[i : i+4])

		// Parse header fields
		version := (header >> 19) & 0x03       // MPEG version
		layer := (header >> 17) & 0x03         // Layer
		bitrateIdx := (header >> 12) & 0x0F    // Bitrate index
		srIdx := (header >> 10) & 0x03         // Sample rate index
		padding := (header >> 9) & 0x01        // Padding bit
		channelMode := int((header >> 6) & 0x03) // Channel mode

		// Only process MPEG1 Layer 3 (most common MP3)
		if version != 3 || layer != 1 {
			continue
		}

		if bitrateIdx == 0 || bitrateIdx == 15 || srIdx == 3 {
			continue // Invalid values
		}

		bitrate := bitrateTable[bitrateIdx]
		sampleRate := sampleRateTable[srIdx]

		// Calculate frame size: FrameSize = (144 * BitRate / SampleRate) + Padding
		frameSize := (144 * bitrate * 1000 / sampleRate)
		if padding == 1 {
			frameSize++
		}

		frame := MP3Frame{
			Offset:      i,
			Bitrate:     bitrate,
			SampleRate:  sampleRate,
			Padding:     padding == 1,
			ChannelMode: channelMode,
			FrameSize:   frameSize,
		}

		frames = append(frames, frame)

		// Jump to next frame
		if frameSize > 0 && i+frameSize < len(data) {
			i += frameSize - 1 // -1 because loop will increment
		}
	}

	return frames
}

// calculateBitrateVariance computes variance in bitrates across frames.
func (a *AudioAnalyzer) calculateBitrateVariance(frames []MP3Frame) float64 {
	if len(frames) < 2 {
		return 0
	}

	sum := 0.0
	for _, f := range frames {
		sum += float64(f.Bitrate)
	}
	mean := sum / float64(len(frames))

	variance := 0.0
	for _, f := range frames {
		diff := float64(f.Bitrate) - mean
		variance += diff * diff
	}
	variance /= float64(len(frames))

	// Normalize by mean squared for relative variance
	if mean > 0 {
		return variance / (mean * mean)
	}
	return 0
}

// analyzeFramePadding checks padding patterns for anomalies.
func (a *AudioAnalyzer) analyzeFramePadding(frames []MP3Frame) float64 {
	if len(frames) < 10 {
		return 0
	}

	paddedCount := 0
	for _, f := range frames {
		if f.Padding {
			paddedCount++
		}
	}

	paddingRatio := float64(paddedCount) / float64(len(frames))

	// Extremely regular padding (all or none) is suspicious
	if paddingRatio < 0.01 || paddingRatio > 0.99 {
		return 0.05 // Slight AI indicator
	}

	// Very irregular padding is natural
	if paddingRatio > 0.3 && paddingRatio < 0.7 {
		return -0.05 // Slight human indicator
	}

	return 0
}

// analyzeFrameAlignment checks for suspiciously perfect frame alignment.
func (a *AudioAnalyzer) analyzeFrameAlignment(frames []MP3Frame) float64 {
	if len(frames) < 10 {
		return 0
	}

	// Check if frame sizes are identical (CBR with same padding)
	sameSize := 0
	for i := 1; i < len(frames); i++ {
		if frames[i].FrameSize == frames[i-1].FrameSize {
			sameSize++
		}
	}

	identicalRatio := float64(sameSize) / float64(len(frames)-1)

	// Perfect alignment (all frames same size) is suspicious
	if identicalRatio > 0.99 {
		return 0.1
	}

	return 0
}

// detectSilenceFrames estimates proportion of silent frames.
func (a *AudioAnalyzer) detectSilenceFrames(frames []MP3Frame) float64 {
	// In MP3, we can't easily detect silence from headers alone
	// We'd need the audio data. For now, use channel mode as proxy.
	// This is a placeholder for future audio decoding support.

	monoCount := 0
	for _, f := range frames {
		if f.ChannelMode == 3 { // Mono
			monoCount++
		}
	}

	// High mono ratio might indicate voice (TTS often mono)
	if len(frames) > 0 {
		return float64(monoCount) / float64(len(frames)) * 0.1
	}
	return 0
}

// analyzeWAVPatterns analyzes WAV-specific patterns.
func (a *AudioAnalyzer) analyzeWAVPatterns(data []byte) float64 {
	if len(data) < 1000 {
		return 0.5
	}

	score := 0.5

	// Find data chunk
	dataOffset := 44 // Standard WAV header size
	for i := 12; i < len(data)-8 && i < 1000; i++ {
		if bytes.Equal(data[i:i+4], []byte("data")) {
			dataOffset = i + 8
			break
		}
	}

	if dataOffset >= len(data) {
		return 0.5
	}

	// Analyze PCM sample patterns
	sampleData := data[dataOffset:]
	if len(sampleData) < 1000 {
		return 0.5
	}

	// Check for DC offset (common in AI-generated)
	dcOffset := a.calculateDCOffset(sampleData)
	if math.Abs(dcOffset) < 0.001 {
		// Perfect zero DC offset is suspicious (over-processed)
		score += 0.05
	}

	// Check for clipping (saturated samples)
	clippingRatio := a.detectClipping(sampleData)
	if clippingRatio > 0.01 {
		// Real recordings sometimes clip; AI rarely does
		score -= 0.1
	}

	// Analyze zero-crossing rate
	zcrVariance := a.analyzeZeroCrossingVariance(sampleData)
	if zcrVariance < 0.1 {
		// Very consistent zero-crossing rate is suspicious
		score += 0.05
	}

	return math.Max(0, math.Min(1, score))
}

// calculateDCOffset estimates DC offset from PCM samples.
func (a *AudioAnalyzer) calculateDCOffset(data []byte) float64 {
	if len(data) < 100 {
		return 0
	}

	// Assume 16-bit PCM for simplicity
	sum := int64(0)
	count := 0

	for i := 0; i < len(data)-1 && count < 10000; i += 2 {
		sample := int16(binary.LittleEndian.Uint16(data[i : i+2]))
		sum += int64(sample)
		count++
	}

	if count == 0 {
		return 0
	}

	// Normalize to -1 to 1 range
	return float64(sum) / float64(count) / 32768.0
}

// detectClipping finds proportion of clipped samples.
func (a *AudioAnalyzer) detectClipping(data []byte) float64 {
	if len(data) < 100 {
		return 0
	}

	clipped := 0
	total := 0

	// Assume 16-bit PCM
	for i := 0; i < len(data)-1 && total < 10000; i += 2 {
		sample := int16(binary.LittleEndian.Uint16(data[i : i+2]))
		if sample >= 32767 || sample <= -32768 {
			clipped++
		}
		total++
	}

	if total == 0 {
		return 0
	}

	return float64(clipped) / float64(total)
}

// analyzeZeroCrossingVariance calculates variance in zero-crossing rate.
func (a *AudioAnalyzer) analyzeZeroCrossingVariance(data []byte) float64 {
	if len(data) < 2000 {
		return 0.5
	}

	// Calculate ZCR for segments
	segmentSize := 500 // bytes (250 samples for 16-bit)
	var zcrs []float64

	for i := 0; i < len(data)-segmentSize && len(zcrs) < 20; i += segmentSize {
		segment := data[i : i+segmentSize]
		zcr := a.calculateZeroCrossingRate(segment)
		zcrs = append(zcrs, zcr)
	}

	if len(zcrs) < 2 {
		return 0.5
	}

	// Calculate variance
	sum := 0.0
	for _, z := range zcrs {
		sum += z
	}
	mean := sum / float64(len(zcrs))

	variance := 0.0
	for _, z := range zcrs {
		diff := z - mean
		variance += diff * diff
	}

	return variance / float64(len(zcrs))
}

// calculateZeroCrossingRate computes ZCR for a PCM segment.
func (a *AudioAnalyzer) calculateZeroCrossingRate(data []byte) float64 {
	if len(data) < 4 {
		return 0
	}

	crossings := 0
	var prevSample int16

	for i := 0; i < len(data)-1; i += 2 {
		sample := int16(binary.LittleEndian.Uint16(data[i : i+2]))
		if i > 0 && ((prevSample >= 0 && sample < 0) || (prevSample < 0 && sample >= 0)) {
			crossings++
		}
		prevSample = sample
	}

	numSamples := len(data) / 2
	if numSamples > 0 {
		return float64(crossings) / float64(numSamples)
	}
	return 0
}

// analyzeFLACPatterns analyzes FLAC-specific patterns.
func (a *AudioAnalyzer) analyzeFLACPatterns(data []byte) float64 {
	if len(data) < 1000 {
		return 0.5
	}

	score := 0.5

	// FLAC frames start with sync code 0x3FFE (14 bits)
	// Count frames and analyze their distribution

	frameCount := 0
	for i := 42; i < len(data)-2 && frameCount < 500; i++ {
		// Look for frame sync (simplified)
		if data[i] == 0xFF && (data[i+1]&0xFC) == 0xF8 {
			frameCount++
		}
	}

	// Very few frames detected might indicate issues
	expectedFrames := len(data) / 4000 // Rough estimate
	if frameCount > 0 && frameCount < expectedFrames/4 {
		score += 0.05 // Unusual frame structure
	}

	// Check metadata blocks for AI markers (already done in metadata analysis)
	// Add frame-level analysis here

	return score
}

// analyzeGenericAudioPatterns provides fallback analysis for unknown formats.
func (a *AudioAnalyzer) analyzeGenericAudioPatterns(data []byte) float64 {
	if len(data) < 5000 {
		return 0.5
	}

	// Fall back to improved entropy analysis
	// But focus on specific regions that are more meaningful

	score := 0.5

	// Analyze header region vs data region entropy difference
	headerEntropy := calculateEntropy(data[:min(500, len(data))])
	dataStart := len(data) / 4
	dataEntropy := calculateEntropy(data[dataStart : dataStart+min(2000, len(data)-dataStart)])

	// Large entropy difference is expected (headers are structured)
	entropyDiff := math.Abs(headerEntropy - dataEntropy)
	if entropyDiff < 0.5 {
		// Unusual - headers should be lower entropy than audio data
		score += 0.1
	}

	// Check for repeated patterns in audio data region
	repeatScore := a.detectRepeatedPatterns(data[dataStart:])
	score += repeatScore

	return math.Max(0, math.Min(1, score))
}

// detectRepeatedPatterns looks for suspicious repetition in audio data.
func (a *AudioAnalyzer) detectRepeatedPatterns(data []byte) float64 {
	if len(data) < 2000 {
		return 0
	}

	// Check for exact repetition of chunks
	chunkSize := 256
	repetitions := 0
	checks := 0

	for i := 0; i < len(data)-chunkSize*2 && checks < 50; i += chunkSize {
		chunk1 := data[i : i+chunkSize]
		for j := i + chunkSize; j < len(data)-chunkSize && j < i+chunkSize*10; j += chunkSize {
			chunk2 := data[j : j+chunkSize]
			if bytes.Equal(chunk1, chunk2) {
				repetitions++
			}
			checks++
		}
	}

	// High repetition is suspicious
	if checks > 0 && float64(repetitions)/float64(checks) > 0.1 {
		return 0.15
	}

	return 0
}

// analyzeQuality evaluates audio quality indicators.
func (a *AudioAnalyzer) analyzeQuality(meta AudioMetadata, stats AudioStats) float64 {
	score := 0.5

	// Very low bitrate might indicate AI optimization
	if meta.Bitrate > 0 && meta.Bitrate < 64 {
		score += 0.1
	}

	// Unusually high bitrate
	if meta.Bitrate > 320 {
		score += 0.1
	}

	// File size sanity check
	if stats.FileSize > 0 && stats.FileSize < 10000 {
		score += 0.2 // Very small file is suspicious
	}

	return score
}

// detectAISignatures looks for known AI audio signatures.
func (a *AudioAnalyzer) detectAISignatures(data []byte, meta AudioMetadata) float64 {
	score := 0.0

	// Check for AI tool watermarks
	searchData := data[:min(len(data), 50000)]

	aiMarkers := []string{
		"elevenlabs", "murf.ai", "play.ht", "resemble.ai",
		"suno", "udio", "generated", "synthetic", "ai voice",
		"text-to-speech", "tts", "voice clone",
	}

	for _, marker := range aiMarkers {
		if bytes.Contains(bytes.ToLower(searchData), []byte(marker)) {
			score += 0.3
			break
		}
	}

	// Check metadata encoder
	if meta.IsAIMarked {
		score += 0.4
	}

	// Certain exact durations are suspicious (AI often generates exact lengths)
	// This would need actual duration parsing

	return math.Min(1, score)
}

// analyzeNoiseProfile checks for natural noise characteristics.
func (a *AudioAnalyzer) analyzeNoiseProfile(data []byte, format string) float64 {
	if len(data) < 10000 {
		return 0.5
	}

	// Look at byte distribution as proxy for audio characteristics
	// Real recordings have natural noise; AI audio is often "too clean"

	// Sample from middle of file (skip headers)
	start := len(data) / 3
	end := start + min(5000, len(data)-start)
	sample := data[start:end]

	// Check for repeated patterns (AI sometimes has artifacts)
	repeatCount := 0
	windowSize := 100

	for i := 0; i < len(sample)-windowSize*2; i += windowSize {
		window1 := sample[i : i+windowSize]
		window2 := sample[i+windowSize : i+windowSize*2]

		if byteSimilarity(window1, window2) > 0.9 {
			repeatCount++
		}
	}

	if repeatCount > 5 {
		return 0.7 // Suspicious repetition
	}

	return 0.4
}

// calculateWeightedScore combines signals into final score.
func (a *AudioAnalyzer) calculateWeightedScore(signals AudioSignals) float64 {
	w := a.weights

	score := signals.MetadataScore*w.MetadataScore +
		signals.FormatAnalysis*w.FormatAnalysis +
		signals.PatternAnalysis*w.PatternAnalysis +
		signals.QualityIndicators*w.QualityIndicators +
		signals.AISignatures*w.AISignatures +
		signals.NoiseProfile*w.NoiseProfile

	totalWeight := w.MetadataScore + w.FormatAnalysis + w.PatternAnalysis +
		w.QualityIndicators + w.AISignatures + w.NoiseProfile

	if totalWeight > 0 {
		score /= totalWeight
	}

	return math.Max(0, math.Min(1, score))
}

// =============================================================================
// Helper Functions
// =============================================================================

// containsAIAudioMarker checks for AI audio generator markers.
func containsAIAudioMarker(s string) bool {
	markers := []string{
		"elevenlabs", "eleven labs", "murf", "play.ht",
		"resemble", "descript", "synthesia", "wellsaid",
		"suno", "udio", "musicgen", "riffusion",
		"ai generated", "ai-generated", "synthetic voice",
		"text to speech", "text-to-speech", "tts",
		"voice clone", "cloned voice",
	}

	lower := bytes.ToLower([]byte(s))
	for _, marker := range markers {
		if bytes.Contains(lower, []byte(marker)) {
			return true
		}
	}
	return false
}

// containsRecordingMarker checks for real recording indicators.
func containsRecordingMarker(s string) bool {
	markers := []string{
		"recorded", "recording", "studio", "microphone",
		"live", "concert", "session", "interview",
		"iphone", "android", "voice memo",
	}

	lower := bytes.ToLower([]byte(s))
	for _, marker := range markers {
		if bytes.Contains(lower, []byte(marker)) {
			return true
		}
	}
	return false
}

// extractAudioEncoder tries to find encoder name.
func extractAudioEncoder(s string) string {
	encoders := map[string]string{
		"lame":        "LAME",
		"ffmpeg":      "ffmpeg",
		"audacity":    "Audacity",
		"adobe":       "Adobe Audition",
		"logic":       "Logic Pro",
		"pro tools":   "Pro Tools",
		"ableton":     "Ableton Live",
		"fl studio":   "FL Studio",
		"elevenlabs":  "ElevenLabs",
		"suno":        "Suno AI",
		"udio":        "Udio",
	}

	lower := bytes.ToLower([]byte(s))
	for key, name := range encoders {
		if bytes.Contains(lower, []byte(key)) {
			return name
		}
	}

	return ""
}
