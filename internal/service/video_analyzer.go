package service

import (
	"bytes"
	"encoding/binary"
	"math"
)

// =============================================================================
// Manuscript Video Detection Algorithm
// =============================================================================
//
// Video detection is challenging because AI-generated videos have evolved rapidly.
// We use forensic analysis that doesn't require ML models:
//
// Key signals:
//   1. Container/codec metadata analysis
//   2. Frame consistency (extracted from container structure)
//   3. Audio track presence and characteristics
//   4. Temporal patterns in compressed data
//   5. Encoding signatures
//   6. Duration and bitrate anomalies
//
// Limitations:
//   - Full frame-by-frame analysis requires decoding (ffmpeg)
//   - Our analysis works on container metadata and byte patterns
//   - For production accuracy, combine with external APIs
//
// =============================================================================

// VideoAnalyzer performs forensic analysis on video files.
type VideoAnalyzer struct {
	weights VideoAnalyzerWeights
}

// VideoAnalyzerWeights controls signal importance.
type VideoAnalyzerWeights struct {
	MetadataScore      float64
	ContainerAnalysis  float64
	AudioPresence      float64
	TemporalPattern    float64
	EncodingSignature  float64
	BitrateConsistency float64
}

// DefaultVideoWeights returns tuned weights.
func DefaultVideoWeights() VideoAnalyzerWeights {
	return VideoAnalyzerWeights{
		MetadataScore:      0.25,
		ContainerAnalysis:  0.20,
		AudioPresence:      0.15,
		TemporalPattern:    0.15,
		EncodingSignature:  0.15,
		BitrateConsistency: 0.10,
	}
}

// NewVideoAnalyzer creates a new analyzer.
func NewVideoAnalyzer() *VideoAnalyzer {
	return &VideoAnalyzer{
		weights: DefaultVideoWeights(),
	}
}

// VideoAnalysisResult contains analysis results.
type VideoAnalysisResult struct {
	AIScore  float64
	Signals  VideoSignals
	Metadata VideoMetadata
	Stats    VideoStats
}

// VideoSignals contains individual signal scores.
type VideoSignals struct {
	MetadataScore      float64 // Missing/fake metadata = AI-like
	ContainerAnalysis  float64 // Unusual container structure = AI-like
	AudioPresence      float64 // Missing audio = suspicious
	TemporalPattern    float64 // Unusual patterns = AI-like
	EncodingSignature  float64 // Unknown encoder = suspicious
	BitrateConsistency float64 // Unusual bitrate = AI-like
}

// VideoMetadata contains extracted metadata.
type VideoMetadata struct {
	Format       string // mp4, webm, avi, etc.
	HasAudio     bool
	HasVideo     bool
	EncoderName  string
	CreationTime string
	Duration     float64 // seconds (estimated)
	IsAIMarked   bool    // Contains AI generator markers
}

// VideoStats contains video statistics.
type VideoStats struct {
	FileSize       int64
	EstimatedFPS   float64
	Width          int
	Height         int
	VideoBitrate   int // estimated kbps
	AudioBitrate   int // estimated kbps
	KeyframeCount  int
	ChunkCount     int

	// Frame extraction results
	ExtractedFrames []ExtractedFrame
}

// ExtractedFrame represents raw frame data extracted from video.
type ExtractedFrame struct {
	Index      int     // Frame index in video
	FrameType  byte    // 'I' for keyframe, 'P' for predicted, 'B' for bidirectional
	Offset     int     // Byte offset in video file
	Size       int     // Frame size in bytes
	Data       []byte  // Raw frame data (NAL unit for H.264)
	Timestamp  float64 // Estimated timestamp in seconds
	Width      int     // Frame width (if available)
	Height     int     // Frame height (if available)
}

// FrameExtractionConfig controls frame extraction behavior.
type FrameExtractionConfig struct {
	Enabled       bool // Whether to extract frames
	MaxFrames     int  // Maximum number of frames to extract (default 10)
	KeyframesOnly bool // Only extract I-frames/keyframes (default true)
	SampleRate    int  // Sample every N frames (default 1 = all qualifying frames)
}

// Analyze performs forensic analysis on video data.
func (a *VideoAnalyzer) Analyze(data []byte) VideoAnalysisResult {
	result := VideoAnalysisResult{}
	result.Stats.FileSize = int64(len(data))

	// Detect format
	format := a.detectVideoFormat(data)
	result.Metadata.Format = format

	// Extract metadata based on format
	switch format {
	case "mp4", "mov":
		result.Metadata, result.Stats = a.analyzeMP4(data)
	case "webm":
		result.Metadata, result.Stats = a.analyzeWebM(data)
	case "avi":
		result.Metadata, result.Stats = a.analyzeAVI(data)
	default:
		result.Metadata.Format = format
	}

	// Calculate signals
	result.Signals.MetadataScore = a.analyzeMetadata(result.Metadata)
	result.Signals.ContainerAnalysis = a.analyzeContainer(data, format)
	result.Signals.AudioPresence = a.analyzeAudioPresence(result.Metadata, data, format)
	result.Signals.TemporalPattern = a.analyzeTemporalPattern(data, format)
	result.Signals.EncodingSignature = a.analyzeEncodingSignature(data, result.Metadata)
	result.Signals.BitrateConsistency = a.analyzeBitrateConsistency(data, result.Stats)

	// Calculate weighted score
	result.AIScore = a.calculateWeightedScore(result.Signals)

	return result
}

// detectVideoFormat identifies video format from magic bytes.
func (a *VideoAnalyzer) detectVideoFormat(data []byte) string {
	if len(data) < 4 {
		return "unknown"
	}

	// MP4/MOV: ftyp atom at offset 4
	if len(data) >= 12 && bytes.Equal(data[4:8], []byte("ftyp")) {
		// Check brand to distinguish MP4 from MOV
		brand := string(data[8:12])
		switch brand {
		case "isom", "iso2", "mp41", "mp42", "avc1", "M4V ":
			return "mp4"
		case "qt  ":
			return "mov"
		}
		return "mp4" // Default to mp4 for ftyp
	}

	// WebM/MKV: EBML header (1A 45 DF A3)
	if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
		// Check for webm doctype
		if bytes.Contains(data[:min(100, len(data))], []byte("webm")) {
			return "webm"
		}
		return "mkv"
	}

	// AVI: RIFF....AVI
	if len(data) >= 12 {
		if bytes.Equal(data[0:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("AVI ")) {
			return "avi"
		}
	}

	// FLV: FLV + version
	if len(data) >= 4 && data[0] == 'F' && data[1] == 'L' && data[2] == 'V' {
		return "flv"
	}

	// MPEG-TS: sync byte 0x47 repeating
	if data[0] == 0x47 {
		// Check for more sync bytes at 188-byte intervals
		if len(data) > 376 && data[188] == 0x47 && data[376] == 0x47 {
			return "ts"
		}
	}

	return "unknown"
}

// analyzeMP4 extracts metadata from MP4/MOV containers.
func (a *VideoAnalyzer) analyzeMP4(data []byte) (VideoMetadata, VideoStats) {
	meta := VideoMetadata{Format: "mp4", HasVideo: true}
	stats := VideoStats{FileSize: int64(len(data))}

	// Parse MP4 atoms/boxes
	offset := 0
	for offset < len(data)-8 {
		if offset+8 > len(data) {
			break
		}

		// Atom size (4 bytes) + type (4 bytes)
		atomSize := int(binary.BigEndian.Uint32(data[offset : offset+4]))
		atomType := string(data[offset+4 : offset+8])

		if atomSize < 8 {
			break // Invalid atom
		}
		if offset+atomSize > len(data) {
			atomSize = len(data) - offset // Truncated file
		}

		switch atomType {
		case "moov":
			// Movie atom - contains metadata
			meta, stats = a.parseMovieAtom(data[offset:offset+atomSize], meta, stats)

		case "mdat":
			// Media data - actual video/audio content
			stats.ChunkCount++

		case "ftyp":
			// File type - check for AI markers
			if atomSize > 8 {
				ftypData := string(data[offset+8 : min(offset+atomSize, offset+100)])
				if containsAIMarker(ftypData) {
					meta.IsAIMarked = true
				}
			}

		case "udta":
			// User data - may contain encoder info
			if atomSize > 8 {
				udtaData := string(data[offset+8 : min(offset+atomSize, offset+500)])
				meta.EncoderName = extractEncoder(udtaData)
				if containsAIMarker(udtaData) {
					meta.IsAIMarked = true
				}
			}
		}

		offset += atomSize
	}

	return meta, stats
}

// parseMovieAtom parses the moov atom for metadata.
func (a *VideoAnalyzer) parseMovieAtom(data []byte, meta VideoMetadata, stats VideoStats) (VideoMetadata, VideoStats) {
	offset := 8 // Skip moov header

	for offset < len(data)-8 {
		if offset+8 > len(data) {
			break
		}

		atomSize := int(binary.BigEndian.Uint32(data[offset : offset+4]))
		atomType := string(data[offset+4 : offset+8])

		if atomSize < 8 || offset+atomSize > len(data) {
			break
		}

		switch atomType {
		case "mvhd":
			// Movie header - contains duration, timescale
			if atomSize >= 32 {
				// Duration and timescale for calculating length
				// Version 0: timescale at 20, duration at 24
				// Version 1: timescale at 28, duration at 32
			}

		case "trak":
			// Track - check if audio or video
			trackData := data[offset : offset+atomSize]
			if bytes.Contains(trackData, []byte("soun")) {
				meta.HasAudio = true
			}
			if bytes.Contains(trackData, []byte("vide")) {
				meta.HasVideo = true
			}

			// Look for resolution in video track
			if bytes.Contains(trackData, []byte("avc1")) || bytes.Contains(trackData, []byte("hvc1")) {
				// H.264 or H.265 video
			}

		case "meta":
			// Metadata atom
			if atomSize > 8 {
				metaData := string(data[offset+8 : min(offset+atomSize, offset+1000)])
				if containsAIMarker(metaData) {
					meta.IsAIMarked = true
				}
				if enc := extractEncoder(metaData); enc != "" {
					meta.EncoderName = enc
				}
			}
		}

		offset += atomSize
	}

	return meta, stats
}

// analyzeWebM extracts metadata from WebM/MKV containers.
func (a *VideoAnalyzer) analyzeWebM(data []byte) (VideoMetadata, VideoStats) {
	meta := VideoMetadata{Format: "webm", HasVideo: true}
	stats := VideoStats{FileSize: int64(len(data))}

	// Look for common elements in EBML structure
	dataStr := string(data[:min(len(data), 2000)])

	// Check for audio
	if bytes.Contains(data, []byte{0x81}) { // Audio track type
		meta.HasAudio = true
	}

	// Check for encoder
	meta.EncoderName = extractEncoder(dataStr)

	// Check for AI markers
	if containsAIMarker(dataStr) {
		meta.IsAIMarked = true
	}

	return meta, stats
}

// analyzeAVI extracts metadata from AVI containers.
func (a *VideoAnalyzer) analyzeAVI(data []byte) (VideoMetadata, VideoStats) {
	meta := VideoMetadata{Format: "avi", HasVideo: true}
	stats := VideoStats{FileSize: int64(len(data))}

	// AVI uses RIFF structure
	// Look for audio stream
	if bytes.Contains(data, []byte("auds")) {
		meta.HasAudio = true
	}

	// Look for encoder info in headers
	if len(data) > 500 {
		headerData := string(data[:500])
		meta.EncoderName = extractEncoder(headerData)
		if containsAIMarker(headerData) {
			meta.IsAIMarked = true
		}
	}

	return meta, stats
}

// analyzeMetadata scores based on metadata presence and quality.
func (a *VideoAnalyzer) analyzeMetadata(meta VideoMetadata) float64 {
	score := 0.5 // Start neutral

	// AI-generated videos often marked
	if meta.IsAIMarked {
		score += 0.4
	}

	// Real videos usually have audio
	if !meta.HasAudio {
		score += 0.15 // Many AI videos lack audio
	}

	// Known AI video generators
	aiEncoders := []string{
		"runway", "pika", "sora", "gen-2", "gen2",
		"stable video", "stablevideo", "modelscope",
		"deforum", "animatediff", "zeroscope",
	}

	encoderLower := bytes.ToLower([]byte(meta.EncoderName))
	for _, ai := range aiEncoders {
		if bytes.Contains(encoderLower, []byte(ai)) {
			score += 0.3
			break
		}
	}

	// Professional encoders suggest real video
	proEncoders := []string{
		"premiere", "final cut", "davinci", "avid",
		"ffmpeg", "handbrake", "x264", "x265",
	}
	for _, pro := range proEncoders {
		if bytes.Contains(encoderLower, []byte(pro)) {
			score -= 0.1
			break
		}
	}

	return math.Max(0, math.Min(1, score))
}

// analyzeContainer checks container structure for anomalies.
func (a *VideoAnalyzer) analyzeContainer(data []byte, format string) float64 {
	if len(data) < 1000 {
		return 0.5
	}

	score := 0.5

	switch format {
	case "mp4", "mov":
		// Check for proper atom structure
		hasMoviAtom := bytes.Contains(data[:min(len(data), 100000)], []byte("moov"))
		hasMdatAtom := bytes.Contains(data, []byte("mdat"))

		if !hasMoviAtom {
			score += 0.2 // Unusual
		}
		if !hasMdatAtom {
			score += 0.2 // Very unusual
		}

	case "webm":
		// Check for proper EBML structure
		hasSegment := bytes.Contains(data[:min(len(data), 1000)], []byte{0x18, 0x53, 0x80, 0x67})
		if !hasSegment {
			score += 0.2
		}
	}

	return score
}

// analyzeAudioPresence evaluates audio track characteristics.
func (a *VideoAnalyzer) analyzeAudioPresence(meta VideoMetadata, data []byte, format string) float64 {
	// Most real videos have audio
	// AI-generated videos often lack audio or have synthetic audio

	if meta.HasAudio {
		return 0.3 // Good sign
	}

	// No audio is suspicious but not definitive
	// Short clips and specific content may legitimately lack audio
	return 0.7
}

// analyzeTemporalPattern looks for unusual patterns in the video data.
// Uses format-specific frame analysis instead of arbitrary byte comparison.
func (a *VideoAnalyzer) analyzeTemporalPattern(data []byte, format string) float64 {
	if len(data) < 10000 {
		return 0.5
	}

	switch format {
	case "mp4", "mov":
		return a.analyzeMP4TemporalPattern(data)
	case "webm", "mkv":
		return a.analyzeWebMTemporalPattern(data)
	case "avi":
		return a.analyzeAVITemporalPattern(data)
	default:
		return a.analyzeGenericTemporalPattern(data)
	}
}

// VideoFrame represents a detected video frame for analysis.
type VideoFrame struct {
	Offset    int
	Size      int
	FrameType byte // 'I', 'P', 'B', or 0 for unknown
	NALType   int  // For H.264/H.265
}

// analyzeMP4TemporalPattern analyzes I-frame positions and GOP structure in MP4.
func (a *VideoAnalyzer) analyzeMP4TemporalPattern(data []byte) float64 {
	score := 0.5

	// Find mdat atom (media data) for frame analysis
	mdatOffset, mdatSize := a.findMP4Atom(data, "mdat")
	if mdatOffset < 0 || mdatSize < 1000 {
		return 0.5
	}

	// Analyze NAL units in H.264/H.265 video
	frames := a.extractH264Frames(data[mdatOffset:min(mdatOffset+mdatSize, len(data))])
	if len(frames) < 5 {
		return 0.5
	}

	// Analyze I-frame distribution (keyframe positions)
	iFrameScore := a.analyzeIFrameDistribution(frames)
	score += iFrameScore

	// Analyze GOP structure consistency
	gopScore := a.analyzeGOPStructure(frames)
	score += gopScore

	// Analyze frame size variation
	frameSizeScore := a.analyzeFrameSizeVariation(frames)
	score += frameSizeScore

	// Check for unnatural frame patterns
	patternScore := a.detectUnaturalFramePatterns(frames)
	score += patternScore

	return math.Max(0, math.Min(1, score))
}

// findMP4Atom locates an MP4 atom by type and returns offset and size.
func (a *VideoAnalyzer) findMP4Atom(data []byte, atomType string) (int, int) {
	offset := 0
	for offset < len(data)-8 {
		if offset+8 > len(data) {
			break
		}

		atomSize := int(binary.BigEndian.Uint32(data[offset : offset+4]))
		if atomSize == 0 {
			break // Extends to end of file
		}
		if atomSize == 1 && offset+16 <= len(data) {
			// 64-bit size
			atomSize = int(binary.BigEndian.Uint64(data[offset+8 : offset+16]))
		}

		if atomSize < 8 {
			break
		}

		currentType := string(data[offset+4 : offset+8])
		if currentType == atomType {
			return offset + 8, atomSize - 8
		}

		// For container atoms, search recursively
		if currentType == "moov" || currentType == "trak" || currentType == "mdia" || currentType == "minf" || currentType == "stbl" {
			subOffset, subSize := a.findMP4Atom(data[offset+8:min(offset+atomSize, len(data))], atomType)
			if subOffset >= 0 {
				return offset + 8 + subOffset, subSize
			}
		}

		offset += atomSize
	}

	return -1, 0
}

// extractH264Frames finds H.264/H.265 NAL units and their frame types.
func (a *VideoAnalyzer) extractH264Frames(data []byte) []VideoFrame {
	var frames []VideoFrame
	maxFrames := 500

	// Look for NAL unit start codes: 00 00 01 or 00 00 00 01
	i := 0
	for i < len(data)-4 && len(frames) < maxFrames {
		// Check for 3-byte or 4-byte start code
		startCodeLen := 0
		if data[i] == 0 && data[i+1] == 0 && data[i+2] == 1 {
			startCodeLen = 3
		} else if i < len(data)-4 && data[i] == 0 && data[i+1] == 0 && data[i+2] == 0 && data[i+3] == 1 {
			startCodeLen = 4
		}

		if startCodeLen == 0 {
			i++
			continue
		}

		nalStart := i + startCodeLen
		if nalStart >= len(data) {
			break
		}

		// Get NAL unit type (5 bits for H.264)
		nalHeader := data[nalStart]
		nalType := int(nalHeader & 0x1F)

		// Find next start code to determine NAL size
		nalEnd := len(data)
		for j := nalStart + 1; j < len(data)-3 && j < nalStart+100000; j++ {
			if data[j] == 0 && data[j+1] == 0 && (data[j+2] == 1 || (data[j+2] == 0 && j+3 < len(data) && data[j+3] == 1)) {
				nalEnd = j
				break
			}
		}

		frame := VideoFrame{
			Offset:  i,
			Size:    nalEnd - i,
			NALType: nalType,
		}

		// Classify frame type based on NAL unit type
		switch nalType {
		case 5: // IDR picture (keyframe)
			frame.FrameType = 'I'
		case 1: // Non-IDR picture (P or B frame)
			frame.FrameType = 'P' // Simplified - could be P or B
		case 7, 8: // SPS, PPS - skip these
			i = nalEnd
			continue
		}

		if frame.FrameType != 0 {
			frames = append(frames, frame)
		}

		i = nalEnd
	}

	return frames
}

// analyzeIFrameDistribution checks I-frame (keyframe) positions.
func (a *VideoAnalyzer) analyzeIFrameDistribution(frames []VideoFrame) float64 {
	if len(frames) < 5 {
		return 0
	}

	var iFramePositions []int
	for i, f := range frames {
		if f.FrameType == 'I' {
			iFramePositions = append(iFramePositions, i)
		}
	}

	if len(iFramePositions) < 2 {
		// Only one I-frame is unusual for longer videos
		if len(frames) > 100 {
			return 0.1 // Suspicious
		}
		return 0
	}

	// Calculate intervals between I-frames
	var intervals []int
	for i := 1; i < len(iFramePositions); i++ {
		intervals = append(intervals, iFramePositions[i]-iFramePositions[i-1])
	}

	// Calculate variance of intervals
	sum := 0
	for _, interval := range intervals {
		sum += interval
	}
	mean := float64(sum) / float64(len(intervals))

	variance := 0.0
	for _, interval := range intervals {
		diff := float64(interval) - mean
		variance += diff * diff
	}
	variance /= float64(len(intervals))

	// Perfectly regular intervals (variance = 0) can indicate AI
	if variance < 0.5 && len(intervals) > 3 {
		return 0.05 // Slight AI indicator
	}

	// Very irregular intervals are more natural
	if variance > mean*0.5 {
		return -0.05 // Slight human indicator
	}

	return 0
}

// analyzeGOPStructure checks Group of Pictures consistency.
func (a *VideoAnalyzer) analyzeGOPStructure(frames []VideoFrame) float64 {
	if len(frames) < 10 {
		return 0
	}

	// Find GOP patterns (frames between I-frames)
	var gopSizes []int
	currentGOPSize := 0

	for _, f := range frames {
		if f.FrameType == 'I' {
			if currentGOPSize > 0 {
				gopSizes = append(gopSizes, currentGOPSize)
			}
			currentGOPSize = 0
		}
		currentGOPSize++
	}

	if len(gopSizes) < 2 {
		return 0
	}

	// Calculate GOP size variance
	sum := 0
	for _, size := range gopSizes {
		sum += size
	}
	mean := float64(sum) / float64(len(gopSizes))

	variance := 0.0
	for _, size := range gopSizes {
		diff := float64(size) - mean
		variance += diff * diff
	}
	variance /= float64(len(gopSizes))

	// Normalize by mean
	coeffOfVar := 0.0
	if mean > 0 {
		coeffOfVar = math.Sqrt(variance) / mean
	}

	// Very consistent GOP size might indicate AI generation
	if coeffOfVar < 0.05 && len(gopSizes) > 3 {
		return 0.05
	}

	return 0
}

// analyzeFrameSizeVariation checks if frame sizes vary naturally.
func (a *VideoAnalyzer) analyzeFrameSizeVariation(frames []VideoFrame) float64 {
	if len(frames) < 10 {
		return 0
	}

	// Separate I-frames and P-frames
	var iFrameSizes, pFrameSizes []int

	for _, f := range frames {
		if f.FrameType == 'I' {
			iFrameSizes = append(iFrameSizes, f.Size)
		} else if f.FrameType == 'P' {
			pFrameSizes = append(pFrameSizes, f.Size)
		}
	}

	score := 0.0

	// Analyze I-frame size consistency
	if len(iFrameSizes) > 2 {
		iVariance := a.calculateSizeVariance(iFrameSizes)
		// Very consistent I-frame sizes are suspicious
		if iVariance < 0.02 {
			score += 0.03
		}
	}

	// Analyze P-frame size consistency
	if len(pFrameSizes) > 5 {
		pVariance := a.calculateSizeVariance(pFrameSizes)
		// Very consistent P-frame sizes are suspicious
		if pVariance < 0.02 {
			score += 0.02
		}
	}

	return score
}

// calculateSizeVariance computes normalized variance of frame sizes.
func (a *VideoAnalyzer) calculateSizeVariance(sizes []int) float64 {
	if len(sizes) < 2 {
		return 0
	}

	sum := 0
	for _, s := range sizes {
		sum += s
	}
	mean := float64(sum) / float64(len(sizes))

	if mean == 0 {
		return 0
	}

	variance := 0.0
	for _, s := range sizes {
		diff := float64(s) - mean
		variance += diff * diff
	}
	variance /= float64(len(sizes))

	// Return coefficient of variation
	return math.Sqrt(variance) / mean
}

// detectUnaturalFramePatterns looks for AI-specific frame patterns.
func (a *VideoAnalyzer) detectUnaturalFramePatterns(frames []VideoFrame) float64 {
	if len(frames) < 5 {
		return 0
	}

	score := 0.0

	// Check for repeating frame size patterns
	sizes := make([]int, len(frames))
	for i, f := range frames {
		sizes[i] = f.Size
	}

	// Look for exact size repetitions
	sizeCount := make(map[int]int)
	for _, s := range sizes {
		sizeCount[s]++
	}

	// Count how many sizes appear more than once
	repeatedSizes := 0
	for _, count := range sizeCount {
		if count > 1 {
			repeatedSizes++
		}
	}

	// High proportion of repeated sizes is suspicious
	if float64(repeatedSizes)/float64(len(sizeCount)) > 0.5 && len(sizes) > 20 {
		score += 0.05
	}

	return score
}

// analyzeWebMTemporalPattern analyzes frame patterns in WebM/MKV.
func (a *VideoAnalyzer) analyzeWebMTemporalPattern(data []byte) float64 {
	score := 0.5

	// WebM uses EBML format with SimpleBlock or Block elements
	// Look for Cluster elements (ID: 0x1F43B675)
	clusterCount := 0
	clusterSizes := make([]int, 0)

	for i := 0; i < len(data)-8 && clusterCount < 100; i++ {
		// Cluster element ID
		if data[i] == 0x1F && data[i+1] == 0x43 && data[i+2] == 0xB6 && data[i+3] == 0x75 {
			clusterCount++

			// Try to get cluster size (variable length)
			if i+4 < len(data) {
				sizeStart := i + 4
				size := a.readEBMLSize(data[sizeStart:])
				if size > 0 && size < 10000000 {
					clusterSizes = append(clusterSizes, size)
				}
			}
		}
	}

	if len(clusterSizes) < 3 {
		return 0.5
	}

	// Analyze cluster size variance
	variance := a.calculateSizeVariance(clusterSizes)

	// Very consistent cluster sizes might indicate AI
	if variance < 0.05 {
		score += 0.05
	}

	return score
}

// readEBMLSize reads a variable-length EBML size.
func (a *VideoAnalyzer) readEBMLSize(data []byte) int {
	if len(data) == 0 {
		return 0
	}

	first := data[0]
	if first == 0 {
		return 0
	}

	// Count leading zeros to determine size length
	length := 0
	mask := byte(0x80)
	for i := 0; i < 8; i++ {
		if (first & mask) != 0 {
			length = i + 1
			break
		}
		mask >>= 1
	}

	if length == 0 || len(data) < length {
		return 0
	}

	// Read size value
	value := int(first & (0xFF >> length))
	for i := 1; i < length; i++ {
		value = (value << 8) | int(data[i])
	}

	return value
}

// analyzeAVITemporalPattern analyzes frame patterns in AVI.
func (a *VideoAnalyzer) analyzeAVITemporalPattern(data []byte) float64 {
	score := 0.5

	// AVI uses RIFF chunks - look for video frames (##dc or ##db)
	frameSizes := make([]int, 0)
	maxFrames := 200

	for i := 0; i < len(data)-8 && len(frameSizes) < maxFrames; i++ {
		// Video frame chunks: 00dc (compressed) or 00db (uncompressed)
		if (data[i] == '0' && data[i+1] == '0' && data[i+2] == 'd' && (data[i+3] == 'c' || data[i+3] == 'b')) ||
			(data[i] == '0' && data[i+1] == '1' && data[i+2] == 'd' && (data[i+3] == 'c' || data[i+3] == 'b')) {
			// Chunk size follows (little-endian)
			if i+8 <= len(data) {
				size := int(binary.LittleEndian.Uint32(data[i+4 : i+8]))
				if size > 0 && size < 10000000 {
					frameSizes = append(frameSizes, size)
				}
			}
		}
	}

	if len(frameSizes) < 5 {
		return 0.5
	}

	// Analyze frame size variance
	variance := a.calculateSizeVariance(frameSizes)

	// Very consistent frame sizes might indicate AI
	if variance < 0.05 {
		score += 0.05
	}

	return score
}

// analyzeGenericTemporalPattern provides fallback analysis.
func (a *VideoAnalyzer) analyzeGenericTemporalPattern(data []byte) float64 {
	if len(data) < 10000 {
		return 0.5
	}

	score := 0.5

	// Look for common video start codes and analyze distribution
	// H.264 start code: 00 00 00 01 or 00 00 01
	startCodeCount := 0
	var startCodePositions []int

	for i := 0; i < len(data)-4 && startCodeCount < 500; i++ {
		if data[i] == 0 && data[i+1] == 0 {
			if data[i+2] == 1 || (data[i+2] == 0 && i+3 < len(data) && data[i+3] == 1) {
				startCodePositions = append(startCodePositions, i)
				startCodeCount++
			}
		}
	}

	if len(startCodePositions) < 5 {
		return 0.5
	}

	// Analyze spacing between start codes
	var intervals []int
	for i := 1; i < len(startCodePositions); i++ {
		intervals = append(intervals, startCodePositions[i]-startCodePositions[i-1])
	}

	// Calculate variance of intervals
	sum := 0
	for _, interval := range intervals {
		sum += interval
	}
	mean := float64(sum) / float64(len(intervals))

	if mean == 0 {
		return 0.5
	}

	variance := 0.0
	for _, interval := range intervals {
		diff := float64(interval) - mean
		variance += diff * diff
	}
	variance /= float64(len(intervals))
	coeffVar := math.Sqrt(variance) / mean

	// Very regular spacing is suspicious
	if coeffVar < 0.1 && len(intervals) > 10 {
		score += 0.05
	}

	return score
}

// analyzeEncodingSignature checks for known encoder signatures.
func (a *VideoAnalyzer) analyzeEncodingSignature(data []byte, meta VideoMetadata) float64 {
	score := 0.5

	// Check for H.264/H.265 encoding (common in both real and AI)
	hasH264 := bytes.Contains(data, []byte("avc1")) || bytes.Contains(data, []byte("h264"))
	hasH265 := bytes.Contains(data, []byte("hvc1")) || bytes.Contains(data, []byte("hevc"))
	hasVP9 := bytes.Contains(data, []byte("vp09"))
	hasAV1 := bytes.Contains(data, []byte("av01"))

	// Standard codecs are neutral
	if hasH264 || hasH265 || hasVP9 || hasAV1 {
		score = 0.45
	}

	// Unknown/missing codec is suspicious
	if !hasH264 && !hasH265 && !hasVP9 && !hasAV1 {
		if meta.Format == "mp4" || meta.Format == "webm" {
			score = 0.6
		}
	}

	return score
}

// analyzeBitrateConsistency estimates bitrate consistency.
func (a *VideoAnalyzer) analyzeBitrateConsistency(data []byte, stats VideoStats) float64 {
	// Very rough estimation without full parsing
	// AI videos sometimes have unusual bitrate characteristics

	if stats.FileSize < 100000 { // Less than 100KB
		return 0.6 // Suspiciously small for video
	}

	// Estimate bits per byte ratio in different regions
	if len(data) < 10000 {
		return 0.5
	}

	// Check entropy in different parts
	entropies := make([]float64, 0)
	chunkSize := len(data) / 5

	for i := 0; i < 5; i++ {
		start := i * chunkSize
		end := start + min(chunkSize, len(data)-start)
		if end > start {
			entropy := calculateEntropy(data[start:end])
			entropies = append(entropies, entropy)
		}
	}

	if len(entropies) < 2 {
		return 0.5
	}

	// Calculate variance of entropy
	mean := 0.0
	for _, e := range entropies {
		mean += e
	}
	mean /= float64(len(entropies))

	variance := 0.0
	for _, e := range entropies {
		diff := e - mean
		variance += diff * diff
	}
	variance /= float64(len(entropies))

	// High variance in entropy might indicate AI-generated content
	// (inconsistent compression)
	if variance > 0.1 {
		return 0.6
	}

	return 0.4
}

// calculateWeightedScore combines signals into final score.
func (a *VideoAnalyzer) calculateWeightedScore(signals VideoSignals) float64 {
	w := a.weights

	score := signals.MetadataScore*w.MetadataScore +
		signals.ContainerAnalysis*w.ContainerAnalysis +
		signals.AudioPresence*w.AudioPresence +
		signals.TemporalPattern*w.TemporalPattern +
		signals.EncodingSignature*w.EncodingSignature +
		signals.BitrateConsistency*w.BitrateConsistency

	totalWeight := w.MetadataScore + w.ContainerAnalysis + w.AudioPresence +
		w.TemporalPattern + w.EncodingSignature + w.BitrateConsistency

	if totalWeight > 0 {
		score /= totalWeight
	}

	return math.Max(0, math.Min(1, score))
}

// =============================================================================
// Helper Functions
// =============================================================================

// containsAIMarker checks for known AI video generator markers.
func containsAIMarker(s string) bool {
	markers := []string{
		"runway", "pika", "sora", "gen-2", "gen2",
		"stable video", "stablevideo", "modelscope",
		"deforum", "animatediff", "zeroscope",
		"ai generated", "ai-generated", "synthetic",
		"dall-e", "midjourney", // Sometimes in video metadata
	}

	lower := bytes.ToLower([]byte(s))
	for _, marker := range markers {
		if bytes.Contains(lower, []byte(marker)) {
			return true
		}
	}
	return false
}

// extractEncoder tries to find encoder name in metadata string.
func extractEncoder(s string) string {
	// Common encoder identifiers
	encoders := map[string]string{
		"lavf":        "ffmpeg",
		"ffmpeg":      "ffmpeg",
		"handbrake":   "HandBrake",
		"premiere":    "Adobe Premiere",
		"final cut":   "Final Cut Pro",
		"davinci":     "DaVinci Resolve",
		"x264":        "x264",
		"x265":        "x265",
		"runway":      "Runway",
		"pika":        "Pika Labs",
		"sora":        "OpenAI Sora",
		"modelscope":  "ModelScope",
	}

	lower := bytes.ToLower([]byte(s))
	for key, name := range encoders {
		if bytes.Contains(lower, []byte(key)) {
			return name
		}
	}

	return ""
}

// calculateEntropy computes Shannon entropy of byte data.
func calculateEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0
	}

	freq := make([]int, 256)
	for _, b := range data {
		freq[b]++
	}

	entropy := 0.0
	total := float64(len(data))

	for _, f := range freq {
		if f > 0 {
			p := float64(f) / total
			entropy -= p * math.Log2(p)
		}
	}

	// Normalize to 0-1 (max entropy for bytes is 8)
	return entropy / 8.0
}

// min is now a builtin in Go 1.21+, so we use the builtin version.

// =============================================================================
// Video Frame Extraction
// =============================================================================

// DefaultFrameExtractionConfig returns default frame extraction settings.
func DefaultFrameExtractionConfig() FrameExtractionConfig {
	return FrameExtractionConfig{
		Enabled:       true,
		MaxFrames:     10,
		KeyframesOnly: true,
		SampleRate:    1,
	}
}

// ExtractFrames extracts video frames from container data.
func (a *VideoAnalyzer) ExtractFrames(data []byte, config FrameExtractionConfig) []ExtractedFrame {
	if !config.Enabled || len(data) < 100 {
		return nil
	}

	format := a.detectVideoFormat(data)

	switch format {
	case "mp4", "mov":
		return a.extractMP4Frames(data, config)
	case "webm", "mkv":
		return a.extractWebMFrames(data, config)
	case "avi":
		return a.extractAVIFrames(data, config)
	default:
		return a.extractGenericH264Frames(data, config)
	}
}

// extractMP4Frames extracts frames from MP4/MOV containers.
func (a *VideoAnalyzer) extractMP4Frames(data []byte, config FrameExtractionConfig) []ExtractedFrame {
	var frames []ExtractedFrame

	// Find mdat atom
	mdatOffset, mdatSize := a.findMP4Atom(data, "mdat")
	if mdatOffset < 0 || mdatSize < 100 {
		return frames
	}

	mdatData := data[mdatOffset:min(mdatOffset+mdatSize, len(data))]

	// Look for stss atom (sync sample table) for keyframe indices
	keyframeIndices := a.findMP4KeyframeIndices(data)

	// Extract NAL units from mdat
	nalUnits := a.extractNALUnits(mdatData)

	frameIndex := 0
	extractedCount := 0
	sampleCounter := 0

	for _, nal := range nalUnits {
		isKeyframe := nal.NALType == 5 || nal.NALType == 19 || nal.NALType == 20 // IDR for H.264/H.265

		// Check if this frame should be extracted
		shouldExtract := false
		if config.KeyframesOnly {
			shouldExtract = isKeyframe
		} else {
			shouldExtract = true
		}

		// Also check against stss keyframe indices if available
		if len(keyframeIndices) > 0 {
			inKeyframeList := false
			for _, kfIdx := range keyframeIndices {
				if kfIdx == frameIndex {
					inKeyframeList = true
					break
				}
			}
			if config.KeyframesOnly && !inKeyframeList {
				shouldExtract = false
			}
		}

		if shouldExtract {
			sampleCounter++
			if sampleCounter >= config.SampleRate {
				sampleCounter = 0

				frame := ExtractedFrame{
					Index:     frameIndex,
					FrameType: 'I',
					Offset:    mdatOffset + nal.Offset,
					Size:      nal.Size,
					Data:      mdatData[nal.Offset:min(nal.Offset+nal.Size, len(mdatData))],
				}

				if !isKeyframe {
					frame.FrameType = 'P'
				}

				// Extract dimensions from SPS if this is a keyframe
				if isKeyframe {
					frame.Width, frame.Height = a.extractDimensionsFromNAL(frame.Data)
				}

				frames = append(frames, frame)
				extractedCount++

				if extractedCount >= config.MaxFrames {
					break
				}
			}
		}

		frameIndex++
	}

	return frames
}

// NALUnit represents a parsed NAL unit.
type NALUnit struct {
	Offset  int
	Size    int
	NALType int
}

// extractNALUnits finds NAL units in H.264/H.265 data.
func (a *VideoAnalyzer) extractNALUnits(data []byte) []NALUnit {
	var nalUnits []NALUnit

	i := 0
	for i < len(data)-4 {
		// Look for start code
		startCodeLen := 0
		if data[i] == 0 && data[i+1] == 0 && data[i+2] == 1 {
			startCodeLen = 3
		} else if data[i] == 0 && data[i+1] == 0 && data[i+2] == 0 && data[i+3] == 1 {
			startCodeLen = 4
		}

		if startCodeLen == 0 {
			i++
			continue
		}

		nalStart := i + startCodeLen
		if nalStart >= len(data) {
			break
		}

		// Get NAL type
		nalHeader := data[nalStart]
		nalType := int(nalHeader & 0x1F) // H.264 NAL type is 5 bits

		// Find end of NAL unit
		nalEnd := len(data)
		for j := nalStart + 1; j < len(data)-3; j++ {
			if data[j] == 0 && data[j+1] == 0 &&
				(data[j+2] == 1 || (data[j+2] == 0 && j+3 < len(data) && data[j+3] == 1)) {
				nalEnd = j
				break
			}
		}

		nal := NALUnit{
			Offset:  i,
			Size:    nalEnd - i,
			NALType: nalType,
		}

		nalUnits = append(nalUnits, nal)
		i = nalEnd

		// Limit to prevent excessive processing
		if len(nalUnits) > 1000 {
			break
		}
	}

	return nalUnits
}

// findMP4KeyframeIndices extracts keyframe indices from stss atom.
func (a *VideoAnalyzer) findMP4KeyframeIndices(data []byte) []int {
	var indices []int

	// Find stss atom (sync sample table)
	stssOffset := -1
	for i := 0; i < len(data)-8; i++ {
		if bytes.Equal(data[i:i+4], []byte("stss")) {
			stssOffset = i
			break
		}
	}

	if stssOffset < 0 {
		return indices
	}

	// Parse stss atom
	// Format: version(1) + flags(3) + entry_count(4) + entries(4 each)
	if stssOffset+12 > len(data) {
		return indices
	}

	entryCount := int(binary.BigEndian.Uint32(data[stssOffset+8 : stssOffset+12]))

	// Limit entries
	if entryCount > 1000 {
		entryCount = 1000
	}

	for i := 0; i < entryCount && stssOffset+12+i*4+4 <= len(data); i++ {
		sampleNum := int(binary.BigEndian.Uint32(data[stssOffset+12+i*4 : stssOffset+16+i*4]))
		indices = append(indices, sampleNum-1) // Convert to 0-based index
	}

	return indices
}

// extractDimensionsFromNAL extracts video dimensions from NAL unit (SPS).
func (a *VideoAnalyzer) extractDimensionsFromNAL(data []byte) (int, int) {
	// Look for SPS NAL unit (type 7 for H.264)
	for i := 0; i < len(data)-10; i++ {
		// Find start code
		if data[i] == 0 && data[i+1] == 0 && (data[i+2] == 1 || (data[i+2] == 0 && data[i+3] == 1)) {
			startLen := 3
			if data[i+2] == 0 {
				startLen = 4
			}

			nalStart := i + startLen
			if nalStart >= len(data) {
				continue
			}

			nalType := data[nalStart] & 0x1F
			if nalType == 7 { // SPS
				return a.parseSPSDimensions(data[nalStart:])
			}
		}
	}

	return 0, 0
}

// parseSPSDimensions parses SPS to extract dimensions.
// This is a simplified parser - full SPS parsing is complex.
func (a *VideoAnalyzer) parseSPSDimensions(sps []byte) (int, int) {
	if len(sps) < 10 {
		return 0, 0
	}

	// SPS parsing is complex due to exp-Golomb encoding
	// This is a simplified heuristic approach

	// Look for common resolution patterns in the SPS
	// Try to find pic_width_in_mbs and pic_height_in_map_units

	// Common resolutions as fallback detection
	commonResolutions := [][2]int{
		{1920, 1080}, {1280, 720}, {854, 480}, {640, 480},
		{3840, 2160}, {2560, 1440}, {1080, 1920}, {720, 1280},
	}

	// Look for resolution indicators in SPS
	// This is heuristic - actual SPS parsing requires bit-by-bit exp-Golomb decoding
	for _, res := range commonResolutions {
		mbWidth := (res[0] + 15) / 16
		mbHeight := (res[1] + 15) / 16

		// Check if these macroblock counts appear in SPS
		for i := 0; i < len(sps)-2; i++ {
			val := int(sps[i])
			if val == mbWidth || val == mbHeight {
				// Rough match found
				return res[0], res[1]
			}
		}
	}

	return 0, 0
}

// extractWebMFrames extracts frames from WebM/MKV containers.
func (a *VideoAnalyzer) extractWebMFrames(data []byte, config FrameExtractionConfig) []ExtractedFrame {
	var frames []ExtractedFrame

	// Find SimpleBlock elements containing keyframes
	// SimpleBlock header: track number (variable) + timecode (2 bytes) + flags (1 byte)
	// Keyframe is indicated by flag bit

	extractedCount := 0
	sampleCounter := 0

	for i := 0; i < len(data)-10 && extractedCount < config.MaxFrames; i++ {
		// Look for SimpleBlock element ID (0xA3)
		if data[i] == 0xA3 {
			// Read EBML size
			size := a.readEBMLSize(data[i+1:])
			if size <= 0 || size > 10000000 || i+1+size > len(data) {
				continue
			}

			sizeLen := a.ebmlSizeLength(data[i+1:])
			blockStart := i + 1 + sizeLen

			if blockStart+4 > len(data) {
				continue
			}

			// Parse SimpleBlock header
			// First byte is track number (variable length)
			// Then 2 bytes timecode, 1 byte flags
			trackNumLen := 1
			if data[blockStart]&0x80 == 0 {
				trackNumLen = 2
			}

			if blockStart+trackNumLen+3 > len(data) {
				continue
			}

			flags := data[blockStart+trackNumLen+2]
			isKeyframe := (flags & 0x80) != 0

			if config.KeyframesOnly && !isKeyframe {
				continue
			}

			sampleCounter++
			if sampleCounter < config.SampleRate {
				continue
			}
			sampleCounter = 0

			frame := ExtractedFrame{
				Index:     extractedCount,
				FrameType: 'I',
				Offset:    blockStart,
				Size:      size,
				Data:      data[blockStart:min(blockStart+size, len(data))],
			}

			if !isKeyframe {
				frame.FrameType = 'P'
			}

			frames = append(frames, frame)
			extractedCount++
		}
	}

	return frames
}

// ebmlSizeLength returns the length of an EBML size field.
func (a *VideoAnalyzer) ebmlSizeLength(data []byte) int {
	if len(data) == 0 {
		return 0
	}

	first := data[0]
	if first&0x80 != 0 {
		return 1
	}
	if first&0x40 != 0 {
		return 2
	}
	if first&0x20 != 0 {
		return 3
	}
	if first&0x10 != 0 {
		return 4
	}
	return 5
}

// extractAVIFrames extracts frames from AVI containers.
func (a *VideoAnalyzer) extractAVIFrames(data []byte, config FrameExtractionConfig) []ExtractedFrame {
	var frames []ExtractedFrame

	extractedCount := 0
	sampleCounter := 0
	frameIndex := 0

	// Look for video chunks (##dc for compressed, ##db for uncompressed)
	for i := 0; i < len(data)-8 && extractedCount < config.MaxFrames; i++ {
		// Video stream 0 or 1
		isVideoChunk := (data[i] == '0' && (data[i+1] == '0' || data[i+1] == '1') &&
			data[i+2] == 'd' && (data[i+3] == 'c' || data[i+3] == 'b'))

		if !isVideoChunk {
			continue
		}

		chunkSize := int(binary.LittleEndian.Uint32(data[i+4 : i+8]))
		if chunkSize <= 0 || chunkSize > 10000000 || i+8+chunkSize > len(data) {
			continue
		}

		// In AVI, keyframes are typically marked in the idx1 chunk
		// For simplicity, we'll use heuristics: larger frames are more likely keyframes
		// Also, first frame is always a keyframe

		isKeyframe := frameIndex == 0 || chunkSize > 50000

		if config.KeyframesOnly && !isKeyframe {
			frameIndex++
			continue
		}

		sampleCounter++
		if sampleCounter < config.SampleRate {
			frameIndex++
			continue
		}
		sampleCounter = 0

		frame := ExtractedFrame{
			Index:     frameIndex,
			FrameType: 'I',
			Offset:    i + 8,
			Size:      chunkSize,
			Data:      data[i+8 : i+8+chunkSize],
		}

		if !isKeyframe {
			frame.FrameType = 'P'
		}

		frames = append(frames, frame)
		extractedCount++
		frameIndex++
	}

	return frames
}

// extractGenericH264Frames extracts H.264 frames from raw NAL stream.
func (a *VideoAnalyzer) extractGenericH264Frames(data []byte, config FrameExtractionConfig) []ExtractedFrame {
	var frames []ExtractedFrame

	nalUnits := a.extractNALUnits(data)

	extractedCount := 0
	sampleCounter := 0

	for _, nal := range nalUnits {
		// IDR = keyframe (type 5), also types 19/20 for H.265
		isKeyframe := nal.NALType == 5 || nal.NALType == 19 || nal.NALType == 20

		// Non-IDR slices
		isFrame := nal.NALType == 1 || nal.NALType == 5 || nal.NALType == 19 || nal.NALType == 20

		if !isFrame {
			continue
		}

		if config.KeyframesOnly && !isKeyframe {
			continue
		}

		sampleCounter++
		if sampleCounter < config.SampleRate {
			continue
		}
		sampleCounter = 0

		frame := ExtractedFrame{
			Index:     extractedCount,
			FrameType: 'I',
			Offset:    nal.Offset,
			Size:      nal.Size,
			Data:      data[nal.Offset:min(nal.Offset+nal.Size, len(data))],
		}

		if !isKeyframe {
			frame.FrameType = 'P'
		}

		frames = append(frames, frame)
		extractedCount++

		if extractedCount >= config.MaxFrames {
			break
		}
	}

	return frames
}

// AnalyzeExtractedFrames applies analysis to extracted frames.
func (a *VideoAnalyzer) AnalyzeExtractedFrames(frames []ExtractedFrame, imageAnalyzer *ImageAnalyzer) []ImageAnalysisResult {
	var results []ImageAnalysisResult

	for _, frame := range frames {
		if len(frame.Data) < 100 {
			continue
		}

		// Analyze the frame data as an image
		// Note: Raw NAL data needs decoding for full image analysis
		// This provides container-level forensics on the frame data
		result := imageAnalyzer.Analyze(frame.Data)
		results = append(results, result)
	}

	return results
}

// ComputeInterFrameConsistency measures consistency between extracted frames.
func (a *VideoAnalyzer) ComputeInterFrameConsistency(frames []ExtractedFrame) float64 {
	if len(frames) < 2 {
		return 0.5 // Neutral when insufficient frames
	}

	// Compute various consistency metrics
	var sizeVariance float64
	var entropyVariance float64

	// Calculate mean size
	sumSize := 0
	for _, f := range frames {
		sumSize += f.Size
	}
	meanSize := float64(sumSize) / float64(len(frames))

	// Calculate size variance
	for _, f := range frames {
		diff := float64(f.Size) - meanSize
		sizeVariance += diff * diff
	}
	sizeVariance /= float64(len(frames))

	// Calculate entropy variance
	entropies := make([]float64, len(frames))
	sumEntropy := 0.0
	for i, f := range frames {
		entropies[i] = calculateEntropy(f.Data)
		sumEntropy += entropies[i]
	}
	meanEntropy := sumEntropy / float64(len(frames))

	for _, e := range entropies {
		diff := e - meanEntropy
		entropyVariance += diff * diff
	}
	entropyVariance /= float64(len(frames))

	// Normalize variance to 0-1 score
	// High variance = more natural (human video)
	// Low variance = potentially AI (too consistent)
	sizeCV := 0.0
	if meanSize > 0 {
		sizeCV = math.Sqrt(sizeVariance) / meanSize
	}

	entropyCV := 0.0
	if meanEntropy > 0 {
		entropyCV = math.Sqrt(entropyVariance) / meanEntropy
	}

	// Combine scores
	// Lower variance = higher AI score
	score := 0.5
	if sizeCV < 0.1 {
		score += 0.1 // Very consistent sizes
	} else if sizeCV > 0.5 {
		score -= 0.1 // Natural variation
	}

	if entropyCV < 0.05 {
		score += 0.1 // Very consistent entropy
	} else if entropyCV > 0.2 {
		score -= 0.1 // Natural variation
	}

	return math.Max(0, math.Min(1, score))
}

// =============================================================================
// Temporal Consistency Analysis
// =============================================================================

// TemporalAnalysisResult contains comprehensive temporal consistency metrics.
type TemporalAnalysisResult struct {
	OverallScore        float64   // Combined AI likelihood score (0=human, 1=AI)
	FlickeringScore     float64   // Unusual flickering patterns
	FrameDiffVariance   float64   // Variance in frame-to-frame differences
	LightingScore       float64   // Lighting consistency analysis
	MorphingScore       float64   // Morphing artifact detection
	SizeDeltaPattern    []float64 // Frame size changes over time
	EntropyDeltaPattern []float64 // Entropy changes over time
}

// AnalyzeTemporalConsistency performs comprehensive temporal analysis on frames.
func (a *VideoAnalyzer) AnalyzeTemporalConsistency(frames []ExtractedFrame) TemporalAnalysisResult {
	result := TemporalAnalysisResult{
		OverallScore: 0.5,
	}

	if len(frames) < 3 {
		return result
	}

	// Compute frame-to-frame differences
	frameDiffs := a.computeFrameDifferences(frames)

	// Analyze flickering (rapid intensity changes)
	result.FlickeringScore = a.analyzeFlickering(frameDiffs)

	// Analyze frame difference variance
	result.FrameDiffVariance = a.computeDiffVariance(frameDiffs)

	// Analyze lighting consistency
	result.LightingScore = a.analyzeLightingConsistency(frames)

	// Detect morphing artifacts
	result.MorphingScore = a.detectMorphingArtifacts(frames, frameDiffs)

	// Compute size and entropy delta patterns
	result.SizeDeltaPattern, result.EntropyDeltaPattern = a.computeDeltaPatterns(frames)

	// Combine into overall score
	result.OverallScore = a.computeTemporalOverallScore(result)

	return result
}

// FrameDifference represents differences between consecutive frames.
type FrameDifference struct {
	FrameA      int     // First frame index
	FrameB      int     // Second frame index
	SizeDelta   int     // Size difference
	EntropyDiff float64 // Entropy difference
	ByteDiff    float64 // Proportion of different bytes
	HistDiff    float64 // Histogram difference
}

// computeFrameDifferences calculates differences between consecutive frames.
func (a *VideoAnalyzer) computeFrameDifferences(frames []ExtractedFrame) []FrameDifference {
	var diffs []FrameDifference

	for i := 1; i < len(frames); i++ {
		prevFrame := frames[i-1]
		currFrame := frames[i]

		diff := FrameDifference{
			FrameA:    i - 1,
			FrameB:    i,
			SizeDelta: currFrame.Size - prevFrame.Size,
		}

		// Entropy difference
		prevEntropy := calculateEntropy(prevFrame.Data)
		currEntropy := calculateEntropy(currFrame.Data)
		diff.EntropyDiff = currEntropy - prevEntropy

		// Byte-level difference (sampled for performance)
		diff.ByteDiff = a.computeByteDifference(prevFrame.Data, currFrame.Data)

		// Histogram difference
		diff.HistDiff = a.computeHistogramDifference(prevFrame.Data, currFrame.Data)

		diffs = append(diffs, diff)
	}

	return diffs
}

// computeByteDifference calculates proportion of differing bytes.
func (a *VideoAnalyzer) computeByteDifference(dataA, dataB []byte) float64 {
	minLen := min(len(dataA), len(dataB))
	if minLen == 0 {
		return 0
	}

	// Sample for performance
	sampleSize := min(minLen, 10000)
	step := minLen / sampleSize

	differences := 0
	comparisons := 0

	for i := 0; i < minLen; i += step {
		if dataA[i] != dataB[i] {
			differences++
		}
		comparisons++
	}

	if comparisons == 0 {
		return 0
	}

	return float64(differences) / float64(comparisons)
}

// computeHistogramDifference compares byte histograms between frames.
func (a *VideoAnalyzer) computeHistogramDifference(dataA, dataB []byte) float64 {
	histA := make([]int, 256)
	histB := make([]int, 256)

	for _, b := range dataA {
		histA[b]++
	}
	for _, b := range dataB {
		histB[b]++
	}

	// Chi-square distance between histograms
	chiSquare := 0.0
	for i := 0; i < 256; i++ {
		expected := float64(histA[i]+histB[i]) / 2
		if expected > 0 {
			diffA := float64(histA[i]) - expected
			diffB := float64(histB[i]) - expected
			chiSquare += (diffA*diffA + diffB*diffB) / expected
		}
	}

	// Normalize to 0-1
	maxChiSquare := float64(len(dataA) + len(dataB))
	if maxChiSquare > 0 {
		return math.Min(1, chiSquare/maxChiSquare)
	}
	return 0
}

// analyzeFlickering detects unusual flickering patterns.
func (a *VideoAnalyzer) analyzeFlickering(diffs []FrameDifference) float64 {
	if len(diffs) < 2 {
		return 0.5
	}

	// Look for rapid oscillations in brightness/entropy
	oscillations := 0

	for i := 1; i < len(diffs); i++ {
		// Check for sign changes in entropy difference
		if (diffs[i-1].EntropyDiff > 0) != (diffs[i].EntropyDiff > 0) {
			oscillations++
		}
	}

	oscillationRate := float64(oscillations) / float64(len(diffs)-1)

	// High oscillation rate indicates flickering (AI artifact)
	if oscillationRate > 0.8 {
		return 0.7 // Strong AI indicator
	}
	if oscillationRate > 0.6 {
		return 0.6
	}
	if oscillationRate < 0.3 {
		return 0.3 // More natural
	}

	return 0.5
}

// computeDiffVariance calculates variance in frame differences.
func (a *VideoAnalyzer) computeDiffVariance(diffs []FrameDifference) float64 {
	if len(diffs) < 2 {
		return 0
	}

	// Compute variance of byte differences
	sum := 0.0
	for _, d := range diffs {
		sum += d.ByteDiff
	}
	mean := sum / float64(len(diffs))

	variance := 0.0
	for _, d := range diffs {
		diff := d.ByteDiff - mean
		variance += diff * diff
	}
	variance /= float64(len(diffs))

	return variance
}

// analyzeLightingConsistency checks for unnatural lighting changes.
func (a *VideoAnalyzer) analyzeLightingConsistency(frames []ExtractedFrame) float64 {
	if len(frames) < 3 {
		return 0.5
	}

	// Compute average byte value as proxy for brightness
	brightnesses := make([]float64, len(frames))
	for i, f := range frames {
		sum := int64(0)
		for _, b := range f.Data {
			sum += int64(b)
		}
		if len(f.Data) > 0 {
			brightnesses[i] = float64(sum) / float64(len(f.Data))
		}
	}

	// Compute variance in brightness
	sum := 0.0
	for _, b := range brightnesses {
		sum += b
	}
	mean := sum / float64(len(brightnesses))

	variance := 0.0
	for _, b := range brightnesses {
		diff := b - mean
		variance += diff * diff
	}
	variance /= float64(len(brightnesses))

	// Compute coefficient of variation
	cv := 0.0
	if mean > 0 {
		cv = math.Sqrt(variance) / mean
	}

	// Very low variance = suspiciously consistent lighting (AI)
	// Very high variance = also potentially AI (flickering)
	if cv < 0.01 {
		return 0.65 // Too consistent
	}
	if cv > 0.3 {
		return 0.6 // Too variable (flickering)
	}
	if cv > 0.02 && cv < 0.15 {
		return 0.35 // Natural variation
	}

	return 0.5
}

// detectMorphingArtifacts looks for morphing artifacts common in AI video.
func (a *VideoAnalyzer) detectMorphingArtifacts(frames []ExtractedFrame, diffs []FrameDifference) float64 {
	if len(diffs) < 3 {
		return 0.5
	}

	// Morphing artifacts often show:
	// 1. Gradual, consistent changes between frames
	// 2. Very low frame difference variance (too smooth transitions)
	// 3. Unusual patterns in size changes

	// Check for suspiciously smooth transitions
	smoothCount := 0
	for _, d := range diffs {
		// Very low byte difference = suspiciously smooth
		if d.ByteDiff < 0.1 {
			smoothCount++
		}
	}

	smoothRatio := float64(smoothCount) / float64(len(diffs))

	// Check for gradual size changes (morphing often creates consistent deltas)
	sizeDeltaVariance := 0.0
	sumDelta := 0.0
	for _, d := range diffs {
		sumDelta += float64(d.SizeDelta)
	}
	meanDelta := sumDelta / float64(len(diffs))

	for _, d := range diffs {
		diff := float64(d.SizeDelta) - meanDelta
		sizeDeltaVariance += diff * diff
	}
	sizeDeltaVariance /= float64(len(diffs))

	// Very low size delta variance with smooth transitions = morphing
	score := 0.5
	if smoothRatio > 0.7 {
		score += 0.1
	}
	if sizeDeltaVariance < 1000 && smoothRatio > 0.5 {
		score += 0.1
	}

	return math.Max(0, math.Min(1, score))
}

// computeDeltaPatterns extracts temporal patterns in frame changes.
func (a *VideoAnalyzer) computeDeltaPatterns(frames []ExtractedFrame) ([]float64, []float64) {
	sizeDeltas := make([]float64, 0, len(frames)-1)
	entropyDeltas := make([]float64, 0, len(frames)-1)

	for i := 1; i < len(frames); i++ {
		sizeDeltas = append(sizeDeltas, float64(frames[i].Size-frames[i-1].Size))

		entropyA := calculateEntropy(frames[i-1].Data)
		entropyB := calculateEntropy(frames[i].Data)
		entropyDeltas = append(entropyDeltas, entropyB-entropyA)
	}

	return sizeDeltas, entropyDeltas
}

// computeTemporalOverallScore combines temporal metrics into single score.
func (a *VideoAnalyzer) computeTemporalOverallScore(result TemporalAnalysisResult) float64 {
	// Weight the different signals
	weights := map[string]float64{
		"flickering": 0.25,
		"lighting":   0.20,
		"morphing":   0.25,
		"diffVar":    0.30,
	}

	score := result.FlickeringScore*weights["flickering"] +
		result.LightingScore*weights["lighting"] +
		result.MorphingScore*weights["morphing"]

	// Add diff variance contribution
	// Low variance = suspicious
	if result.FrameDiffVariance < 0.001 {
		score += weights["diffVar"] * 0.7
	} else if result.FrameDiffVariance < 0.01 {
		score += weights["diffVar"] * 0.6
	} else if result.FrameDiffVariance > 0.1 {
		score += weights["diffVar"] * 0.4
	} else {
		score += weights["diffVar"] * 0.5
	}

	return math.Max(0, math.Min(1, score))
}
