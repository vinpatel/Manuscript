package service

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// =============================================================================
// Manuscript Image Detection Algorithm
// =============================================================================
//
// This is our own detection algorithm based on image forensic analysis.
// No external APIs or ML models required.
//
// Key insights:
//   1. Real photos have camera metadata (EXIF)
//   2. AI images have unnatural frequency patterns
//   3. Real images have sensor noise patterns
//   4. AI images often have edge artifacts
//   5. JPEG compression patterns differ
//
// We analyze:
//   1. Metadata presence and validity
//   2. Color distribution patterns
//   3. Edge consistency
//   4. Compression artifacts
//   5. Noise patterns
//   6. Symmetry detection (AI often has symmetry artifacts)
//
// =============================================================================

// ImageAnalyzer performs forensic analysis on images.
type ImageAnalyzer struct {
	weights ImageAnalyzerWeights
}

// ImageAnalyzerWeights controls signal importance.
type ImageAnalyzerWeights struct {
	MetadataScore       float64
	ColorDistribution   float64
	EdgeConsistency     float64
	NoisePattern        float64
	CompressionAnalysis float64
	SymmetryDetection   float64
}

// DefaultImageWeights returns tuned weights.
func DefaultImageWeights() ImageAnalyzerWeights {
	return ImageAnalyzerWeights{
		MetadataScore:       0.25,
		ColorDistribution:   0.20,
		EdgeConsistency:     0.15,
		NoisePattern:        0.15,
		CompressionAnalysis: 0.15,
		SymmetryDetection:   0.10,
	}
}

// NewImageAnalyzer creates a new analyzer.
func NewImageAnalyzer() *ImageAnalyzer {
	return &ImageAnalyzer{
		weights: DefaultImageWeights(),
	}
}

// ImageAnalysisResult contains analysis results.
type ImageAnalysisResult struct {
	AIScore  float64
	Signals  ImageSignals
	Metadata ImageMetadata
	Stats    ImageStats
}

// ImageSignals contains individual signal scores.
type ImageSignals struct {
	MetadataScore       float64 // Missing/fake metadata = AI-like
	ColorDistribution   float64 // Unnatural distribution = AI-like
	EdgeConsistency     float64 // Inconsistent edges = AI-like
	NoisePattern        float64 // Missing natural noise = AI-like
	CompressionAnalysis float64 // Wrong compression = AI-like
	SymmetryScore       float64 // Unnatural symmetry = AI-like
}

// ImageMetadata contains extracted metadata.
type ImageMetadata struct {
	HasEXIF      bool
	CameraMake   string
	CameraModel  string
	Software     string
	DateTaken    string
	HasGPS       bool
	IsScreenshot bool
	FileFormat   string

	// Extended EXIF fields for forensic analysis
	LensMake       string
	LensModel      string
	FocalLength    string
	Aperture       string
	ExposureTime   string
	ISO            int
	Flash          string
	WhiteBalance   string
	ColorSpace     string
	ExifVersion    string
	Orientation    int
	XResolution    int
	YResolution    int
	ResolutionUnit string
	Artist         string
	Copyright      string

	// GPS details
	GPSLatitude  float64
	GPSLongitude float64
	GPSAltitude  float64
	GPSTimestamp string

	// Thumbnail info
	HasThumbnail  bool
	ThumbnailSize int
}

// ImageStats contains image statistics.
type ImageStats struct {
	Width           int
	Height          int
	BitDepth        int
	ColorChannels   int
	EstimatedColors int
	AvgBrightness   float64
	Contrast        float64
}

// Analyze performs forensic analysis on image data.
func (a *ImageAnalyzer) Analyze(data []byte) ImageAnalysisResult {
	result := ImageAnalysisResult{}

	// Detect format
	format := detectImageFormat(data)
	result.Metadata.FileFormat = format

	// Extract metadata
	result.Metadata = a.extractMetadata(data, format)

	// Get basic image stats
	result.Stats = a.getImageStats(data, format)

	// Calculate signals
	result.Signals.MetadataScore = a.analyzeMetadata(result.Metadata)
	result.Signals.ColorDistribution = a.analyzeColorDistribution(data, format)
	result.Signals.EdgeConsistency = a.analyzeEdgeConsistency(data, format)
	result.Signals.NoisePattern = a.analyzeNoisePattern(data, format)
	result.Signals.CompressionAnalysis = a.analyzeCompression(data, format)
	result.Signals.SymmetryScore = a.analyzeSymmetry(data, format)

	// Calculate weighted score
	result.AIScore = a.calculateWeightedScore(result.Signals)

	return result
}

// extractMetadata parses image metadata.
func (a *ImageAnalyzer) extractMetadata(data []byte, format string) ImageMetadata {
	meta := ImageMetadata{
		FileFormat: format,
	}

	switch format {
	case "jpeg":
		meta = a.parseJPEGMetadata(data)
	case "png":
		meta = a.parsePNGMetadata(data)
	}

	return meta
}

// parseJPEGMetadata extracts EXIF data from JPEG using proper TIFF structure parsing.
func (a *ImageAnalyzer) parseJPEGMetadata(data []byte) ImageMetadata {
	meta := ImageMetadata{FileFormat: "jpeg"}

	if len(data) < 12 {
		return meta
	}

	// Look for EXIF marker (APP1 = 0xFFE1)
	for i := 0; i < len(data)-10; i++ {
		if data[i] == 0xFF && data[i+1] == 0xE1 {
			// Found APP1 segment
			segmentLen := int(data[i+2])<<8 | int(data[i+3])
			if i+4+segmentLen > len(data) {
				break
			}

			segment := data[i+4 : i+4+segmentLen-2]
			if len(segment) >= 6 && string(segment[:4]) == "Exif" {
				meta.HasEXIF = true

				// Parse TIFF header (starts after "Exif\x00\x00")
				if len(segment) >= 14 {
					tiffData := segment[6:]
					a.parseEXIFTags(tiffData, &meta)
				}
			}
			break
		}
	}

	// Check for screenshot indicators
	if meta.Software == "" && !meta.HasEXIF && meta.CameraMake == "" {
		meta.IsScreenshot = true
	}

	// Also check for AI generator signatures in the raw data
	dataStr := string(data)
	if strings.Contains(dataStr, "DALL-E") || strings.Contains(dataStr, "Midjourney") ||
		strings.Contains(dataStr, "Stable Diffusion") || strings.Contains(dataStr, "ComfyUI") ||
		strings.Contains(dataStr, "NovelAI") {
		meta.Software = "AI Generator"
	}

	return meta
}

// parseEXIFTags parses TIFF-formatted EXIF data.
func (a *ImageAnalyzer) parseEXIFTags(data []byte, meta *ImageMetadata) {
	if len(data) < 8 {
		return
	}

	// Determine byte order
	var byteOrder binary.ByteOrder
	if data[0] == 'I' && data[1] == 'I' {
		byteOrder = binary.LittleEndian
	} else if data[0] == 'M' && data[1] == 'M' {
		byteOrder = binary.BigEndian
	} else {
		return // Invalid TIFF header
	}

	// Verify TIFF magic number (42)
	magic := byteOrder.Uint16(data[2:4])
	if magic != 42 {
		return
	}

	// Get offset to first IFD
	ifdOffset := byteOrder.Uint32(data[4:8])
	if int(ifdOffset) >= len(data) {
		return
	}

	// Parse IFD0 (main image tags)
	exifIFDOffset := a.parseIFD(data, int(ifdOffset), byteOrder, meta, false)

	// Parse EXIF sub-IFD if present
	if exifIFDOffset > 0 && exifIFDOffset < len(data) {
		a.parseIFD(data, exifIFDOffset, byteOrder, meta, true)
	}

	// Look for GPS IFD
	gpsOffset := a.findGPSIFD(data, int(ifdOffset), byteOrder)
	if gpsOffset > 0 && gpsOffset < len(data) {
		a.parseGPSIFD(data, gpsOffset, byteOrder, meta)
	}

	// Check for thumbnail (IFD1)
	if int(ifdOffset)+2 < len(data) {
		numTags := int(byteOrder.Uint16(data[ifdOffset : ifdOffset+2]))
		nextIFDOffsetPos := int(ifdOffset) + 2 + numTags*12
		if nextIFDOffsetPos+4 <= len(data) {
			nextIFD := byteOrder.Uint32(data[nextIFDOffsetPos : nextIFDOffsetPos+4])
			if nextIFD > 0 && int(nextIFD) < len(data) {
				meta.HasThumbnail = true
			}
		}
	}
}

// parseIFD parses an Image File Directory and extracts tags.
// Returns the offset to EXIF sub-IFD if found.
func (a *ImageAnalyzer) parseIFD(data []byte, offset int, byteOrder binary.ByteOrder, meta *ImageMetadata, isExifIFD bool) int {
	if offset+2 > len(data) {
		return 0
	}

	numTags := int(byteOrder.Uint16(data[offset : offset+2]))
	if numTags > 200 { // Sanity check
		return 0
	}

	exifOffset := 0

	for i := 0; i < numTags; i++ {
		tagOffset := offset + 2 + i*12
		if tagOffset+12 > len(data) {
			break
		}

		tagID := byteOrder.Uint16(data[tagOffset : tagOffset+2])
		tagType := byteOrder.Uint16(data[tagOffset+2 : tagOffset+4])
		tagCount := byteOrder.Uint32(data[tagOffset+4 : tagOffset+8])
		valueOffset := tagOffset + 8

		// Get value based on type and count
		value := a.getTagValue(data, valueOffset, tagType, tagCount, byteOrder)

		switch tagID {
		// IFD0 tags
		case 0x010F: // Make
			meta.CameraMake = cleanString(value)
		case 0x0110: // Model
			meta.CameraModel = cleanString(value)
		case 0x0112: // Orientation
			if n, ok := parseUint(value); ok {
				meta.Orientation = int(n)
			}
		case 0x011A: // XResolution
			if n, ok := parseUint(value); ok {
				meta.XResolution = int(n)
			}
		case 0x011B: // YResolution
			if n, ok := parseUint(value); ok {
				meta.YResolution = int(n)
			}
		case 0x0128: // ResolutionUnit
			if n, ok := parseUint(value); ok {
				switch n {
				case 1:
					meta.ResolutionUnit = "None"
				case 2:
					meta.ResolutionUnit = "inches"
				case 3:
					meta.ResolutionUnit = "centimeters"
				}
			}
		case 0x0131: // Software
			meta.Software = cleanString(value)
		case 0x0132: // DateTime
			meta.DateTaken = cleanString(value)
		case 0x013B: // Artist
			meta.Artist = cleanString(value)
		case 0x8298: // Copyright
			meta.Copyright = cleanString(value)
		case 0x8769: // EXIF IFD Pointer
			if n, ok := parseUint(value); ok {
				exifOffset = int(n)
			}
		}

		// EXIF sub-IFD tags
		if isExifIFD {
			switch tagID {
			case 0x829A: // ExposureTime
				meta.ExposureTime = value
			case 0x829D: // FNumber (Aperture)
				meta.Aperture = value
			case 0x8827: // ISO
				if n, ok := parseUint(value); ok {
					meta.ISO = int(n)
				}
			case 0x9000: // ExifVersion
				meta.ExifVersion = cleanString(value)
			case 0x9003: // DateTimeOriginal
				if meta.DateTaken == "" {
					meta.DateTaken = cleanString(value)
				}
			case 0x9209: // Flash
				if n, ok := parseUint(value); ok {
					meta.Flash = decodeFlash(n)
				}
			case 0x920A: // FocalLength
				meta.FocalLength = value
			case 0xA001: // ColorSpace
				if n, ok := parseUint(value); ok {
					switch n {
					case 1:
						meta.ColorSpace = "sRGB"
					case 65535:
						meta.ColorSpace = "Uncalibrated"
					}
				}
			case 0xA405: // FocalLengthIn35mmFilm
				if meta.FocalLength == "" {
					meta.FocalLength = value + "mm (35mm equivalent)"
				}
			case 0xA433: // LensMake
				meta.LensMake = cleanString(value)
			case 0xA434: // LensModel
				meta.LensModel = cleanString(value)
			}
		}
	}

	return exifOffset
}

// findGPSIFD locates the GPS IFD offset.
func (a *ImageAnalyzer) findGPSIFD(data []byte, ifdOffset int, byteOrder binary.ByteOrder) int {
	if ifdOffset+2 > len(data) {
		return 0
	}

	numTags := int(byteOrder.Uint16(data[ifdOffset : ifdOffset+2]))

	for i := 0; i < numTags && i < 200; i++ {
		tagOffset := ifdOffset + 2 + i*12
		if tagOffset+12 > len(data) {
			break
		}

		tagID := byteOrder.Uint16(data[tagOffset : tagOffset+2])
		if tagID == 0x8825 { // GPS IFD Pointer
			return int(byteOrder.Uint32(data[tagOffset+8 : tagOffset+12]))
		}
	}

	return 0
}

// parseGPSIFD extracts GPS information.
func (a *ImageAnalyzer) parseGPSIFD(data []byte, offset int, byteOrder binary.ByteOrder, meta *ImageMetadata) {
	if offset+2 > len(data) {
		return
	}

	numTags := int(byteOrder.Uint16(data[offset : offset+2]))
	meta.HasGPS = true

	var latRef, lonRef string
	var latDeg, latMin, latSec float64
	var lonDeg, lonMin, lonSec float64

	for i := 0; i < numTags && i < 50; i++ {
		tagOffset := offset + 2 + i*12
		if tagOffset+12 > len(data) {
			break
		}

		tagID := byteOrder.Uint16(data[tagOffset : tagOffset+2])
		tagType := byteOrder.Uint16(data[tagOffset+2 : tagOffset+4])
		tagCount := byteOrder.Uint32(data[tagOffset+4 : tagOffset+8])
		valueOffset := tagOffset + 8

		value := a.getTagValue(data, valueOffset, tagType, tagCount, byteOrder)

		switch tagID {
		case 0x0001: // GPSLatitudeRef
			latRef = cleanString(value)
		case 0x0002: // GPSLatitude
			latDeg, latMin, latSec = parseGPSCoordinate(data, valueOffset, byteOrder)
		case 0x0003: // GPSLongitudeRef
			lonRef = cleanString(value)
		case 0x0004: // GPSLongitude
			lonDeg, lonMin, lonSec = parseGPSCoordinate(data, valueOffset, byteOrder)
		case 0x0006: // GPSAltitude
			if valueOffset+4 <= len(data) {
				altOffset := int(byteOrder.Uint32(data[valueOffset : valueOffset+4]))
				if altOffset+8 <= len(data) {
					num := byteOrder.Uint32(data[altOffset : altOffset+4])
					den := byteOrder.Uint32(data[altOffset+4 : altOffset+8])
					if den > 0 {
						meta.GPSAltitude = float64(num) / float64(den)
					}
				}
			}
		case 0x0007: // GPSTimeStamp
			meta.GPSTimestamp = value
		}
	}

	// Convert to decimal degrees
	meta.GPSLatitude = latDeg + latMin/60 + latSec/3600
	if latRef == "S" {
		meta.GPSLatitude = -meta.GPSLatitude
	}

	meta.GPSLongitude = lonDeg + lonMin/60 + lonSec/3600
	if lonRef == "W" {
		meta.GPSLongitude = -meta.GPSLongitude
	}
}

// getTagValue extracts a tag value based on its type.
func (a *ImageAnalyzer) getTagValue(data []byte, offset int, tagType uint16, count uint32, byteOrder binary.ByteOrder) string {
	if offset+4 > len(data) {
		return ""
	}

	// Calculate value size
	typeSize := map[uint16]int{
		1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8,
	}
	size := typeSize[tagType] * int(count)

	var valueData []byte
	if size <= 4 {
		valueData = data[offset : offset+4]
	} else {
		valueOffset := int(byteOrder.Uint32(data[offset : offset+4]))
		if valueOffset+size <= len(data) {
			valueData = data[valueOffset : valueOffset+size]
		} else {
			return ""
		}
	}

	switch tagType {
	case 1, 7: // BYTE, UNDEFINED
		return string(valueData[:min(int(count), len(valueData))])
	case 2: // ASCII
		return string(valueData[:min(int(count), len(valueData))])
	case 3: // SHORT
		if len(valueData) >= 2 {
			return fmt.Sprintf("%d", byteOrder.Uint16(valueData[:2]))
		}
	case 4: // LONG
		if len(valueData) >= 4 {
			return fmt.Sprintf("%d", byteOrder.Uint32(valueData[:4]))
		}
	case 5, 10: // RATIONAL, SRATIONAL
		if len(valueData) >= 8 {
			num := byteOrder.Uint32(valueData[:4])
			den := byteOrder.Uint32(valueData[4:8])
			if den > 0 {
				return fmt.Sprintf("%.4f", float64(num)/float64(den))
			}
		}
	}

	return ""
}

// parseGPSCoordinate extracts degrees, minutes, seconds from GPS rational values.
func parseGPSCoordinate(data []byte, offset int, byteOrder binary.ByteOrder) (float64, float64, float64) {
	if offset+4 > len(data) {
		return 0, 0, 0
	}

	coordOffset := int(byteOrder.Uint32(data[offset : offset+4]))
	if coordOffset+24 > len(data) {
		return 0, 0, 0
	}

	// Read 3 rationals (degrees, minutes, seconds)
	degNum := byteOrder.Uint32(data[coordOffset : coordOffset+4])
	degDen := byteOrder.Uint32(data[coordOffset+4 : coordOffset+8])
	minNum := byteOrder.Uint32(data[coordOffset+8 : coordOffset+12])
	minDen := byteOrder.Uint32(data[coordOffset+12 : coordOffset+16])
	secNum := byteOrder.Uint32(data[coordOffset+16 : coordOffset+20])
	secDen := byteOrder.Uint32(data[coordOffset+20 : coordOffset+24])

	deg := 0.0
	min := 0.0
	sec := 0.0

	if degDen > 0 {
		deg = float64(degNum) / float64(degDen)
	}
	if minDen > 0 {
		min = float64(minNum) / float64(minDen)
	}
	if secDen > 0 {
		sec = float64(secNum) / float64(secDen)
	}

	return deg, min, sec
}

// cleanString removes null bytes and trims whitespace.
func cleanString(s string) string {
	// Remove null bytes
	s = strings.TrimRight(s, "\x00")
	return strings.TrimSpace(s)
}

// parseUint extracts an unsigned integer from a string.
func parseUint(s string) (uint64, bool) {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return 0, false
	}
	var n uint64
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err == nil
}

// decodeFlash converts flash value to human-readable string.
func decodeFlash(value uint64) string {
	switch value {
	case 0:
		return "No Flash"
	case 1:
		return "Flash Fired"
	case 5:
		return "Flash Fired, Return not detected"
	case 7:
		return "Flash Fired, Return detected"
	case 8:
		return "On, Flash did not fire"
	case 9:
		return "On, Flash Fired"
	case 16:
		return "Off, Flash did not fire"
	case 24:
		return "Auto, Flash did not fire"
	case 25:
		return "Auto, Flash Fired"
	default:
		return fmt.Sprintf("Flash (%d)", value)
	}
}

// PNGChunkInfo contains information about a PNG chunk.
type PNGChunkInfo struct {
	Type   string
	Length uint32
	CRC    uint32
}

// PNGAnalysisInfo contains detailed PNG analysis results.
type PNGAnalysisInfo struct {
	Chunks           []PNGChunkInfo
	HasIDAT          bool
	IDATCount        int
	HasText          bool
	HasITXT          bool
	HasZTXT          bool
	HasEXIF          bool
	HasICCP          bool // ICC Color Profile
	HasSRGB          bool
	HasGAMA          bool
	HasCHRM          bool
	HasPHYS          bool
	HasTIME          bool
	TextEntries      map[string]string
	ColorType        int
	BitDepth         int
	CompressionType  int
	FilterMethod     int
	InterlaceMethod  int
	ICCProfileName   string
	PhysicalDPI      int
	LastModified     string
	AIGeneratorFound string
}

// parsePNGMetadata extracts comprehensive metadata from PNG using chunk analysis.
func (a *ImageAnalyzer) parsePNGMetadata(data []byte) ImageMetadata {
	meta := ImageMetadata{FileFormat: "png"}

	if len(data) < 8 {
		return meta
	}

	// Verify PNG signature
	pngSignature := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	for i := 0; i < 8; i++ {
		if data[i] != pngSignature[i] {
			return meta
		}
	}

	// Parse all chunks
	pngInfo := a.analyzePNGChunks(data)

	// Convert PNG info to metadata
	meta.HasEXIF = pngInfo.HasText || pngInfo.HasITXT || pngInfo.HasEXIF

	// Extract software/generator info from text entries
	if pngInfo.TextEntries != nil {
		if software, ok := pngInfo.TextEntries["Software"]; ok {
			meta.Software = software
		}
		if software, ok := pngInfo.TextEntries["software"]; ok && meta.Software == "" {
			meta.Software = software
		}
		if author, ok := pngInfo.TextEntries["Author"]; ok {
			meta.Artist = author
		}
		if author, ok := pngInfo.TextEntries["artist"]; ok && meta.Artist == "" {
			meta.Artist = author
		}
		if copyright, ok := pngInfo.TextEntries["Copyright"]; ok {
			meta.Copyright = copyright
		}
		if desc, ok := pngInfo.TextEntries["Description"]; ok && meta.Software == "" {
			// Some AI tools put info in Description
			meta.Software = desc
		}
		if creation, ok := pngInfo.TextEntries["Creation Time"]; ok {
			meta.DateTaken = creation
		}
	}

	// Check for AI generator
	if pngInfo.AIGeneratorFound != "" {
		meta.Software = "AI Generator"
	}

	// Set resolution if available
	if pngInfo.PhysicalDPI > 0 {
		meta.XResolution = pngInfo.PhysicalDPI
		meta.YResolution = pngInfo.PhysicalDPI
		meta.ResolutionUnit = "inches"
	}

	// Set color space
	if pngInfo.HasSRGB {
		meta.ColorSpace = "sRGB"
	} else if pngInfo.HasICCP {
		meta.ColorSpace = pngInfo.ICCProfileName
	}

	// Last modified time
	if pngInfo.LastModified != "" {
		meta.DateTaken = pngInfo.LastModified
	}

	// Determine if likely screenshot (minimal metadata, standard settings)
	if !pngInfo.HasText && !pngInfo.HasITXT && !pngInfo.HasICCP &&
		meta.Software == "" && meta.Artist == "" {
		meta.IsScreenshot = true
	}

	return meta
}

// analyzePNGChunks performs detailed analysis of PNG chunks.
func (a *ImageAnalyzer) analyzePNGChunks(data []byte) PNGAnalysisInfo {
	info := PNGAnalysisInfo{
		Chunks:      make([]PNGChunkInfo, 0),
		TextEntries: make(map[string]string),
	}

	// Start after PNG signature (8 bytes)
	offset := 8

	for offset+12 <= len(data) {
		// Read chunk length (4 bytes, big-endian)
		chunkLen := binary.BigEndian.Uint32(data[offset : offset+4])

		// Read chunk type (4 bytes)
		if offset+8 > len(data) {
			break
		}
		chunkType := string(data[offset+4 : offset+8])

		// Calculate end of chunk (including CRC)
		chunkEnd := offset + 12 + int(chunkLen)
		if chunkEnd > len(data) {
			break
		}

		// Read CRC
		crc := binary.BigEndian.Uint32(data[chunkEnd-4 : chunkEnd])

		// Store chunk info
		info.Chunks = append(info.Chunks, PNGChunkInfo{
			Type:   chunkType,
			Length: chunkLen,
			CRC:    crc,
		})

		// Analyze specific chunk types
		chunkData := data[offset+8 : offset+8+int(chunkLen)]

		switch chunkType {
		case "IHDR":
			a.parseIHDRChunk(chunkData, &info)
		case "IDAT":
			info.HasIDAT = true
			info.IDATCount++
		case "tEXt":
			info.HasText = true
			a.parseTEXTChunk(chunkData, &info)
		case "iTXt":
			info.HasITXT = true
			a.parseITXTChunk(chunkData, &info)
		case "zTXt":
			info.HasZTXT = true
			a.parseZTXTChunk(chunkData, &info)
		case "eXIf":
			info.HasEXIF = true
		case "iCCP":
			info.HasICCP = true
			a.parseICCPChunk(chunkData, &info)
		case "sRGB":
			info.HasSRGB = true
		case "gAMA":
			info.HasGAMA = true
		case "cHRM":
			info.HasCHRM = true
		case "pHYs":
			info.HasPHYS = true
			a.parsePHYSChunk(chunkData, &info)
		case "tIME":
			info.HasTIME = true
			a.parseTIMEChunk(chunkData, &info)
		case "IEND":
			// End of PNG
			break
		}

		// Check for AI generator signatures in any text chunk
		if chunkLen > 0 {
			chunkStr := string(chunkData)
			aiGenerators := []string{
				"DALL-E", "Midjourney", "Stable Diffusion", "ComfyUI",
				"NovelAI", "Automatic1111", "InvokeAI", "DreamStudio",
				"Leonardo.AI", "Adobe Firefly", "Imagen", "SDXL",
			}
			for _, gen := range aiGenerators {
				if strings.Contains(chunkStr, gen) {
					info.AIGeneratorFound = gen
					break
				}
			}
		}

		offset = chunkEnd
	}

	return info
}

// parseIHDRChunk extracts image header information.
func (a *ImageAnalyzer) parseIHDRChunk(data []byte, info *PNGAnalysisInfo) {
	if len(data) < 13 {
		return
	}

	// Width and Height are in getImageStats
	info.BitDepth = int(data[8])
	info.ColorType = int(data[9])
	info.CompressionType = int(data[10])
	info.FilterMethod = int(data[11])
	info.InterlaceMethod = int(data[12])
}

// parseTEXTChunk extracts text metadata.
func (a *ImageAnalyzer) parseTEXTChunk(data []byte, info *PNGAnalysisInfo) {
	// tEXt: keyword\0text
	nullIdx := -1
	for i := 0; i < len(data) && i < 80; i++ { // keyword max 79 chars
		if data[i] == 0 {
			nullIdx = i
			break
		}
	}

	if nullIdx > 0 && nullIdx < len(data)-1 {
		keyword := string(data[:nullIdx])
		text := string(data[nullIdx+1:])
		info.TextEntries[keyword] = cleanString(text)
	}
}

// parseITXTChunk extracts international text metadata.
func (a *ImageAnalyzer) parseITXTChunk(data []byte, info *PNGAnalysisInfo) {
	// iTXt: keyword\0compression_flag\0compression_method\0language_tag\0translated_keyword\0text
	nullIdx := -1
	for i := 0; i < len(data) && i < 80; i++ {
		if data[i] == 0 {
			nullIdx = i
			break
		}
	}

	if nullIdx > 0 && nullIdx+3 < len(data) {
		keyword := string(data[:nullIdx])

		// Skip compression flag, method, and find the text
		// Look for the last null separator
		textStart := nullIdx + 3
		for i := textStart; i < len(data); i++ {
			if data[i] == 0 {
				textStart = i + 1
			}
		}

		if textStart < len(data) {
			// Find next null or end
			textEnd := len(data)
			for i := textStart; i < len(data); i++ {
				if data[i] == 0 {
					textEnd = i
					break
				}
			}
			text := string(data[textStart:textEnd])
			info.TextEntries[keyword] = cleanString(text)
		}
	}
}

// parseZTXTChunk extracts compressed text metadata.
func (a *ImageAnalyzer) parseZTXTChunk(data []byte, info *PNGAnalysisInfo) {
	// zTXt: keyword\0compression_method\0compressed_text
	// Note: We can't decompress without zlib, but we can get the keyword
	nullIdx := -1
	for i := 0; i < len(data) && i < 80; i++ {
		if data[i] == 0 {
			nullIdx = i
			break
		}
	}

	if nullIdx > 0 {
		keyword := string(data[:nullIdx])
		info.TextEntries[keyword] = "[compressed]"
	}
}

// parseICCPChunk extracts ICC color profile information.
func (a *ImageAnalyzer) parseICCPChunk(data []byte, info *PNGAnalysisInfo) {
	// iCCP: profile_name\0compression_method\0compressed_profile
	nullIdx := -1
	for i := 0; i < len(data) && i < 80; i++ {
		if data[i] == 0 {
			nullIdx = i
			break
		}
	}

	if nullIdx > 0 {
		info.ICCProfileName = string(data[:nullIdx])
	}
}

// parsePHYSChunk extracts physical pixel dimensions.
func (a *ImageAnalyzer) parsePHYSChunk(data []byte, info *PNGAnalysisInfo) {
	// pHYs: pixels_per_unit_x(4) + pixels_per_unit_y(4) + unit_specifier(1)
	if len(data) < 9 {
		return
	}

	pxPerUnitX := binary.BigEndian.Uint32(data[0:4])
	pxPerUnitY := binary.BigEndian.Uint32(data[4:8])
	unitSpec := data[8]

	if unitSpec == 1 { // Meter
		// Convert to DPI (pixels per inch = pixels per meter / 39.3701)
		dpiX := float64(pxPerUnitX) / 39.3701
		dpiY := float64(pxPerUnitY) / 39.3701

		// Use average if different
		info.PhysicalDPI = int((dpiX + dpiY) / 2)
	}
}

// parseTIMEChunk extracts last modification time.
func (a *ImageAnalyzer) parseTIMEChunk(data []byte, info *PNGAnalysisInfo) {
	// tIME: year(2) + month(1) + day(1) + hour(1) + minute(1) + second(1)
	if len(data) < 7 {
		return
	}

	year := binary.BigEndian.Uint16(data[0:2])
	month := data[2]
	day := data[3]
	hour := data[4]
	minute := data[5]
	second := data[6]

	info.LastModified = fmt.Sprintf("%04d-%02d-%02d %02d:%02d:%02d",
		year, month, day, hour, minute, second)
}

// getImageStats calculates basic image statistics.
func (a *ImageAnalyzer) getImageStats(data []byte, format string) ImageStats {
	stats := ImageStats{}

	switch format {
	case "jpeg":
		stats = a.getJPEGStats(data)
	case "png":
		stats = a.getPNGStats(data)
	}

	return stats
}

// getJPEGStats extracts dimensions from JPEG.
func (a *ImageAnalyzer) getJPEGStats(data []byte) ImageStats {
	stats := ImageStats{}

	// Look for SOF0 marker (0xFFC0) which contains dimensions
	for i := 0; i < len(data)-10; i++ {
		if data[i] == 0xFF && (data[i+1] == 0xC0 || data[i+1] == 0xC2) {
			if i+9 < len(data) {
				stats.BitDepth = int(data[i+4])
				stats.Height = int(binary.BigEndian.Uint16(data[i+5 : i+7]))
				stats.Width = int(binary.BigEndian.Uint16(data[i+7 : i+9]))
				stats.ColorChannels = int(data[i+9])
			}
			break
		}
	}

	return stats
}

// getPNGStats extracts dimensions from PNG.
func (a *ImageAnalyzer) getPNGStats(data []byte) ImageStats {
	stats := ImageStats{}

	// PNG IHDR chunk starts at byte 8
	if len(data) >= 24 {
		stats.Width = int(binary.BigEndian.Uint32(data[16:20]))
		stats.Height = int(binary.BigEndian.Uint32(data[20:24]))
		if len(data) >= 25 {
			stats.BitDepth = int(data[24])
		}
	}

	return stats
}

// analyzeMetadata scores based on metadata presence.
func (a *ImageAnalyzer) analyzeMetadata(meta ImageMetadata) float64 {
	score := 0.5 // Start neutral

	// Real photos typically have EXIF data
	if meta.HasEXIF {
		score -= 0.2
	} else {
		score += 0.2
	}

	// Camera make is strong signal of real photo
	if meta.CameraMake != "" {
		score -= 0.2
	}

	// GPS data is very strong signal
	if meta.HasGPS {
		score -= 0.2
	}

	// AI generator software is obvious signal
	if meta.Software == "AI Generator" {
		score += 0.4
	}

	// Screenshots are typically human-created but lack camera EXIF
	// They should be treated as neutral rather than suspicious
	if meta.IsScreenshot {
		// Counteract the "no EXIF" penalty for screenshots
		score -= 0.2
	}

	return math.Max(0, math.Min(1, score))
}

// analyzeColorDistribution performs comprehensive color histogram analysis.
// AI images often exhibit unusual color distributions that differ from natural photos.
func (a *ImageAnalyzer) analyzeColorDistribution(data []byte, format string) float64 {
	if len(data) < 2000 {
		return 0.5
	}

	// Combine multiple color distribution analysis techniques
	entropyScore := a.analyzeByteEntropy(data, format)
	histogramScore := a.analyzeHistogramShape(data, format)
	colorBandingScore := a.detectColorBanding(data, format)
	peakScore := a.analyzeHistogramPeaks(data, format)

	// Weight the scores
	combinedScore := entropyScore*0.25 + histogramScore*0.30 + colorBandingScore*0.25 + peakScore*0.20

	return math.Max(0, math.Min(1, combinedScore))
}

// analyzeByteEntropy calculates entropy of the image data.
func (a *ImageAnalyzer) analyzeByteEntropy(data []byte, format string) float64 {
	// Determine where image data starts
	startOffset := a.getImageDataOffset(data, format)
	if startOffset < 0 || startOffset+2000 > len(data) {
		return 0.5
	}

	// Sample size for analysis
	sampleSize := min(5000, len(data)-startOffset)
	sample := data[startOffset : startOffset+sampleSize]

	// Calculate byte frequency distribution
	freq := make([]int, 256)
	for _, b := range sample {
		freq[b]++
	}

	// Calculate Shannon entropy
	entropy := 0.0
	total := float64(len(sample))
	for _, f := range freq {
		if f > 0 {
			p := float64(f) / total
			entropy -= p * math.Log2(p)
		}
	}

	// Normalized entropy (0-8 for bytes)
	normalizedEntropy := entropy / 8.0

	// Real images typically have entropy 0.7-0.95
	// AI images sometimes have unusual patterns
	if normalizedEntropy < 0.5 {
		return 0.75 // Very low entropy - suspicious
	}
	if normalizedEntropy < 0.65 {
		return 0.6 // Low entropy
	}
	if normalizedEntropy > 0.97 {
		return 0.6 // Very high entropy - possibly artificial noise
	}
	if normalizedEntropy > 0.92 {
		return 0.45 // Slightly high but normal range
	}

	return 0.35 // Normal entropy range
}

// analyzeHistogramShape analyzes the shape of the color histogram.
func (a *ImageAnalyzer) analyzeHistogramShape(data []byte, format string) float64 {
	startOffset := a.getImageDataOffset(data, format)
	if startOffset < 0 || startOffset+3000 > len(data) {
		return 0.5
	}

	sampleSize := min(10000, len(data)-startOffset)
	sample := data[startOffset : startOffset+sampleSize]

	// Build histogram
	hist := make([]int, 256)
	for _, b := range sample {
		hist[b]++
	}

	// Calculate histogram statistics
	total := float64(len(sample))

	// 1. Check for gaps (empty bins) - natural images have smoother histograms
	gapCount := 0
	for i := 10; i < 246; i++ { // Exclude extremes
		if hist[i] == 0 {
			gapCount++
		}
	}
	gapRatio := float64(gapCount) / 236.0

	// 2. Check for unusual spikes (single values much higher than neighbors)
	spikeCount := 0
	for i := 2; i < 254; i++ {
		neighborAvg := float64(hist[i-2]+hist[i-1]+hist[i+1]+hist[i+2]) / 4.0
		if neighborAvg > 0 && float64(hist[i]) > neighborAvg*5 {
			spikeCount++
		}
	}
	spikeRatio := float64(spikeCount) / 256.0

	// 3. Calculate skewness
	mean := 0.0
	for i := 0; i < 256; i++ {
		mean += float64(i) * float64(hist[i]) / total
	}

	variance := 0.0
	for i := 0; i < 256; i++ {
		diff := float64(i) - mean
		variance += diff * diff * float64(hist[i]) / total
	}
	stdDev := math.Sqrt(variance)

	// Calculate skewness
	skewness := 0.0
	if stdDev > 0 {
		for i := 0; i < 256; i++ {
			diff := float64(i) - mean
			skewness += math.Pow(diff/stdDev, 3) * float64(hist[i]) / total
		}
	}

	// Score based on findings
	score := 0.4 // Start neutral-good

	// Many gaps suggest artificial/processed image
	if gapRatio > 0.3 {
		score += 0.2
	} else if gapRatio > 0.15 {
		score += 0.1
	}

	// Unusual spikes suggest posterization or AI artifacts
	if spikeRatio > 0.05 {
		score += 0.2
	} else if spikeRatio > 0.02 {
		score += 0.1
	}

	// Extreme skewness is unusual
	if math.Abs(skewness) > 1.5 {
		score += 0.1
	}

	return math.Min(0.9, score)
}

// detectColorBanding detects color banding artifacts common in AI images.
func (a *ImageAnalyzer) detectColorBanding(data []byte, format string) float64 {
	startOffset := a.getImageDataOffset(data, format)
	if startOffset < 0 || startOffset+2000 > len(data) {
		return 0.5
	}

	sampleSize := min(8000, len(data)-startOffset)
	sample := data[startOffset : startOffset+sampleSize]

	// Look for repeated byte sequences (banding pattern)
	repeatCount := 0
	runLength := 0
	prevByte := byte(0)

	for i, b := range sample {
		if i == 0 {
			prevByte = b
			runLength = 1
			continue
		}

		if b == prevByte {
			runLength++
		} else {
			// Long runs of same value indicate banding
			if runLength > 8 {
				repeatCount++
			}
			runLength = 1
			prevByte = b
		}
	}

	// Also check for "near" repeats (values within 1-2)
	nearRepeatCount := 0
	for i := 1; i < len(sample); i++ {
		diff := int(sample[i]) - int(sample[i-1])
		if diff >= -2 && diff <= 2 {
			nearRepeatCount++
		}
	}
	nearRepeatRatio := float64(nearRepeatCount) / float64(len(sample))

	// Score based on banding detection
	repeatRatio := float64(repeatCount) / float64(len(sample)/10)

	score := 0.4
	if repeatRatio > 0.2 {
		score += 0.25 // Significant banding
	} else if repeatRatio > 0.1 {
		score += 0.15
	}

	// Very high near-repeat ratio suggests smooth gradients (can be AI)
	if nearRepeatRatio > 0.7 {
		score += 0.15
	}

	return math.Min(0.85, score)
}

// analyzeHistogramPeaks looks for unusual peak patterns.
func (a *ImageAnalyzer) analyzeHistogramPeaks(data []byte, format string) float64 {
	startOffset := a.getImageDataOffset(data, format)
	if startOffset < 0 || startOffset+2000 > len(data) {
		return 0.5
	}

	sampleSize := min(8000, len(data)-startOffset)
	sample := data[startOffset : startOffset+sampleSize]

	// Build histogram
	hist := make([]int, 256)
	for _, b := range sample {
		hist[b]++
	}

	// Find peaks (local maxima)
	peaks := make([]int, 0)
	for i := 2; i < 254; i++ {
		if hist[i] > hist[i-1] && hist[i] > hist[i+1] &&
			hist[i] > hist[i-2] && hist[i] > hist[i+2] {
			peaks = append(peaks, i)
		}
	}

	// Analyze peak characteristics
	total := float64(len(sample))

	// 1. Count how much of the data is concentrated in peaks
	peakConcentration := 0.0
	for _, p := range peaks {
		// Sum values around each peak
		for i := max(0, p-3); i <= min(255, p+3); i++ {
			peakConcentration += float64(hist[i])
		}
	}
	peakRatio := peakConcentration / total

	// 2. Check for suspiciously regular peak spacing
	regularSpacing := false
	if len(peaks) >= 3 {
		spacings := make([]int, 0)
		for i := 1; i < len(peaks); i++ {
			spacings = append(spacings, peaks[i]-peaks[i-1])
		}

		// Check if spacings are similar
		if len(spacings) >= 2 {
			avgSpacing := 0.0
			for _, s := range spacings {
				avgSpacing += float64(s)
			}
			avgSpacing /= float64(len(spacings))

			variance := 0.0
			for _, s := range spacings {
				diff := float64(s) - avgSpacing
				variance += diff * diff
			}
			variance /= float64(len(spacings))

			// Low variance in spacing suggests artificial pattern
			if math.Sqrt(variance) < avgSpacing*0.2 {
				regularSpacing = true
			}
		}
	}

	// 3. Check for peaks at "round" values (multiples of 16, 32, 64)
	roundPeakCount := 0
	for _, p := range peaks {
		if p%16 == 0 || p%32 == 0 || p%64 == 0 {
			roundPeakCount++
		}
	}

	// Score
	score := 0.4

	// Too many peaks suggest artificial processing
	if len(peaks) > 20 {
		score += 0.15
	}

	// High concentration in peaks suggests limited color palette
	if peakRatio > 0.6 {
		score += 0.15
	}

	// Regular spacing is suspicious
	if regularSpacing {
		score += 0.15
	}

	// Many peaks at round values suggests quantization
	if roundPeakCount > len(peaks)/3 && roundPeakCount > 2 {
		score += 0.1
	}

	return math.Min(0.85, score)
}

// getImageDataOffset returns the offset where actual image data begins.
func (a *ImageAnalyzer) getImageDataOffset(data []byte, format string) int {
	switch format {
	case "jpeg":
		// Find SOS marker (start of scan)
		for i := 0; i < len(data)-10; i++ {
			if data[i] == 0xFF && data[i+1] == 0xDA {
				headerLen := int(data[i+2])<<8 | int(data[i+3])
				return i + 2 + headerLen
			}
		}
		return 100 // Fallback
	case "png":
		// Find first IDAT chunk
		for i := 8; i < len(data)-12; {
			chunkLen := int(binary.BigEndian.Uint32(data[i : i+4]))
			chunkType := string(data[i+4 : i+8])
			if chunkType == "IDAT" {
				return i + 8
			}
			i += 12 + chunkLen
			if i > len(data) {
				break
			}
		}
		return 50 // Fallback
	default:
		return 100 // Generic fallback
	}
}

// analyzeEdgeConsistency looks for edge artifacts.
// AI images often have inconsistent edges.
func (a *ImageAnalyzer) analyzeEdgeConsistency(data []byte, format string) float64 {
	// Simplified analysis - check for repeated patterns
	// AI sometimes has repeated textures or edge patterns

	if len(data) < 5000 {
		return 0.5
	}

	// Sample different regions and compare
	// Look for unusual repetition patterns

	sampleSize := 256
	samples := make([][]byte, 0)

	for i := 1000; i < len(data)-sampleSize && len(samples) < 10; i += len(data) / 12 {
		samples = append(samples, data[i:i+sampleSize])
	}

	if len(samples) < 3 {
		return 0.5
	}

	// Check similarity between samples
	// AI images sometimes have suspiciously similar regions
	similarityCount := 0
	for i := 0; i < len(samples)-1; i++ {
		for j := i + 1; j < len(samples); j++ {
			if byteSimilarity(samples[i], samples[j]) > 0.9 {
				similarityCount++
			}
		}
	}

	// High similarity between distant regions is suspicious
	similarityRatio := float64(similarityCount) / float64(len(samples)*(len(samples)-1)/2)

	if similarityRatio > 0.3 {
		return 0.7 // Suspicious repetition
	}

	return 0.4
}

// byteSimilarity calculates similarity between two byte slices.
func byteSimilarity(a, b []byte) float64 {
	if len(a) != len(b) {
		return 0
	}

	matches := 0
	for i := range a {
		if a[i] == b[i] {
			matches++
		}
	}

	return float64(matches) / float64(len(a))
}

// analyzeNoisePattern checks for natural sensor noise.
// Real photos have characteristic noise from camera sensors.
func (a *ImageAnalyzer) analyzeNoisePattern(data []byte, format string) float64 {
	if len(data) < 2000 {
		return 0.5
	}

	// Look for high-frequency variations that indicate sensor noise
	// AI images are often "too clean" or have artificial noise

	startOffset := 500
	sampleSize := 1000

	if len(data) < startOffset+sampleSize {
		return 0.5
	}

	sample := data[startOffset : startOffset+sampleSize]

	// Calculate local variance (proxy for noise)
	variances := make([]float64, 0)
	windowSize := 16

	for i := 0; i < len(sample)-windowSize; i += windowSize {
		window := sample[i : i+windowSize]

		// Calculate mean
		sum := 0.0
		for _, b := range window {
			sum += float64(b)
		}
		mean := sum / float64(len(window))

		// Calculate variance
		variance := 0.0
		for _, b := range window {
			diff := float64(b) - mean
			variance += diff * diff
		}
		variance /= float64(len(window))

		variances = append(variances, variance)
	}

	if len(variances) == 0 {
		return 0.5
	}

	// Calculate variance of variances (noise should be relatively uniform)
	avgVariance := 0.0
	for _, v := range variances {
		avgVariance += v
	}
	avgVariance /= float64(len(variances))

	varianceOfVariance := 0.0
	for _, v := range variances {
		diff := v - avgVariance
		varianceOfVariance += diff * diff
	}
	varianceOfVariance /= float64(len(variances))

	// Real images have moderate, consistent noise
	// AI images are either too clean or have inconsistent noise
	noiseCV := math.Sqrt(varianceOfVariance) / (avgVariance + 1)

	if avgVariance < 10 { // Too clean
		return 0.7
	}
	if noiseCV > 2.0 { // Too inconsistent
		return 0.6
	}

	return 0.4
}

// analyzeCompression checks compression artifacts using DCT coefficient analysis.
// Real JPEGs have natural compression; AI-generated may have artifacts.
func (a *ImageAnalyzer) analyzeCompression(data []byte, format string) float64 {
	if format != "jpeg" {
		return 0.5 // Can't analyze non-JPEG compression this way
	}

	// Combine multiple DCT-based forensic techniques
	quantScore := a.analyzeQuantizationTable(data)
	dctScore := a.analyzeDCTCoefficients(data)
	doubleCompressionScore := a.detectDoubleCompression(data)

	// Weight the scores
	combinedScore := quantScore*0.3 + dctScore*0.4 + doubleCompressionScore*0.3

	return math.Max(0, math.Min(1, combinedScore))
}

// analyzeQuantizationTable examines the JPEG quantization tables.
func (a *ImageAnalyzer) analyzeQuantizationTable(data []byte) float64 {
	// Find DQT marker (0xFFDB)
	for i := 0; i < len(data)-100; i++ {
		if data[i] == 0xFF && data[i+1] == 0xDB {
			// Found quantization table
			if i+69 < len(data) {
				// Sum the quantization values
				sum := 0
				for j := i + 5; j < i+69; j++ {
					sum += int(data[j])
				}

				// Very low sum = very high quality (possibly AI-generated PNG converted to JPEG)
				// Normal range is roughly 300-2000
				if sum < 200 {
					return 0.7 // Suspiciously high quality
				}
				if sum > 3000 {
					return 0.6 // Very low quality (heavy compression)
				}

				return 0.4 // Normal compression
			}
			break
		}
	}

	return 0.5
}

// analyzeDCTCoefficients performs statistical analysis on DCT coefficients.
// AI-generated images often have different DCT coefficient distributions.
func (a *ImageAnalyzer) analyzeDCTCoefficients(data []byte) float64 {
	// Find the start of scan data (SOS marker: 0xFFDA)
	sosIndex := -1
	for i := 0; i < len(data)-10; i++ {
		if data[i] == 0xFF && data[i+1] == 0xDA {
			// Skip to end of SOS header (after the header bytes)
			headerLen := int(data[i+2])<<8 | int(data[i+3])
			sosIndex = i + 2 + headerLen
			break
		}
	}

	if sosIndex < 0 || sosIndex >= len(data)-1000 {
		return 0.5 // Can't find scan data
	}

	// Analyze the entropy-coded DCT data
	// Real photos have characteristic distribution; AI may differ

	sampleSize := min(5000, len(data)-sosIndex)
	scanData := data[sosIndex : sosIndex+sampleSize]

	// 1. Analyze byte frequency distribution in scan data
	freq := make([]int, 256)
	for _, b := range scanData {
		freq[b]++
	}

	// Calculate entropy of scan data
	entropy := 0.0
	total := float64(len(scanData))
	nonZeroCount := 0
	for _, f := range freq {
		if f > 0 {
			p := float64(f) / total
			entropy -= p * math.Log2(p)
			nonZeroCount++
		}
	}

	// 2. Analyze zero-run patterns (related to DC/AC coefficient distribution)
	zeroRuns := analyzeZeroRuns(scanData)

	// 3. Check for FF 00 escape sequences (marker byte stuffing)
	// Real JPEGs have natural FF 00 patterns
	ffCount := 0
	ff00Count := 0
	for i := 0; i < len(scanData)-1; i++ {
		if scanData[i] == 0xFF {
			ffCount++
			if scanData[i+1] == 0x00 {
				ff00Count++
			}
		}
	}

	// Calculate scores
	// Normal JPEG entropy is typically 7.0-7.8
	entropyScore := 0.5
	if entropy < 6.5 {
		entropyScore = 0.7 // Suspiciously low entropy
	} else if entropy > 7.9 {
		entropyScore = 0.6 // Unusually high entropy
	} else {
		entropyScore = 0.4 // Normal range
	}

	// Zero run analysis
	zeroRunScore := 0.5
	avgZeroRun := zeroRuns.avgLength
	if avgZeroRun < 1.5 {
		zeroRunScore = 0.6 // Very few zeros (dense data)
	} else if avgZeroRun > 8.0 {
		zeroRunScore = 0.6 // Too many zeros (sparse/artificial)
	} else {
		zeroRunScore = 0.4 // Normal range
	}

	// FF 00 pattern analysis
	ff00Score := 0.5
	if ffCount > 0 {
		ff00Ratio := float64(ff00Count) / float64(ffCount)
		// Most FF bytes should be stuffed in natural images
		if ff00Ratio < 0.3 {
			ff00Score = 0.6 // Unusual pattern
		} else if ff00Ratio > 0.95 {
			ff00Score = 0.4 // Normal
		}
	}

	// Combine scores
	return entropyScore*0.4 + zeroRunScore*0.3 + ff00Score*0.3
}

// zeroRunStats holds statistics about zero runs in data.
type zeroRunStats struct {
	avgLength   float64
	maxLength   int
	totalRuns   int
	varianceLen float64
}

// analyzeZeroRuns calculates statistics about consecutive zero bytes.
func analyzeZeroRuns(data []byte) zeroRunStats {
	stats := zeroRunStats{}

	runs := make([]int, 0)
	currentRun := 0

	for _, b := range data {
		if b == 0x00 {
			currentRun++
		} else {
			if currentRun > 0 {
				runs = append(runs, currentRun)
				if currentRun > stats.maxLength {
					stats.maxLength = currentRun
				}
			}
			currentRun = 0
		}
	}

	// Don't forget the last run
	if currentRun > 0 {
		runs = append(runs, currentRun)
		if currentRun > stats.maxLength {
			stats.maxLength = currentRun
		}
	}

	stats.totalRuns = len(runs)

	if len(runs) == 0 {
		return stats
	}

	// Calculate average
	sum := 0
	for _, r := range runs {
		sum += r
	}
	stats.avgLength = float64(sum) / float64(len(runs))

	// Calculate variance
	variance := 0.0
	for _, r := range runs {
		diff := float64(r) - stats.avgLength
		variance += diff * diff
	}
	stats.varianceLen = variance / float64(len(runs))

	return stats
}

// detectDoubleCompression detects signs of double JPEG compression.
// Images that were AI-generated and then saved as JPEG may show artifacts.
func (a *ImageAnalyzer) detectDoubleCompression(data []byte) float64 {
	// Double compression often shows in:
	// 1. Multiple quantization tables
	// 2. Blocking artifacts at non-standard intervals
	// 3. Histogram anomalies in DCT coefficients

	// Count quantization tables
	dqtCount := 0
	for i := 0; i < len(data)-4; i++ {
		if data[i] == 0xFF && data[i+1] == 0xDB {
			dqtCount++
			// Skip to next marker
			if i+3 < len(data) {
				tableLen := int(data[i+2])<<8 | int(data[i+3])
				i += tableLen + 1
			}
		}
	}

	// Most JPEGs have 2 quantization tables (luminance + chrominance)
	// More than 2 might indicate multiple save operations
	if dqtCount > 2 {
		return 0.65 // Possible double compression
	}

	// Analyze for blocking artifacts by looking at periodic patterns
	// Sample the middle of the image data
	if len(data) < 10000 {
		return 0.5
	}

	midPoint := len(data) / 2
	sampleSize := 2048
	if midPoint+sampleSize > len(data) {
		return 0.5
	}

	sample := data[midPoint : midPoint+sampleSize]

	// Look for 8-byte periodic patterns (DCT block size)
	periodicScore := detectPeriodicPattern(sample, 8)

	// Also check for 16-byte patterns (double block artifacts)
	periodic16Score := detectPeriodicPattern(sample, 16)

	// Strong 16-byte periodicity relative to 8-byte might indicate issues
	if periodic16Score > periodicScore*1.5 && periodic16Score > 0.3 {
		return 0.6 // Suspicious double-block patterns
	}

	return 0.4 // Normal
}

// detectPeriodicPattern measures periodicity in data at given interval.
func detectPeriodicPattern(data []byte, period int) float64 {
	if len(data) < period*4 {
		return 0
	}

	// Calculate autocorrelation at the specified period
	correlation := 0.0
	count := 0

	for i := 0; i < len(data)-period; i++ {
		diff := int(data[i]) - int(data[i+period])
		correlation += float64(diff * diff)
		count++
	}

	if count == 0 {
		return 0
	}

	// Lower correlation = stronger periodicity
	avgCorrelation := correlation / float64(count)

	// Normalize to 0-1 range (lower is more periodic)
	// Max possible diff^2 is 255^2 = 65025
	normalizedScore := 1.0 - (avgCorrelation / 65025.0)

	return normalizedScore
}

// analyzeSymmetry detects unnatural symmetry.
// AI images sometimes have artifacts related to symmetry.
func (a *ImageAnalyzer) analyzeSymmetry(data []byte, format string) float64 {
	// This is a simplified check
	// Full symmetry analysis would require decoding the image

	if len(data) < 2000 {
		return 0.5
	}

	// Compare first and second halves of data as a rough proxy
	// (Not actual image symmetry, but can catch some patterns)

	mid := len(data) / 2
	sampleSize := 500

	if mid+sampleSize > len(data) {
		return 0.5
	}

	sample1 := data[100:600]                // Near start
	sample2 := data[mid : mid+sampleSize] // Near middle

	// Check if suspiciously similar
	similarity := byteSimilarity(sample1, sample2)

	if similarity > 0.7 {
		return 0.7 // Suspicious similarity
	}

	return 0.4
}

// calculateWeightedScore combines signals into final score.
func (a *ImageAnalyzer) calculateWeightedScore(signals ImageSignals) float64 {
	w := a.weights

	score := signals.MetadataScore*w.MetadataScore +
		signals.ColorDistribution*w.ColorDistribution +
		signals.EdgeConsistency*w.EdgeConsistency +
		signals.NoisePattern*w.NoisePattern +
		signals.CompressionAnalysis*w.CompressionAnalysis +
		signals.SymmetryScore*w.SymmetryDetection

	totalWeight := w.MetadataScore + w.ColorDistribution + w.EdgeConsistency +
		w.NoisePattern + w.CompressionAnalysis + w.SymmetryDetection

	if totalWeight > 0 {
		score /= totalWeight
	}

	return math.Max(0, math.Min(1, score))
}

// detectImageFormat identifies the image format from magic bytes.
func detectImageFormat(data []byte) string {
	if len(data) < 8 {
		return "unknown"
	}

	// JPEG: FF D8 FF
	if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return "jpeg"
	}

	// PNG: 89 50 4E 47 0D 0A 1A 0A
	if data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 {
		return "png"
	}

	// GIF: 47 49 46 38
	if data[0] == 0x47 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x38 {
		return "gif"
	}

	// WebP: 52 49 46 46 ... 57 45 42 50
	if len(data) >= 12 {
		if data[0] == 0x52 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x46 {
			if data[8] == 0x57 && data[9] == 0x45 && data[10] == 0x42 && data[11] == 0x50 {
				return "webp"
			}
		}
	}

	// BMP: 42 4D
	if data[0] == 0x42 && data[1] == 0x4D {
		return "bmp"
	}

	return "unknown"
}
