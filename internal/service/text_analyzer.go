package service

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

// =============================================================================
// Manuscript Text Detection Algorithm
// =============================================================================
//
// This is our own detection algorithm based on statistical analysis of text.
// No external APIs required.
//
// Key insight: AI-generated text is statistically "average" - it optimizes for
// the most likely next token, resulting in predictable patterns.
//
// Human text is messy, variable, and personal.
//
// We measure several signals:
//   1. Sentence length variance (humans vary more)
//   2. Vocabulary richness (humans use rare words, slang)
//   3. Burstiness (humans cluster related words)
//   4. Punctuation patterns (humans use more variety)
//   5. AI phrase detection (common AI patterns)
//   6. Perplexity proxy (word predictability)
//
// =============================================================================

// TextAnalyzer performs statistical analysis on text to detect AI generation.
type TextAnalyzer struct {
	// Weights for each signal (tuned based on testing)
	weights TextAnalyzerWeights
}

// TextAnalyzerWeights controls the importance of each signal.
type TextAnalyzerWeights struct {
	SentenceVariance    float64
	VocabularyRichness  float64
	Burstiness          float64
	PunctuationVariety  float64
	AIPhraseDetection   float64
	WordLengthVariance  float64
	ContractionsUsage   float64
	RepetitionPenalty   float64
	ReadabilityVariance float64
	NGramUniformity     float64
}

// DefaultWeights returns tuned weights for the analyzer.
func DefaultWeights() TextAnalyzerWeights {
	return TextAnalyzerWeights{
		SentenceVariance:    0.10,
		VocabularyRichness:  0.16,
		Burstiness:          0.07,
		PunctuationVariety:  0.07,
		AIPhraseDetection:   0.20,
		WordLengthVariance:  0.05,
		ContractionsUsage:   0.09,
		RepetitionPenalty:   0.08,
		ReadabilityVariance: 0.09,
		NGramUniformity:     0.09, // NEW: Uniform n-gram distribution = AI-like
	}
}

// NewTextAnalyzer creates a new analyzer with default weights.
func NewTextAnalyzer() *TextAnalyzer {
	return &TextAnalyzer{
		weights: DefaultWeights(),
	}
}

// NewTextAnalyzerWithWeights creates a new analyzer with custom weights.
// Use this for fine-tuning detection for specific domains or use cases.
func NewTextAnalyzerWithWeights(weights TextAnalyzerWeights) *TextAnalyzer {
	return &TextAnalyzer{
		weights: weights,
	}
}

// SetWeights updates the analyzer's weights.
func (a *TextAnalyzer) SetWeights(weights TextAnalyzerWeights) {
	a.weights = weights
}

// GetWeights returns the current weights.
func (a *TextAnalyzer) GetWeights() TextAnalyzerWeights {
	return a.weights
}

// WeightsPreset defines preset weight configurations for different use cases.
type WeightsPreset string

const (
	// PresetBalanced is the default balanced preset for general use.
	PresetBalanced WeightsPreset = "balanced"
	// PresetStrict emphasizes AI phrase detection and formal patterns.
	PresetStrict WeightsPreset = "strict"
	// PresetLenient is more forgiving, useful for technical/formal writing.
	PresetLenient WeightsPreset = "lenient"
	// PresetCreative is tuned for creative writing detection.
	PresetCreative WeightsPreset = "creative"
	// PresetAcademic is tuned for academic/research writing.
	PresetAcademic WeightsPreset = "academic"
)

// GetPresetWeights returns weights for a specific preset.
func GetPresetWeights(preset WeightsPreset) TextAnalyzerWeights {
	switch preset {
	case PresetStrict:
		// Emphasizes AI phrase detection and formal patterns
		return TextAnalyzerWeights{
			SentenceVariance:    0.08,
			VocabularyRichness:  0.12,
			Burstiness:          0.05,
			PunctuationVariety:  0.05,
			AIPhraseDetection:   0.30, // Higher weight for AI phrases
			WordLengthVariance:  0.05,
			ContractionsUsage:   0.12, // Higher weight - lack of contractions is suspicious
			RepetitionPenalty:   0.08,
			ReadabilityVariance: 0.08,
			NGramUniformity:     0.07,
		}
	case PresetLenient:
		// More forgiving, reduces penalty for formal writing patterns
		return TextAnalyzerWeights{
			SentenceVariance:    0.12,
			VocabularyRichness:  0.18,
			Burstiness:          0.10,
			PunctuationVariety:  0.08,
			AIPhraseDetection:   0.12, // Lower weight for AI phrases
			WordLengthVariance:  0.08,
			ContractionsUsage:   0.05, // Lower weight - formal writing is OK
			RepetitionPenalty:   0.10,
			ReadabilityVariance: 0.08,
			NGramUniformity:     0.09,
		}
	case PresetCreative:
		// Tuned for creative writing - emphasizes variance and personality
		return TextAnalyzerWeights{
			SentenceVariance:    0.15, // Creative writing has varied rhythm
			VocabularyRichness:  0.20, // Rich vocabulary expected
			Burstiness:          0.10,
			PunctuationVariety:  0.10, // Creative punctuation
			AIPhraseDetection:   0.10,
			WordLengthVariance:  0.08,
			ContractionsUsage:   0.07,
			RepetitionPenalty:   0.05, // Some repetition is stylistic
			ReadabilityVariance: 0.10, // Varied complexity is artistic
			NGramUniformity:     0.05,
		}
	case PresetAcademic:
		// Tuned for academic/research writing
		return TextAnalyzerWeights{
			SentenceVariance:    0.08,
			VocabularyRichness:  0.20, // Academic writing uses specialized vocabulary
			Burstiness:          0.08,
			PunctuationVariety:  0.05, // Less variety expected
			AIPhraseDetection:   0.18,
			WordLengthVariance:  0.06,
			ContractionsUsage:   0.03, // Contractions not expected in academic writing
			RepetitionPenalty:   0.12, // Repetitive structure is suspicious
			ReadabilityVariance: 0.08,
			NGramUniformity:     0.12, // Uniform n-grams suspicious in academic text
		}
	default: // PresetBalanced or unknown
		return DefaultWeights()
	}
}

// NewTextAnalyzerWithPreset creates a new analyzer with a preset configuration.
func NewTextAnalyzerWithPreset(preset WeightsPreset) *TextAnalyzer {
	return &TextAnalyzer{
		weights: GetPresetWeights(preset),
	}
}

// TextAnalysisResult contains detailed analysis results.
type TextAnalysisResult struct {
	// Final AI probability (0.0 = human, 1.0 = AI)
	AIScore float64

	// Individual signal scores (0.0 = human-like, 1.0 = AI-like)
	Signals TextSignals

	// Detected AI phrases
	DetectedAIPhrases []string

	// Statistics
	Stats TextStats
}

// TextSignals contains individual signal scores.
type TextSignals struct {
	SentenceVariance    float64 // Low variance = AI-like
	VocabularyRichness  float64 // Low richness = AI-like
	Burstiness          float64 // Low burstiness = AI-like
	PunctuationVariety  float64 // Low variety = AI-like
	AIPhraseScore       float64 // High = AI-like
	WordLengthVariance  float64 // Low variance = AI-like
	ContractionsUsage   float64 // Low usage = AI-like
	RepetitionScore     float64 // High repetition = AI-like
	ReadabilityVariance float64 // Low variance = AI-like (consistent reading level)
	NGramUniformity     float64 // High uniformity = AI-like (even n-gram distribution)
}

// TextStats contains raw statistics about the text.
type TextStats struct {
	CharCount          int
	WordCount          int
	SentenceCount      int
	AvgSentenceLen     float64
	AvgWordLen         float64
	UniqueWords        int
	UniqueRatio        float64
	PunctuationCount   int
	FleschKincaidGrade float64 // Flesch-Kincaid Grade Level (higher = more complex)
	GunningFogIndex    float64 // Gunning Fog Index (higher = more complex)
}

// Analyze performs comprehensive text analysis.
func (a *TextAnalyzer) Analyze(text string) TextAnalysisResult {
	result := TextAnalysisResult{}

	// Calculate basic stats (includes readability scores)
	result.Stats = a.calculateStats(text)

	// Calculate individual signals
	result.Signals.SentenceVariance = a.analyzeSentenceVariance(text)
	result.Signals.VocabularyRichness = a.analyzeVocabularyRichness(text)
	result.Signals.Burstiness = a.analyzeBurstiness(text)
	result.Signals.PunctuationVariety = a.analyzePunctuationVariety(text)
	result.Signals.AIPhraseScore, result.DetectedAIPhrases = a.detectAIPhrases(text)
	result.Signals.WordLengthVariance = a.analyzeWordLengthVariance(text)
	result.Signals.ContractionsUsage = a.analyzeContractions(text)
	result.Signals.RepetitionScore = a.analyzeRepetition(text)
	result.Signals.ReadabilityVariance = a.analyzeReadabilityVariance(text)
	result.Signals.NGramUniformity = a.analyzeNGramUniformity(text)

	// Calculate weighted AI score
	result.AIScore = a.calculateWeightedScore(result.Signals)

	return result
}

// calculateStats computes basic text statistics.
func (a *TextAnalyzer) calculateStats(text string) TextStats {
	stats := TextStats{}

	stats.CharCount = len(text)

	// Count words
	words := tokenize(text)
	stats.WordCount = len(words)

	// Count sentences
	sentences := splitSentences(text)
	stats.SentenceCount = len(sentences)

	// Average sentence length
	if stats.SentenceCount > 0 {
		totalWords := 0
		for _, s := range sentences {
			totalWords += len(tokenize(s))
		}
		stats.AvgSentenceLen = float64(totalWords) / float64(stats.SentenceCount)
	}

	// Average word length
	if stats.WordCount > 0 {
		totalChars := 0
		for _, w := range words {
			totalChars += len(w)
		}
		stats.AvgWordLen = float64(totalChars) / float64(stats.WordCount)
	}

	// Unique words
	unique := make(map[string]bool)
	for _, w := range words {
		unique[strings.ToLower(w)] = true
	}
	stats.UniqueWords = len(unique)

	if stats.WordCount > 0 {
		stats.UniqueRatio = float64(stats.UniqueWords) / float64(stats.WordCount)
	}

	// Punctuation count
	for _, r := range text {
		if unicode.IsPunct(r) {
			stats.PunctuationCount++
		}
	}

	// Readability scores
	stats.FleschKincaidGrade = calculateFleschKincaidGrade(text)
	stats.GunningFogIndex = calculateGunningFogIndex(text)

	return stats
}

// analyzeSentenceVariance measures variance in sentence lengths.
// Humans write with varied sentence lengths; AI tends to be uniform.
func (a *TextAnalyzer) analyzeSentenceVariance(text string) float64 {
	sentences := splitSentences(text)
	if len(sentences) < 3 {
		return 0.5 // Not enough data
	}

	// Calculate sentence lengths
	lengths := make([]float64, len(sentences))
	sum := 0.0
	for i, s := range sentences {
		words := tokenize(s)
		lengths[i] = float64(len(words))
		sum += lengths[i]
	}

	// Calculate mean
	mean := sum / float64(len(lengths))

	// Calculate variance
	variance := 0.0
	for _, l := range lengths {
		variance += (l - mean) * (l - mean)
	}
	variance /= float64(len(lengths))

	// Standard deviation
	stdDev := math.Sqrt(variance)

	// Coefficient of variation (normalized)
	cv := 0.0
	if mean > 0 {
		cv = stdDev / mean
	}

	// Human text typically has CV > 0.5
	// AI text typically has CV < 0.3
	// Research shows CV of 0.5 is a good threshold for human-like variance
	// Convert to AI score (low variance = high AI score)
	aiScore := 1.0 - math.Min(cv/0.5, 1.0)

	return aiScore
}

// analyzeVocabularyRichness measures lexical diversity.
// Humans use more varied vocabulary; AI uses "safe" common words.
func (a *TextAnalyzer) analyzeVocabularyRichness(text string) float64 {
	words := tokenize(text)
	if len(words) < 10 {
		return 0.5 // Not enough data
	}

	// Type-Token Ratio (TTR)
	unique := make(map[string]bool)
	for _, w := range words {
		unique[strings.ToLower(w)] = true
	}
	ttr := float64(len(unique)) / float64(len(words))

	// Check for rare/unusual words (not in common vocabulary)
	uncommonCount := 0
	for w := range unique {
		if !isCommonWord(w) && len(w) > 3 {
			uncommonCount++
		}
	}
	uncommonRatio := float64(uncommonCount) / float64(len(unique))

	// Human text: higher TTR, more uncommon words
	// Normalize TTR (human typically > 0.5, AI typically < 0.4)
	ttrScore := 1.0 - math.Min(ttr/0.6, 1.0)

	// Combine scores
	aiScore := (ttrScore*0.6 + (1.0-uncommonRatio)*0.4)

	return math.Max(0, math.Min(1, aiScore))
}

// analyzeBurstiness measures topic word clustering.
// Humans tend to cluster related words; AI distributes them evenly.
func (a *TextAnalyzer) analyzeBurstiness(text string) float64 {
	words := tokenize(text)
	if len(words) < 20 {
		return 0.5
	}

	// Find repeated content words
	wordPositions := make(map[string][]int)
	for i, w := range words {
		w = strings.ToLower(w)
		if len(w) > 4 && !isCommonWord(w) {
			wordPositions[w] = append(wordPositions[w], i)
		}
	}

	// Calculate burstiness for words that appear multiple times
	totalBurstiness := 0.0
	count := 0

	for _, positions := range wordPositions {
		if len(positions) < 2 {
			continue
		}

		// Calculate gaps between occurrences
		gaps := make([]float64, len(positions)-1)
		for i := 1; i < len(positions); i++ {
			gaps[i-1] = float64(positions[i] - positions[i-1])
		}

		// Calculate variance of gaps
		mean := 0.0
		for _, g := range gaps {
			mean += g
		}
		mean /= float64(len(gaps))

		variance := 0.0
		for _, g := range gaps {
			variance += (g - mean) * (g - mean)
		}
		variance /= float64(len(gaps))

		// Burstiness: high variance in gaps = bursty (human-like)
		if mean > 0 {
			totalBurstiness += math.Sqrt(variance) / mean
		}
		count++
	}

	if count == 0 {
		return 0.5
	}

	avgBurstiness := totalBurstiness / float64(count)

	// Low burstiness = AI-like
	aiScore := 1.0 - math.Min(avgBurstiness/1.5, 1.0)

	return aiScore
}

// analyzePunctuationVariety measures punctuation diversity.
// Humans use varied punctuation; AI tends to stick to periods and commas.
func (a *TextAnalyzer) analyzePunctuationVariety(text string) float64 {
	punctCounts := make(map[rune]int)
	totalPunct := 0

	for _, r := range text {
		if unicode.IsPunct(r) {
			punctCounts[r]++
			totalPunct++
		}
	}

	if totalPunct < 5 {
		return 0.5
	}

	// Count unique punctuation types
	uniquePunct := len(punctCounts)

	// Check for varied punctuation (!, ?, ;, :, -, etc.)
	variedPunct := 0
	interestingPunct := []rune{'!', '?', ';', ':', '-', '—', '(', ')', '"', '\''}
	for _, p := range interestingPunct {
		if punctCounts[p] > 0 {
			variedPunct++
		}
	}

	// Human text typically has 5+ different punctuation marks
	// AI often uses only . and ,
	varietyScore := float64(uniquePunct) / 8.0
	interestingScore := float64(variedPunct) / 5.0

	// Low variety = AI-like
	aiScore := 1.0 - (varietyScore*0.5 + interestingScore*0.5)

	return math.Max(0, math.Min(1, aiScore))
}

// detectAIPhrases looks for common AI writing patterns.
func (a *TextAnalyzer) detectAIPhrases(text string) (float64, []string) {
	lowerText := strings.ToLower(text)
	detected := []string{}

	// Comprehensive AI phrase database organized by category
	// Weight: 1.0 = definitive AI, 0.3 = weak signal
	aiPhrases := []struct {
		pattern string
		weight  float64
	}{
		// ============================================
		// CATEGORY 1: Direct AI Self-References (1.0)
		// ============================================
		{"as an ai", 1.0},
		{"as a language model", 1.0},
		{"as an artificial intelligence", 1.0},
		{"i'm an ai", 1.0},
		{"i am an ai", 1.0},
		{"as a large language model", 1.0},
		{"i'm just an ai", 1.0},
		{"i'm a language model", 1.0},
		{"as your ai assistant", 1.0},
		{"i don't have personal experiences", 0.95},
		{"i don't have feelings", 0.9},
		{"i don't have opinions", 0.9},
		{"i cannot have personal", 0.9},
		{"i don't have consciousness", 0.95},
		{"i was trained by", 0.85},
		{"my training data", 0.85},
		{"my knowledge cutoff", 0.9},
		{"based on my training", 0.85},

		// ============================================
		// CATEGORY 2: Inability Statements (0.7-0.9)
		// ============================================
		{"i cannot provide", 0.8},
		{"i'm unable to", 0.7},
		{"i am unable to", 0.7},
		{"i can't actually", 0.7},
		{"i cannot actually", 0.7},
		{"i'm not able to", 0.7},
		{"i don't have access to", 0.7},
		{"i cannot access", 0.7},
		{"i cannot browse", 0.8},
		{"i cannot search", 0.75},
		{"i cannot verify", 0.6},
		{"i cannot confirm", 0.6},

		// ============================================
		// CATEGORY 3: Claude-Specific Patterns (0.8-0.9)
		// ============================================
		{"i'd be happy to", 0.7},
		{"i'd be glad to", 0.7},
		{"i should note", 0.7},
		{"i should mention", 0.65},
		{"i appreciate you", 0.5},
		{"i appreciate your", 0.5},
		{"that said,", 0.4},
		{"with that said", 0.4},
		{"that being said", 0.45},
		{"i want to be", 0.4},
		{"i aim to", 0.5},

		// ============================================
		// CATEGORY 4: GPT-Specific Patterns (0.7-0.8)
		// ============================================
		{"certainly!", 0.6},
		{"absolutely!", 0.5},
		{"great question!", 0.7},
		{"excellent question", 0.7},
		{"that's a great question", 0.7},
		{"happy to help", 0.6},
		{"glad you asked", 0.6},
		{"sure thing!", 0.5},
		{"of course!", 0.4},

		// ============================================
		// CATEGORY 5: Hedging & Caveats (0.5-0.8)
		// ============================================
		{"it's important to note", 0.8},
		{"it is important to", 0.7},
		{"it's worth noting", 0.7},
		{"it is worth noting", 0.7},
		{"it should be noted", 0.7},
		{"it bears mentioning", 0.7},
		{"keep in mind that", 0.6},
		{"bear in mind that", 0.6},
		{"please note that", 0.5},
		{"do note that", 0.5},
		{"it's crucial to", 0.6},
		{"it is crucial to", 0.6},
		{"it's essential to", 0.6},
		{"it is essential to", 0.6},
		{"it's vital to", 0.55},
		{"importantly,", 0.5},
		{"crucially,", 0.5},
		{"notably,", 0.45},
		{"significantly,", 0.4},
		{"interestingly,", 0.45},

		// ============================================
		// CATEGORY 6: Transitions (0.3-0.5)
		// ============================================
		{"furthermore", 0.4},
		{"moreover", 0.4},
		{"additionally", 0.4},
		{"in addition to this", 0.4},
		{"in conclusion", 0.5},
		{"to summarize", 0.5},
		{"in summary", 0.5},
		{"to conclude", 0.5},
		{"to sum up", 0.45},
		{"all in all", 0.35},
		{"on the whole", 0.35},
		{"by and large", 0.35},
		{"in essence", 0.4},
		{"essentially,", 0.35},
		{"fundamentally,", 0.4},
		{"ultimately,", 0.35},
		{"overall,", 0.3},
		{"consequently,", 0.35},
		{"accordingly,", 0.35},
		{"thus,", 0.3},
		{"hence,", 0.35},
		{"therefore,", 0.3},
		{"as such,", 0.4},
		{"that being said,", 0.45},
		{"having said that,", 0.45},
		{"with that in mind,", 0.45},

		// ============================================
		// CATEGORY 7: Helpful Closings (0.5-0.7)
		// ============================================
		{"i hope this helps", 0.7},
		{"hope this helps", 0.6},
		{"i hope that helps", 0.7},
		{"hope that helps", 0.6},
		{"i hope this was helpful", 0.7},
		{"feel free to", 0.5},
		{"don't hesitate to", 0.5},
		{"please don't hesitate", 0.5},
		{"let me know if", 0.4},
		{"let me know if you", 0.45},
		{"if you have any questions", 0.5},
		{"if you have any further", 0.55},
		{"if you need any clarification", 0.55},
		{"if there's anything else", 0.5},
		{"is there anything else", 0.5},
		{"happy to clarify", 0.55},
		{"happy to elaborate", 0.55},
		{"happy to explain", 0.5},

		// ============================================
		// CATEGORY 8: Formal/Stilted Vocabulary (0.3-0.6)
		// ============================================
		{"utilize", 0.35},
		{"utilization", 0.35},
		{"facilitate", 0.35},
		{"facilitation", 0.35},
		{"leverage", 0.35},
		{"leveraging", 0.35},
		{"delve into", 0.6},
		{"delve deeper", 0.55},
		{"delving into", 0.55},
		{"dive into", 0.4},
		{"dive deeper", 0.4},
		{"explore the", 0.3},
		{"unpack this", 0.5},
		{"unpacking the", 0.5},
		{"navigate the", 0.4},
		{"navigating the", 0.4},
		{"embark on", 0.45},
		{"embarking on", 0.45},
		{"endeavor to", 0.5},
		{"endeavour to", 0.5},
		{"commence with", 0.45},
		{"prior to", 0.3},
		{"subsequent to", 0.4},
		{"in lieu of", 0.4},
		{"in order to", 0.25},
		{"pertaining to", 0.4},
		{"with regard to", 0.35},
		{"with regards to", 0.35},
		{"in regard to", 0.35},
		{"vis-à-vis", 0.45},
		{"myriad of", 0.5},
		{"a myriad", 0.45},
		{"plethora of", 0.5},
		{"multifaceted", 0.45},
		{"nuanced", 0.35},
		{"holistic", 0.4},
		{"synergy", 0.45},
		{"synergies", 0.45},
		{"paradigm", 0.4},
		{"paradigm shift", 0.45},
		{"cutting-edge", 0.35},
		{"state-of-the-art", 0.35},

		// ============================================
		// CATEGORY 9: List Introductions (0.4-0.6)
		// ============================================
		{"here are some", 0.5},
		{"here's a list", 0.5},
		{"the following", 0.4},
		{"here are a few", 0.5},
		{"here are the key", 0.5},
		{"here are the main", 0.5},
		{"let me outline", 0.5},
		{"let me break down", 0.5},
		{"let me explain", 0.4},
		{"allow me to", 0.45},
		{"i'll outline", 0.45},
		{"i will outline", 0.45},
		{"i'll break down", 0.45},
		{"consider the following", 0.5},
		{"there are several", 0.35},
		{"there are a few", 0.35},
		{"there are multiple", 0.35},

		// ============================================
		// CATEGORY 10: Acknowledgment Phrases (0.4-0.6)
		// ============================================
		{"you raise a good point", 0.6},
		{"you make a good point", 0.6},
		{"that's a valid point", 0.55},
		{"that's a fair point", 0.55},
		{"you're absolutely right", 0.55},
		{"you are absolutely right", 0.55},
		{"you're correct", 0.4},
		{"you are correct", 0.4},
		{"i understand your", 0.4},
		{"i see what you", 0.4},

		// ============================================
		// CATEGORY 11: Safety/Disclaimer Language (0.6-0.9)
		// ============================================
		{"i would recommend consulting", 0.7},
		{"consult with a professional", 0.6},
		{"seek professional advice", 0.6},
		{"consult a doctor", 0.55},
		{"consult a lawyer", 0.55},
		{"this is not professional advice", 0.8},
		{"this is not legal advice", 0.8},
		{"this is not medical advice", 0.8},
		{"this is not financial advice", 0.8},
		{"please consult", 0.5},
		{"i recommend speaking with", 0.6},
		{"for safety reasons", 0.5},
		{"for your safety", 0.45},

		// ============================================
		// CATEGORY 12: Filler Expressions (0.3-0.5)
		// ============================================
		{"in today's world", 0.4},
		{"in this day and age", 0.45},
		{"in the modern era", 0.4},
		{"it goes without saying", 0.45},
		{"needless to say", 0.4},
		{"it is widely known", 0.45},
		{"as we all know", 0.4},
		{"as you may know", 0.4},
		{"as you might expect", 0.4},
		{"as one might expect", 0.45},
		{"it is worth mentioning", 0.5},
		{"this brings us to", 0.45},
		{"this leads us to", 0.45},
		{"let's take a look", 0.4},
		{"let us consider", 0.4},
		{"when it comes to", 0.35},

		// ============================================
		// CATEGORY 13: Emphasis Patterns (0.3-0.5)
		// ============================================
		{"it's really important", 0.4},
		{"this is really important", 0.4},
		{"particularly important", 0.35},
		{"especially important", 0.35},
		{"extremely important", 0.4},
		{"absolutely essential", 0.45},
		{"critically important", 0.45},
		{"cannot be overstated", 0.5},
		{"can't be overstated", 0.5},

		// ============================================
		// CATEGORY 14: Comparative/Balanced Language (0.35-0.5)
		// ============================================
		{"on the one hand", 0.4},
		{"on the other hand", 0.35},
		{"while it's true that", 0.45},
		{"while this is true", 0.45},
		{"although this may be", 0.4},
		{"despite the fact that", 0.4},
		{"notwithstanding", 0.45},
		{"nonetheless", 0.35},
		{"nevertheless", 0.35},
		{"be that as it may", 0.5},
	}

	totalWeight := 0.0
	matchCount := 0

	for _, phrase := range aiPhrases {
		if strings.Contains(lowerText, phrase.pattern) {
			detected = append(detected, phrase.pattern)
			totalWeight += phrase.weight
			matchCount++
		}
	}

	// Calculate AI score based on matches
	// More matches = higher AI probability
	if matchCount == 0 {
		return 0.0, detected
	}

	// Normalize by text length (longer text might naturally have more matches)
	wordCount := len(tokenize(text))
	normalizedScore := totalWeight / (float64(wordCount) / 100.0)

	aiScore := math.Min(normalizedScore, 1.0)

	return aiScore, detected
}

// analyzeWordLengthVariance measures variance in word lengths.
func (a *TextAnalyzer) analyzeWordLengthVariance(text string) float64 {
	words := tokenize(text)
	if len(words) < 10 {
		return 0.5
	}

	// Calculate word lengths
	sum := 0.0
	for _, w := range words {
		sum += float64(len(w))
	}
	mean := sum / float64(len(words))

	// Calculate variance
	variance := 0.0
	for _, w := range words {
		diff := float64(len(w)) - mean
		variance += diff * diff
	}
	variance /= float64(len(words))
	stdDev := math.Sqrt(variance)

	// Human text has more varied word lengths
	// Coefficient of variation
	cv := 0.0
	if mean > 0 {
		cv = stdDev / mean
	}

	// Low variance = AI-like
	aiScore := 1.0 - math.Min(cv/0.6, 1.0)

	return aiScore
}

// analyzeContractions checks for contraction usage.
// Humans use contractions; formal AI often doesn't.
func (a *TextAnalyzer) analyzeContractions(text string) float64 {
	contractions := []string{
		"i'm", "i'll", "i've", "i'd",
		"you're", "you'll", "you've", "you'd",
		"he's", "she's", "it's", "we're", "they're",
		"don't", "doesn't", "didn't", "won't", "wouldn't",
		"can't", "couldn't", "shouldn't", "isn't", "aren't",
		"wasn't", "weren't", "haven't", "hasn't", "hadn't",
		"let's", "that's", "there's", "here's", "what's",
		"who's", "how's", "where's", "when's",
	}

	lowerText := strings.ToLower(text)
	wordCount := len(tokenize(text))

	if wordCount < 20 {
		return 0.5
	}

	contractionCount := 0
	for _, c := range contractions {
		contractionCount += strings.Count(lowerText, c)
	}

	// Contractions per 100 words
	contractionRate := float64(contractionCount) / (float64(wordCount) / 100.0)

	// Human casual text: 2-5 contractions per 100 words
	// Formal AI: often 0-1 contractions per 100 words
	// Convert to AI score (low contractions = AI-like)
	aiScore := 1.0 - math.Min(contractionRate/3.0, 1.0)

	return aiScore
}

// analyzeRepetition checks for repetitive patterns.
// AI sometimes repeats phrases or structures.
func (a *TextAnalyzer) analyzeRepetition(text string) float64 {
	sentences := splitSentences(text)
	if len(sentences) < 3 {
		return 0.5
	}

	// Check for repeated sentence starts
	starts := make(map[string]int)
	for _, s := range sentences {
		words := tokenize(s)
		if len(words) >= 2 {
			start := strings.ToLower(words[0] + " " + words[1])
			starts[start]++
		}
	}

	// Count repetitions
	repetitions := 0
	for _, count := range starts {
		if count > 1 {
			repetitions += count - 1
		}
	}

	// Repetition rate
	repRate := float64(repetitions) / float64(len(sentences))

	// High repetition = AI-like
	aiScore := math.Min(repRate*2, 1.0)

	return aiScore
}

// analyzeNGramUniformity measures the uniformity of bigram and trigram distributions.
// AI-generated text tends to have more uniform n-gram distributions;
// human text has more varied, personalized patterns.
func (a *TextAnalyzer) analyzeNGramUniformity(text string) float64 {
	words := tokenize(text)
	if len(words) < 15 {
		return 0.5 // Not enough data
	}

	// Build bigram frequency map
	bigrams := make(map[string]int)
	for i := 0; i < len(words)-1; i++ {
		bigram := strings.ToLower(words[i] + " " + words[i+1])
		bigrams[bigram]++
	}

	// Build trigram frequency map
	trigrams := make(map[string]int)
	for i := 0; i < len(words)-2; i++ {
		trigram := strings.ToLower(words[i] + " " + words[i+1] + " " + words[i+2])
		trigrams[trigram]++
	}

	// Calculate bigram entropy (higher entropy = more uniform = AI-like)
	bigramEntropy := calculateNGramEntropy(bigrams, len(words)-1)

	// Calculate trigram entropy
	trigramEntropy := calculateNGramEntropy(trigrams, len(words)-2)

	// Count repeated n-grams (human text often has personal catchphrases)
	repeatedBigrams := 0
	for _, count := range bigrams {
		if count > 1 {
			repeatedBigrams++
		}
	}

	repeatedTrigrams := 0
	for _, count := range trigrams {
		if count > 1 {
			repeatedTrigrams++
		}
	}

	// Calculate uniformity score
	// High entropy + few repetitions = uniform = AI-like
	// Low entropy + more repetitions = personal patterns = human-like

	// Normalize entropy (max entropy for n-grams is ~log2(n) where n is number of unique n-grams)
	maxBigramEntropy := math.Log2(float64(len(bigrams)))
	maxTrigramEntropy := math.Log2(float64(len(trigrams)))

	normalizedBigramEntropy := 0.0
	if maxBigramEntropy > 0 {
		normalizedBigramEntropy = bigramEntropy / maxBigramEntropy
	}

	normalizedTrigramEntropy := 0.0
	if maxTrigramEntropy > 0 {
		normalizedTrigramEntropy = trigramEntropy / maxTrigramEntropy
	}

	// Repetition ratio (more repetition = human-like)
	bigramRepetitionRatio := float64(repeatedBigrams) / float64(len(bigrams))
	trigramRepetitionRatio := float64(repeatedTrigrams) / float64(len(trigrams))

	// Combine scores
	// High entropy (uniform) = AI-like (high score)
	// High repetition = human-like (low score)
	entropyScore := (normalizedBigramEntropy + normalizedTrigramEntropy) / 2
	repetitionScore := (bigramRepetitionRatio + trigramRepetitionRatio) / 2

	// AI score: high entropy, low repetition = AI-like
	aiScore := entropyScore * (1 - repetitionScore*0.5)

	return math.Max(0, math.Min(1, aiScore))
}

// calculateNGramEntropy computes Shannon entropy for n-gram distribution.
func calculateNGramEntropy(ngrams map[string]int, total int) float64 {
	if total <= 0 {
		return 0
	}

	entropy := 0.0
	for _, count := range ngrams {
		if count > 0 {
			p := float64(count) / float64(total)
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

// calculateWeightedScore combines all signals into final AI score.
func (a *TextAnalyzer) calculateWeightedScore(signals TextSignals) float64 {
	w := a.weights

	score := signals.SentenceVariance*w.SentenceVariance +
		signals.VocabularyRichness*w.VocabularyRichness +
		signals.Burstiness*w.Burstiness +
		signals.PunctuationVariety*w.PunctuationVariety +
		signals.AIPhraseScore*w.AIPhraseDetection +
		signals.WordLengthVariance*w.WordLengthVariance +
		signals.ContractionsUsage*w.ContractionsUsage +
		signals.RepetitionScore*w.RepetitionPenalty +
		signals.ReadabilityVariance*w.ReadabilityVariance +
		signals.NGramUniformity*w.NGramUniformity

	// Normalize to 0-1
	totalWeight := w.SentenceVariance + w.VocabularyRichness + w.Burstiness +
		w.PunctuationVariety + w.AIPhraseDetection + w.WordLengthVariance +
		w.ContractionsUsage + w.RepetitionPenalty + w.ReadabilityVariance +
		w.NGramUniformity

	if totalWeight > 0 {
		score /= totalWeight
	}

	return math.Max(0, math.Min(1, score))
}

// analyzeReadabilityVariance measures variance in reading level across paragraphs.
// AI-generated text tends to maintain consistent complexity; humans vary.
func (a *TextAnalyzer) analyzeReadabilityVariance(text string) float64 {
	sentences := splitSentences(text)
	if len(sentences) < 4 {
		return 0.5 // Not enough data
	}

	// Calculate Flesch-Kincaid grade for each sentence group (pseudo-paragraphs)
	// Group sentences into chunks of 2-3 for paragraph-level analysis
	chunkSize := 2
	if len(sentences) > 10 {
		chunkSize = 3
	}

	grades := make([]float64, 0)
	for i := 0; i < len(sentences); i += chunkSize {
		end := i + chunkSize
		if end > len(sentences) {
			end = len(sentences)
		}

		// Combine sentences in this chunk
		chunk := ""
		for j := i; j < end; j++ {
			chunk += sentences[j] + ". "
		}

		grade := calculateFleschKincaidGrade(chunk)
		if grade > 0 {
			grades = append(grades, grade)
		}
	}

	if len(grades) < 2 {
		return 0.5
	}

	// Calculate variance of grades
	mean := 0.0
	for _, g := range grades {
		mean += g
	}
	mean /= float64(len(grades))

	variance := 0.0
	for _, g := range grades {
		diff := g - mean
		variance += diff * diff
	}
	variance /= float64(len(grades))
	stdDev := math.Sqrt(variance)

	// Coefficient of variation
	cv := 0.0
	if mean > 0 {
		cv = stdDev / mean
	}

	// Human text typically varies in complexity (CV > 0.2)
	// AI text maintains consistent level (CV < 0.1)
	// Low variance = AI-like
	aiScore := 1.0 - math.Min(cv/0.3, 1.0)

	return aiScore
}

// calculateFleschKincaidGrade computes the Flesch-Kincaid Grade Level.
// Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
func calculateFleschKincaidGrade(text string) float64 {
	words := tokenize(text)
	sentences := splitSentences(text)

	if len(words) == 0 || len(sentences) == 0 {
		return 0
	}

	totalSyllables := 0
	for _, word := range words {
		totalSyllables += countSyllables(word)
	}

	wordsPerSentence := float64(len(words)) / float64(len(sentences))
	syllablesPerWord := float64(totalSyllables) / float64(len(words))

	grade := 0.39*wordsPerSentence + 11.8*syllablesPerWord - 15.59

	// Clamp to reasonable range (K-12+)
	return math.Max(0, math.Min(18, grade))
}

// calculateGunningFogIndex computes the Gunning Fog Index.
// Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
// Complex words = words with 3+ syllables (excluding common suffixes)
func calculateGunningFogIndex(text string) float64 {
	words := tokenize(text)
	sentences := splitSentences(text)

	if len(words) == 0 || len(sentences) == 0 {
		return 0
	}

	complexWords := 0
	for _, word := range words {
		syllables := countSyllables(word)
		// Complex word: 3+ syllables, excluding common suffixes
		if syllables >= 3 {
			lower := strings.ToLower(word)
			// Don't count words ending in common suffixes as complex
			if !strings.HasSuffix(lower, "ed") &&
				!strings.HasSuffix(lower, "es") &&
				!strings.HasSuffix(lower, "ing") {
				complexWords++
			}
		}
	}

	wordsPerSentence := float64(len(words)) / float64(len(sentences))
	complexRatio := float64(complexWords) / float64(len(words))

	fog := 0.4 * (wordsPerSentence + 100*complexRatio)

	return math.Max(0, math.Min(20, fog))
}

// countSyllables estimates syllable count in a word.
// Uses a simple vowel-counting heuristic with adjustments.
func countSyllables(word string) int {
	word = strings.ToLower(word)
	if len(word) == 0 {
		return 0
	}

	vowels := "aeiouy"
	count := 0
	prevVowel := false

	for i, r := range word {
		isVowel := strings.ContainsRune(vowels, r)

		if isVowel && !prevVowel {
			count++
		}
		prevVowel = isVowel

		// Silent 'e' at end
		if i == len(word)-1 && r == 'e' && count > 1 {
			count--
		}
	}

	// Every word has at least one syllable
	if count == 0 {
		count = 1
	}

	return count
}

// =============================================================================
// Helper Functions
// =============================================================================

// tokenize splits text into words using improved tokenization.
// Handles contractions, hyphenated words, numbers, and basic unicode.
func tokenize(text string) []string {
	// Improved tokenization that handles:
	// - Contractions (don't, I'm, etc.)
	// - Hyphenated words (state-of-the-art)
	// - Words with numbers (COVID-19, B2B)
	// - Possessives (John's)
	// - Unicode letters (café, naïve)

	// First, normalize some common patterns
	text = normalizeText(text)

	// Match word tokens including:
	// - Words with apostrophes for contractions (don't, I'm)
	// - Hyphenated compound words (well-known)
	// - Words with embedded numbers (B2B, COVID-19)
	// - Unicode word characters
	re := regexp.MustCompile(`[\p{L}\p{N}]+(?:[-'][\p{L}\p{N}]+)*`)
	tokens := re.FindAllString(text, -1)

	// Filter out pure numbers and single characters (except 'I' and 'a')
	result := make([]string, 0, len(tokens))
	for _, token := range tokens {
		// Skip pure numbers
		if isNumeric(token) {
			continue
		}
		// Skip single characters except common words
		if len(token) == 1 {
			lower := strings.ToLower(token)
			if lower != "i" && lower != "a" {
				continue
			}
		}
		result = append(result, token)
	}

	return result
}

// normalizeText performs text normalization for better tokenization.
func normalizeText(text string) string {
	// Replace smart quotes with regular quotes
	// Right single quote (') \u2019
	text = strings.ReplaceAll(text, "\u2019", "'")
	// Left single quote (') \u2018
	text = strings.ReplaceAll(text, "\u2018", "'")
	// Left double quote (") \u201c
	text = strings.ReplaceAll(text, "\u201c", "\"")
	// Right double quote (") \u201d
	text = strings.ReplaceAll(text, "\u201d", "\"")

	// Replace em-dash and en-dash with regular dash
	// Em-dash (—) \u2014
	text = strings.ReplaceAll(text, "\u2014", "-")
	// En-dash (–) \u2013
	text = strings.ReplaceAll(text, "\u2013", "-")

	return text
}

// isNumeric checks if a string contains only numeric characters.
func isNumeric(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return len(s) > 0
}

// splitSentences splits text into sentences.
func splitSentences(text string) []string {
	// Split on sentence-ending punctuation
	re := regexp.MustCompile(`[.!?]+\s+`)
	parts := re.Split(text, -1)

	// Filter empty strings
	sentences := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if len(p) > 0 {
			sentences = append(sentences, p)
		}
	}

	return sentences
}

// isCommonWord checks if a word is in the common vocabulary.
// Common words are less indicative of human/AI authorship.
func isCommonWord(word string) bool {
	commonWords := map[string]bool{
		// Articles
		"a": true, "an": true, "the": true,
		// Pronouns
		"i": true, "you": true, "he": true, "she": true, "it": true,
		"we": true, "they": true, "me": true, "him": true, "her": true,
		"us": true, "them": true, "my": true, "your": true, "his": true,
		"its": true, "our": true, "their": true, "this": true, "that": true,
		// Prepositions
		"in": true, "on": true, "at": true, "to": true, "for": true,
		"of": true, "with": true, "by": true, "from": true, "about": true,
		"into": true, "through": true, "during": true, "before": true, "after": true,
		// Conjunctions
		"and": true, "or": true, "but": true, "so": true, "yet": true,
		"if": true, "when": true, "while": true, "because": true, "although": true,
		// Verbs
		"is": true, "are": true, "was": true, "were": true, "be": true,
		"been": true, "being": true, "have": true, "has": true, "had": true,
		"do": true, "does": true, "did": true, "will": true, "would": true,
		"could": true, "should": true, "may": true, "might": true, "must": true,
		"can": true, "get": true, "got": true, "make": true, "made": true,
		// Common words
		"not": true, "no": true, "yes": true, "just": true, "only": true,
		"also": true, "very": true, "more": true, "most": true, "some": true,
		"any": true, "all": true, "many": true, "much": true, "other": true,
		"such": true, "than": true, "then": true, "now": true, "here": true,
		"there": true, "where": true, "what": true, "which": true, "who": true,
		"how": true, "why": true, "each": true, "every": true, "both": true,
		"few": true, "new": true, "old": true, "good": true, "bad": true,
		"first": true, "last": true, "long": true, "little": true, "own": true,
		"same": true, "big": true, "high": true, "small": true, "large": true,
		"next": true, "early": true, "young": true, "important": true, "public": true,
		"able": true, "man": true, "woman": true, "time": true, "year": true,
		"people": true, "way": true, "day": true, "thing": true, "world": true,
		"life": true, "hand": true, "part": true, "place": true, "case": true,
		"week": true, "work": true, "fact": true, "group": true, "number": true,
		"night": true, "point": true, "home": true, "water": true, "room": true,
		"mother": true, "area": true, "money": true, "story": true, "month": true,
		"lot": true, "right": true, "study": true, "book": true, "eye": true,
		"job": true, "word": true, "business": true, "issue": true, "side": true,
		"kind": true, "head": true, "house": true, "service": true, "friend": true,
		"father": true, "power": true, "hour": true, "game": true, "line": true,
		"end": true, "member": true, "law": true, "car": true, "city": true,
		"community": true, "name": true,
	}

	return commonWords[strings.ToLower(word)]
}
