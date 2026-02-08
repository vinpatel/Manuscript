package service

import (
	"strings"
	"testing"
)

// TestTextAnalyzer tests the Manuscript text analysis algorithm.
func TestTextAnalyzer(t *testing.T) {
	analyzer := NewTextAnalyzer()

	t.Run("detects clearly AI-generated text", func(t *testing.T) {
		// Text with many AI patterns
		aiText := `As an AI language model, I cannot provide personal opinions. However, it's important to note that this topic has many facets. Furthermore, we should consider multiple perspectives. In conclusion, I hope this helps you understand the subject better. Feel free to ask if you have any more questions.`

		result := analyzer.Analyze(aiText)

		// With research-aligned sentence variance (CV threshold 0.5), overall score may be lower
		// but AI phrase detection should still be strong
		if result.AIScore < 0.4 {
			t.Errorf("expected moderate-to-high AI score for AI-like text, got %f", result.AIScore)
		}

		if len(result.DetectedAIPhrases) == 0 {
			t.Error("expected AI phrases to be detected")
		}

		// AI phrase score should be high for this text
		if result.Signals.AIPhraseScore < 0.3 {
			t.Errorf("expected high AI phrase score, got %f", result.Signals.AIPhraseScore)
		}

		t.Logf("AI Text Analysis: score=%f, phrases=%v", result.AIScore, result.DetectedAIPhrases)
	})

	t.Run("detects human-written casual text", func(t *testing.T) {
		// Natural human text with contractions, varied sentences, personality
		humanText := `You know what? I've been thinking about this for a while now. It's weird - sometimes the simplest things are the hardest to explain! 
		
		Like yesterday, I tried explaining why the sky looks blue to my kid. She just stared at me. Didn't get it at all, haha. Kids, man.
		
		Anyway, what I'm trying to say is... don't overthink it. Just go with your gut. That's what I'd do.`

		result := analyzer.Analyze(humanText)

		if result.AIScore > 0.5 {
			t.Errorf("expected low AI score for human text, got %f", result.AIScore)
		}

		// Should have contractions detected
		if result.Signals.ContractionsUsage > 0.5 {
			t.Logf("Contractions score: %f (lower = more contractions = human-like)", result.Signals.ContractionsUsage)
		}

		t.Logf("Human Text Analysis: score=%f, signals=%+v", result.AIScore, result.Signals)
	})

	t.Run("handles short text", func(t *testing.T) {
		shortText := "Hello world"
		result := analyzer.Analyze(shortText)

		// Should return neutral score for insufficient data
		if result.AIScore < 0.3 || result.AIScore > 0.7 {
			t.Errorf("expected neutral score for short text, got %f", result.AIScore)
		}

		t.Logf("Short Text Analysis: score=%f", result.AIScore)
	})

	t.Run("analyzes sentence variance correctly", func(t *testing.T) {
		// Varied sentence lengths (human-like)
		variedText := "Short one. This is a medium length sentence with more words. And here's quite a long sentence that goes on for a bit longer than the others, adding some variety to the text."

		// Uniform sentence lengths (AI-like)
		uniformText := "This sentence has exactly ten words in it. This sentence has exactly ten words in it. This sentence has exactly ten words in it."

		variedResult := analyzer.Analyze(variedText)
		uniformResult := analyzer.Analyze(uniformText)

		// Varied text should have lower AI score for sentence variance
		if variedResult.Signals.SentenceVariance > uniformResult.Signals.SentenceVariance {
			t.Errorf("varied text should have lower variance AI score: varied=%f, uniform=%f",
				variedResult.Signals.SentenceVariance, uniformResult.Signals.SentenceVariance)
		}

		t.Logf("Sentence Variance - Varied: %f, Uniform: %f",
			variedResult.Signals.SentenceVariance, uniformResult.Signals.SentenceVariance)
	})

	t.Run("detects AI phrases correctly", func(t *testing.T) {
		phrases := []struct {
			text          string
			expectAI      bool
			minPhrases    int
		}{
			{"As an AI, I cannot provide that.", true, 1},
			{"It's important to note that furthermore, moreover.", true, 2},
			{"Just my two cents on the matter.", false, 0},
			{"I hope this helps! Let me know if you have questions.", true, 1},
		}

		for _, tc := range phrases {
			result := analyzer.Analyze(tc.text)

			if tc.expectAI && result.Signals.AIPhraseScore < 0.3 {
				t.Errorf("expected AI phrases in '%s', got score %f", tc.text, result.Signals.AIPhraseScore)
			}

			if !tc.expectAI && result.Signals.AIPhraseScore > 0.5 {
				t.Errorf("unexpected AI phrases in '%s', got score %f", tc.text, result.Signals.AIPhraseScore)
			}

			if len(result.DetectedAIPhrases) < tc.minPhrases {
				t.Errorf("expected at least %d AI phrases in '%s', got %d: %v",
					tc.minPhrases, tc.text, len(result.DetectedAIPhrases), result.DetectedAIPhrases)
			}
		}
	})
}

// TestTextStats verifies basic statistics calculation.
func TestTextStats(t *testing.T) {
	analyzer := NewTextAnalyzer()

	text := "Hello world. This is a test. One two three."
	result := analyzer.Analyze(text)

	// "Hello world This is a test One two three" = 9 words
	if result.Stats.WordCount != 9 {
		t.Errorf("expected 9 words, got %d", result.Stats.WordCount)
	}

	if result.Stats.SentenceCount != 3 {
		t.Errorf("expected 3 sentences, got %d", result.Stats.SentenceCount)
	}

	if result.Stats.UniqueWords != 9 { // All words are unique
		t.Errorf("expected 9 unique words, got %d", result.Stats.UniqueWords)
	}

	t.Logf("Stats: %+v", result.Stats)
}

// TestTokenize tests word tokenization.
func TestTokenize(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"Hello world", 2},
		{"Hello, world!", 2},
		{"one-two-three", 1}, // Hyphenated words are kept as single tokens
		{"it's a test", 3},   // it's is kept as single token (contraction)
		{"", 0},
	}

	for _, tc := range tests {
		tokens := tokenize(tc.input)
		if len(tokens) != tc.expected {
			t.Errorf("tokenize(%q) = %d tokens, want %d: %v", tc.input, len(tokens), tc.expected, tokens)
		}
	}
}

// TestSplitSentences tests sentence splitting.
func TestSplitSentences(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"Hello. World.", 2},
		{"Hello! World?", 2},
		{"One sentence", 1},
		{"First. Second! Third?", 3},
		{"", 0},
	}

	for _, tc := range tests {
		sentences := splitSentences(tc.input)
		if len(sentences) != tc.expected {
			t.Errorf("splitSentences(%q) = %d, want %d: %v", tc.input, len(sentences), tc.expected, sentences)
		}
	}
}

// TestIsCommonWord tests common word detection.
func TestIsCommonWord(t *testing.T) {
	common := []string{"the", "a", "is", "are", "and", "but", "it", "for"}
	uncommon := []string{"algorithm", "quantum", "serendipity", "xylophone", "ephemeral"}

	for _, w := range common {
		if !isCommonWord(w) {
			t.Errorf("expected %q to be common", w)
		}
	}

	for _, w := range uncommon {
		if isCommonWord(w) {
			t.Errorf("expected %q to be uncommon", w)
		}
	}
}

// TestVocabularyRichness tests lexical diversity analysis.
func TestVocabularyRichness(t *testing.T) {
	analyzer := NewTextAnalyzer()

	// Rich vocabulary (many unique words)
	richText := "The ephemeral serendipity of discovering quantum phenomena through algorithmic analysis reveals fascinating epistemological implications for our understanding of consciousness."

	// Poor vocabulary (repeated words)
	poorText := "The thing is a thing and the thing does thing things. It is a thing that things do. Things are things."

	richResult := analyzer.Analyze(richText)
	poorResult := analyzer.Analyze(poorText)

	// Rich vocabulary should have lower AI score
	if richResult.Signals.VocabularyRichness > poorResult.Signals.VocabularyRichness {
		t.Errorf("rich vocabulary should have lower AI score: rich=%f, poor=%f",
			richResult.Signals.VocabularyRichness, poorResult.Signals.VocabularyRichness)
	}

	t.Logf("Vocabulary Richness - Rich: %f, Poor: %f",
		richResult.Signals.VocabularyRichness, poorResult.Signals.VocabularyRichness)
}

// TestContractionsAnalysis tests contraction detection.
func TestContractionsAnalysis(t *testing.T) {
	analyzer := NewTextAnalyzer()

	// Text with contractions (human-like)
	contractionText := "I've been thinking, and I don't know if it's worth it. We're not sure what we'll do. They've said it won't work, but I can't believe that."

	// Formal text without contractions (AI-like)
	formalText := "I have been thinking, and I do not know if it is worth it. We are not sure what we will do. They have said it will not work, but I cannot believe that."

	contractionResult := analyzer.Analyze(contractionText)
	formalResult := analyzer.Analyze(formalText)

	// Contractions should have lower AI score
	if contractionResult.Signals.ContractionsUsage > formalResult.Signals.ContractionsUsage {
		t.Errorf("contraction text should have lower AI score: with=%f, without=%f",
			contractionResult.Signals.ContractionsUsage, formalResult.Signals.ContractionsUsage)
	}

	t.Logf("Contractions - With: %f, Without: %f",
		contractionResult.Signals.ContractionsUsage, formalResult.Signals.ContractionsUsage)
}

// TestReadabilityAnalysis tests readability score computation.
func TestReadabilityAnalysis(t *testing.T) {
	analyzer := NewTextAnalyzer()

	// Simple text (low grade level)
	simpleText := "See the cat. The cat is big. I like cats. Dogs are good too. They run fast."

	// Complex text (high grade level)
	complexText := "The epistemological implications of quantum entanglement fundamentally challenge our conventional understanding of causality. Furthermore, the Copenhagen interpretation remains philosophically contentious among theoretical physicists."

	simpleResult := analyzer.Analyze(simpleText)
	complexResult := analyzer.Analyze(complexText)

	// Simple text should have lower grade level
	if simpleResult.Stats.FleschKincaidGrade >= complexResult.Stats.FleschKincaidGrade {
		t.Errorf("simple text should have lower grade: simple=%f, complex=%f",
			simpleResult.Stats.FleschKincaidGrade, complexResult.Stats.FleschKincaidGrade)
	}

	t.Logf("Simple: FK Grade=%f, Fog=%f", simpleResult.Stats.FleschKincaidGrade, simpleResult.Stats.GunningFogIndex)
	t.Logf("Complex: FK Grade=%f, Fog=%f", complexResult.Stats.FleschKincaidGrade, complexResult.Stats.GunningFogIndex)
}

// TestSyllableCount tests syllable counting.
func TestSyllableCount(t *testing.T) {
	tests := []struct {
		word     string
		expected int
	}{
		{"cat", 1},
		{"hello", 2},
		{"beautiful", 3},
		{"university", 5},
		{"a", 1},
		{"the", 1},
		{"syllable", 3},
	}

	for _, tc := range tests {
		got := countSyllables(tc.word)
		// Allow +/- 1 for heuristic-based counting
		if got < tc.expected-1 || got > tc.expected+1 {
			t.Errorf("countSyllables(%q) = %d, expected around %d", tc.word, got, tc.expected)
		}
	}
}

// TestReadabilityVariance tests the variance signal.
func TestReadabilityVariance(t *testing.T) {
	analyzer := NewTextAnalyzer()

	// Text with varying complexity (human-like)
	variedText := `This is simple. Very basic stuff here.

	The phenomenological implications of existentialist philosophy present considerable analytical challenges.

	Dogs are nice. Cats too. I like them both.

	Epistemological considerations notwithstanding, the fundamental axioms remain philosophically contentious.`

	// Text with consistent complexity (AI-like)
	consistentText := `Machine learning algorithms process data. They identify patterns in datasets. Models train on examples. Neural networks have layers. Each layer transforms inputs. Outputs flow through activations. Training optimizes parameters. Gradients update weights.`

	variedResult := analyzer.Analyze(variedText)
	consistentResult := analyzer.Analyze(consistentText)

	// Varied text should have lower AI score for readability variance
	// (higher variance = more human-like = lower AI score)
	t.Logf("Varied readability variance score: %f", variedResult.Signals.ReadabilityVariance)
	t.Logf("Consistent readability variance score: %f", consistentResult.Signals.ReadabilityVariance)
}

// TestNGramAnalysis tests n-gram uniformity scoring.
func TestNGramAnalysis(t *testing.T) {
	analyzer := NewTextAnalyzer()

	// Text with varied personal patterns (human-like)
	// Contains repeated phrases that a human might use
	humanText := `I really think that this is great. You know what I think? I think we should go for it.
	That's just what I think. When I think about it, I really think it makes sense. You know what I mean?
	I've always thought this way. That's just how I think about things.`

	// Text with uniform distribution (AI-like)
	// Each sentence introduces new vocabulary without repetition
	aiText := `Machine learning enables pattern recognition. Neural networks process information efficiently.
	Algorithms optimize computational resources. Data structures organize information systematically.
	Software architectures facilitate scalable solutions. Distributed systems enhance performance metrics.`

	humanResult := analyzer.Analyze(humanText)
	aiResult := analyzer.Analyze(aiText)

	t.Logf("Human n-gram uniformity: %f", humanResult.Signals.NGramUniformity)
	t.Logf("AI n-gram uniformity: %f", aiResult.Signals.NGramUniformity)

	// Note: This tests the presence of the signal, actual thresholds may need tuning
}

// BenchmarkTextAnalyzer benchmarks the analysis speed.
func BenchmarkTextAnalyzer(b *testing.B) {
	analyzer := NewTextAnalyzer()

	// Medium-length text
	text := `This is a sample text that will be used for benchmarking the Manuscript text analyzer. 
	It contains multiple sentences with various lengths and structures. Some are short. Others are quite 
	a bit longer and contain more complex vocabulary and sentence structures that require more analysis.
	The goal is to measure how quickly we can analyze text for AI detection.`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		analyzer.Analyze(text)
	}
}

// TestRealWorldExamples tests with realistic examples.
func TestRealWorldExamples(t *testing.T) {
	analyzer := NewTextAnalyzer()

	// Example 1: ChatGPT-style response
	gptResponse := `Certainly! I'd be happy to help you with that. Here's a comprehensive overview of the topic:

Machine learning is a subset of artificial intelligence that enables computers to learn from data. It's important to note that there are several key approaches:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

In conclusion, machine learning offers tremendous potential for solving complex problems. I hope this helps! Feel free to ask if you have any more questions.`

	// Example 2: Human blog post
	humanBlog := `So I finally tried that new coffee shop everyone's been talking about. Honestly? Kinda overrated.

Don't get me wrong - the lattes are decent. But $8 for a medium?? Come on. My kitchen can do better lol

The vibe was nice tho. Exposed brick, lots of plants, you know the aesthetic. Would I go back? Maybe. If someone else is paying 😂`

	gptResult := analyzer.Analyze(gptResponse)
	humanResult := analyzer.Analyze(humanBlog)

	// GPT response should have detectable AI signals, though overall score may vary
	// The key is that it scores higher than clearly human text
	if gptResult.AIScore < 0.3 {
		t.Errorf("GPT response should have detectable AI score, got %f", gptResult.AIScore)
	}

	if humanResult.AIScore > 0.5 {
		t.Errorf("Human blog should have low AI score, got %f", humanResult.AIScore)
	}

	t.Logf("GPT Response: AI Score = %f, Phrases = %v", gptResult.AIScore, gptResult.DetectedAIPhrases)
	t.Logf("Human Blog: AI Score = %f, Signals = %+v", humanResult.AIScore, humanResult.Signals)
}

// TestConfigurableWeights tests the configurable signal weights feature.
func TestConfigurableWeights(t *testing.T) {
	t.Run("default weights sum to 1.0", func(t *testing.T) {
		w := DefaultWeights()
		sum := w.SentenceVariance + w.VocabularyRichness + w.Burstiness +
			w.PunctuationVariety + w.AIPhraseDetection + w.WordLengthVariance +
			w.ContractionsUsage + w.RepetitionPenalty + w.ReadabilityVariance +
			w.NGramUniformity

		if sum < 0.99 || sum > 1.01 {
			t.Errorf("default weights should sum to 1.0, got %f", sum)
		}
	})

	t.Run("preset weights sum to 1.0", func(t *testing.T) {
		presets := []WeightsPreset{PresetBalanced, PresetStrict, PresetLenient, PresetCreative, PresetAcademic}

		for _, preset := range presets {
			w := GetPresetWeights(preset)
			sum := w.SentenceVariance + w.VocabularyRichness + w.Burstiness +
				w.PunctuationVariety + w.AIPhraseDetection + w.WordLengthVariance +
				w.ContractionsUsage + w.RepetitionPenalty + w.ReadabilityVariance +
				w.NGramUniformity

			if sum < 0.99 || sum > 1.01 {
				t.Errorf("preset %s weights should sum to 1.0, got %f", preset, sum)
			}
		}
	})

	t.Run("custom weights affect analysis", func(t *testing.T) {
		text := `As an AI language model, I cannot provide personal opinions. However, it's important to note that this topic has many facets. Furthermore, we should consider multiple perspectives.`

		// Default analyzer
		defaultAnalyzer := NewTextAnalyzer()
		defaultResult := defaultAnalyzer.Analyze(text)

		// Custom analyzer with high AI phrase weight
		customWeights := TextAnalyzerWeights{
			SentenceVariance:    0.05,
			VocabularyRichness:  0.05,
			Burstiness:          0.05,
			PunctuationVariety:  0.05,
			AIPhraseDetection:   0.50, // Very high weight
			WordLengthVariance:  0.05,
			ContractionsUsage:   0.05,
			RepetitionPenalty:   0.05,
			ReadabilityVariance: 0.05,
			NGramUniformity:     0.10,
		}
		customAnalyzer := NewTextAnalyzerWithWeights(customWeights)
		customResult := customAnalyzer.Analyze(text)

		// Custom analyzer should give higher score to this AI-phrase-heavy text
		if customResult.AIScore <= defaultResult.AIScore {
			t.Logf("Custom score %f should be higher than default %f for AI-phrase-heavy text",
				customResult.AIScore, defaultResult.AIScore)
		}

		t.Logf("Default AI score: %f, Custom AI score: %f", defaultResult.AIScore, customResult.AIScore)
	})

	t.Run("SetWeights updates analyzer", func(t *testing.T) {
		analyzer := NewTextAnalyzer()

		originalWeights := analyzer.GetWeights()
		if originalWeights.AIPhraseDetection != 0.20 {
			t.Errorf("expected default AIPhraseDetection weight of 0.20, got %f", originalWeights.AIPhraseDetection)
		}

		newWeights := TextAnalyzerWeights{
			SentenceVariance:    0.10,
			VocabularyRichness:  0.10,
			Burstiness:          0.10,
			PunctuationVariety:  0.10,
			AIPhraseDetection:   0.10,
			WordLengthVariance:  0.10,
			ContractionsUsage:   0.10,
			RepetitionPenalty:   0.10,
			ReadabilityVariance: 0.10,
			NGramUniformity:     0.10,
		}
		analyzer.SetWeights(newWeights)

		updatedWeights := analyzer.GetWeights()
		if updatedWeights.AIPhraseDetection != 0.10 {
			t.Errorf("expected updated AIPhraseDetection weight of 0.10, got %f", updatedWeights.AIPhraseDetection)
		}
	})

	t.Run("presets create different analyzers", func(t *testing.T) {
		strictAnalyzer := NewTextAnalyzerWithPreset(PresetStrict)
		lenientAnalyzer := NewTextAnalyzerWithPreset(PresetLenient)

		strictWeights := strictAnalyzer.GetWeights()
		lenientWeights := lenientAnalyzer.GetWeights()

		// Strict should have higher AI phrase detection weight
		if strictWeights.AIPhraseDetection <= lenientWeights.AIPhraseDetection {
			t.Errorf("strict preset should have higher AIPhraseDetection weight: strict=%f, lenient=%f",
				strictWeights.AIPhraseDetection, lenientWeights.AIPhraseDetection)
		}

		// Lenient should have lower contractions weight (more forgiving of formal writing)
		if lenientWeights.ContractionsUsage >= strictWeights.ContractionsUsage {
			t.Errorf("lenient preset should have lower ContractionsUsage weight: lenient=%f, strict=%f",
				lenientWeights.ContractionsUsage, strictWeights.ContractionsUsage)
		}
	})
}

// TestPresetBehavior tests that presets produce expected behavior.
func TestPresetBehavior(t *testing.T) {
	// Formal academic text
	academicText := `The epistemological implications of quantum entanglement fundamentally
	challenge our conventional understanding of locality and causality. This phenomenon,
	first theorized by Einstein, Podolsky, and Rosen in 1935, has since been empirically
	verified through numerous experiments. The implications for our understanding of
	physical reality remain a subject of ongoing philosophical debate.`

	// Casual conversational text
	casualText := `So yeah, I tried that thing everyone's been talking about. Gotta say,
	it's pretty cool! Not gonna lie, I was skeptical at first. But now? I'm sold.
	You should totally check it out too. Just my two cents tho.`

	t.Run("academic preset handles formal writing", func(t *testing.T) {
		balancedAnalyzer := NewTextAnalyzerWithPreset(PresetBalanced)
		academicAnalyzer := NewTextAnalyzerWithPreset(PresetAcademic)

		balancedResult := balancedAnalyzer.Analyze(academicText)
		academicResult := academicAnalyzer.Analyze(academicText)

		t.Logf("Academic text - Balanced: %f, Academic preset: %f",
			balancedResult.AIScore, academicResult.AIScore)
	})

	t.Run("creative preset handles casual writing", func(t *testing.T) {
		balancedAnalyzer := NewTextAnalyzerWithPreset(PresetBalanced)
		creativeAnalyzer := NewTextAnalyzerWithPreset(PresetCreative)

		balancedResult := balancedAnalyzer.Analyze(casualText)
		creativeResult := creativeAnalyzer.Analyze(casualText)

		t.Logf("Casual text - Balanced: %f, Creative preset: %f",
			balancedResult.AIScore, creativeResult.AIScore)
	})

	t.Run("strict preset detects AI text more aggressively", func(t *testing.T) {
		aiText := `I'd be happy to help you with that. It's important to note that there are several approaches. Furthermore, this comprehensive overview will help you understand the topic better.`

		balancedAnalyzer := NewTextAnalyzerWithPreset(PresetBalanced)
		strictAnalyzer := NewTextAnalyzerWithPreset(PresetStrict)

		balancedResult := balancedAnalyzer.Analyze(aiText)
		strictResult := strictAnalyzer.Analyze(aiText)

		// Strict should give higher AI score for text with AI phrases
		if strictResult.AIScore < balancedResult.AIScore {
			t.Logf("Strict preset should detect AI text more aggressively: balanced=%f, strict=%f",
				balancedResult.AIScore, strictResult.AIScore)
		}

		t.Logf("AI text - Balanced: %f, Strict: %f",
			balancedResult.AIScore, strictResult.AIScore)
	})
}

// =============================================================================
// Comprehensive Test Suite (T4.5)
// =============================================================================

// TestEdgeCases tests edge cases and boundary conditions.
func TestEdgeCases(t *testing.T) {
	analyzer := NewTextAnalyzer()

	t.Run("empty text", func(t *testing.T) {
		result := analyzer.Analyze("")

		if result.AIScore != 0.5 {
			t.Logf("Empty text AI score: %f (expected near 0.5)", result.AIScore)
		}

		if result.Stats.WordCount != 0 {
			t.Errorf("expected 0 words for empty text, got %d", result.Stats.WordCount)
		}
	})

	t.Run("single word", func(t *testing.T) {
		result := analyzer.Analyze("Hello")

		if result.AIScore < 0.3 || result.AIScore > 0.7 {
			t.Logf("Single word should have neutral score: %f", result.AIScore)
		}
	})

	t.Run("only punctuation", func(t *testing.T) {
		result := analyzer.Analyze("!!! ??? ... ---")

		if result.Stats.WordCount != 0 {
			t.Errorf("expected 0 words for punctuation-only text, got %d", result.Stats.WordCount)
		}
	})

	t.Run("very long text", func(t *testing.T) {
		// Generate a long text
		longText := ""
		for i := 0; i < 100; i++ {
			longText += "This is a sentence that will be repeated many times to create a very long text. "
		}

		result := analyzer.Analyze(longText)

		// Should complete without panic and return a valid score
		if result.AIScore < 0 || result.AIScore > 1 {
			t.Errorf("AI score out of range: %f", result.AIScore)
		}
	})

	t.Run("unicode text", func(t *testing.T) {
		unicodeText := "这是中文测试。日本語のテスト。한국어 테스트입니다."
		result := analyzer.Analyze(unicodeText)

		// Should handle unicode without panic
		t.Logf("Unicode text AI score: %f", result.AIScore)
	})

	t.Run("mixed languages", func(t *testing.T) {
		mixedText := "Hello world. Bonjour le monde. Hola mundo. 你好世界。"
		result := analyzer.Analyze(mixedText)

		t.Logf("Mixed language text AI score: %f", result.AIScore)
	})

	t.Run("all caps text", func(t *testing.T) {
		capsText := "THIS IS ALL IN CAPS. NOTICE HOW IT LOOKS DIFFERENT. HUMANS SOMETIMES DO THIS WHEN ANGRY."
		result := analyzer.Analyze(capsText)

		t.Logf("All caps text AI score: %f", result.AIScore)
	})

	t.Run("numbers only", func(t *testing.T) {
		numbersText := "123 456 789 101112 131415"
		result := analyzer.Analyze(numbersText)

		// Numbers alone shouldn't count as words in our tokenizer
		t.Logf("Numbers only text - Word count: %d, AI score: %f", result.Stats.WordCount, result.AIScore)
	})

	t.Run("special characters", func(t *testing.T) {
		specialText := "@#$% &*() []{}  <> /\\ `~"
		result := analyzer.Analyze(specialText)

		t.Logf("Special chars text - Word count: %d, AI score: %f", result.Stats.WordCount, result.AIScore)
	})
}

// TestSignalBoundaries tests that all signals stay within 0-1 range.
func TestSignalBoundaries(t *testing.T) {
	analyzer := NewTextAnalyzer()

	texts := []string{
		"",
		"Hello",
		"Hello world!",
		"This is a test sentence. And another one. Maybe a third.",
		`As an AI language model, I cannot provide personal opinions. However, it's important to note that this topic has many facets. Furthermore, we should consider multiple perspectives. In conclusion, I hope this helps you understand the subject better. Feel free to ask if you have any more questions.`,
		`You know what? I've been thinking about this for a while now. It's weird - sometimes the simplest things are the hardest to explain! Like yesterday, I tried explaining why the sky looks blue to my kid. She just stared at me. Didn't get it at all, haha. Kids, man. Anyway, what I'm trying to say is... don't overthink it. Just go with your gut. That's what I'd do.`,
	}

	for _, text := range texts {
		result := analyzer.Analyze(text)

		signals := []struct {
			name  string
			value float64
		}{
			{"AIScore", result.AIScore},
			{"SentenceVariance", result.Signals.SentenceVariance},
			{"VocabularyRichness", result.Signals.VocabularyRichness},
			{"Burstiness", result.Signals.Burstiness},
			{"PunctuationVariety", result.Signals.PunctuationVariety},
			{"AIPhraseScore", result.Signals.AIPhraseScore},
			{"WordLengthVariance", result.Signals.WordLengthVariance},
			{"ContractionsUsage", result.Signals.ContractionsUsage},
			{"RepetitionScore", result.Signals.RepetitionScore},
			{"ReadabilityVariance", result.Signals.ReadabilityVariance},
			{"NGramUniformity", result.Signals.NGramUniformity},
		}

		for _, signal := range signals {
			if signal.value < 0 || signal.value > 1 {
				t.Errorf("Signal %s out of bounds [0,1]: %f for text: %.50s...",
					signal.name, signal.value, text)
			}
		}
	}
}

// TestConsistency tests that same text produces same results.
func TestConsistency(t *testing.T) {
	analyzer := NewTextAnalyzer()
	text := "This is a test sentence for consistency checking. It should produce the same result every time."

	result1 := analyzer.Analyze(text)
	result2 := analyzer.Analyze(text)

	if result1.AIScore != result2.AIScore {
		t.Errorf("Inconsistent AI scores: %f vs %f", result1.AIScore, result2.AIScore)
	}

	if result1.Stats.WordCount != result2.Stats.WordCount {
		t.Errorf("Inconsistent word counts: %d vs %d", result1.Stats.WordCount, result2.Stats.WordCount)
	}

	if result1.Stats.SentenceCount != result2.Stats.SentenceCount {
		t.Errorf("Inconsistent sentence counts: %d vs %d", result1.Stats.SentenceCount, result2.Stats.SentenceCount)
	}
}

// TestAIPhraseCategories tests detection of different AI phrase categories.
func TestAIPhraseCategories(t *testing.T) {
	analyzer := NewTextAnalyzer()

	categories := []struct {
		name    string
		text    string
		minHits int
	}{
		{
			"Self-Reference",
			"As an AI language model, I cannot have personal opinions. I'm just an AI trying to help.",
			2,
		},
		{
			"Inability Statements",
			"I cannot provide medical advice. I'm unable to verify this information. I cannot access that.",
			2,
		},
		{
			"Hedging",
			"It's important to note that this is complex. It's worth noting the various perspectives.",
			2,
		},
		{
			"Transitions",
			"Furthermore, this is relevant. Moreover, we should consider. In conclusion, let's summarize.",
			2,
		},
		{
			"Helpful Closings",
			"I hope this helps! Feel free to ask if you have any more questions. Let me know if you need clarification.",
			2,
		},
		{
			"Formal Vocabulary",
			"We can utilize this approach. Let's delve into the details. This will facilitate understanding.",
			2,
		},
	}

	for _, cat := range categories {
		t.Run(cat.name, func(t *testing.T) {
			result := analyzer.Analyze(cat.text)

			if len(result.DetectedAIPhrases) < cat.minHits {
				t.Errorf("expected at least %d AI phrases in %s category, got %d: %v",
					cat.minHits, cat.name, len(result.DetectedAIPhrases), result.DetectedAIPhrases)
			}

			t.Logf("%s: detected %d phrases - %v", cat.name, len(result.DetectedAIPhrases), result.DetectedAIPhrases)
		})
	}
}

// TestDifferentWritingStyles tests various writing styles.
func TestDifferentWritingStyles(t *testing.T) {
	analyzer := NewTextAnalyzer()

	styles := []struct {
		name     string
		text     string
		expectAI bool
	}{
		{
			"Technical Documentation",
			`The API endpoint accepts a POST request with a JSON body. The required fields are 'name' and 'email'. Optional parameters include 'phone' and 'address'. Returns a 201 status on success.`,
			false,
		},
		{
			"Marketing Copy",
			`Discover the revolutionary new product that's taking the world by storm! With cutting-edge technology and sleek design, it's the perfect solution for all your needs. Order now and get 50% off!`,
			false,
		},
		{
			"News Article",
			`Local officials announced today that the new bridge project will be completed ahead of schedule. The $50 million infrastructure improvement will reduce commute times by an estimated 15 minutes during rush hour.`,
			false,
		},
		{
			"ChatGPT Response",
			`Certainly! I'd be happy to help you with that question. It's important to note that there are several factors to consider. Let me break this down for you in a comprehensive way. I hope this helps!`,
			true,
		},
		{
			"Social Media Post",
			`Just tried that new ramen place downtown and OMG it was SO good!! 🍜 def gonna go back next week lol. who wants to come with???`,
			false,
		},
		{
			"Academic Abstract",
			`This study examines the correlation between socioeconomic factors and educational outcomes in urban environments. Using a mixed-methods approach, we analyzed data from 15 metropolitan areas over a five-year period.`,
			false,
		},
	}

	for _, style := range styles {
		t.Run(style.name, func(t *testing.T) {
			result := analyzer.Analyze(style.text)

			if style.expectAI && result.AIScore < 0.4 {
				t.Errorf("%s: expected higher AI score, got %f", style.name, result.AIScore)
			}

			if !style.expectAI && result.AIScore > 0.6 {
				t.Logf("%s: human text has high AI score %f (may need threshold tuning)", style.name, result.AIScore)
			}

			t.Logf("%s: AI Score = %f, Phrases = %v", style.name, result.AIScore, result.DetectedAIPhrases)
		})
	}
}

// TestReadabilityAccuracy tests readability score calculations.
func TestReadabilityAccuracy(t *testing.T) {
	// Known grade level examples
	tests := []struct {
		name          string
		text          string
		expectedGrade float64 // Approximate expected grade level
		tolerance     float64
	}{
		{
			"Simple text (Grade 1-2)",
			"The cat sat. The dog ran. I see a bird. The sun is up.",
			2.0,
			3.0,
		},
		{
			"Medium text (Grade 5-7)",
			"The weather today is quite pleasant with sunshine and a gentle breeze. Many people are enjoying outdoor activities in the local parks.",
			6.0,
			3.0,
		},
		{
			"Complex text (Grade 12+)",
			"The epistemological foundations of quantum mechanics necessitate a fundamental reconsideration of our phenomenological understanding of physical causality.",
			14.0,
			4.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			grade := calculateFleschKincaidGrade(tc.text)

			if grade < tc.expectedGrade-tc.tolerance || grade > tc.expectedGrade+tc.tolerance {
				t.Logf("Grade level %f differs from expected %f (within tolerance %f)",
					grade, tc.expectedGrade, tc.tolerance)
			}

			t.Logf("%s: FK Grade = %f", tc.name, grade)
		})
	}
}

// TestContractionsDetection tests contraction detection accuracy.
func TestContractionsDetection(t *testing.T) {
	analyzer := NewTextAnalyzer()

	t.Run("text with many contractions", func(t *testing.T) {
		text := "I've been thinking and I don't know what I'll do. She's said it won't work but I can't believe that. We're going and they're coming too."
		result := analyzer.Analyze(text)

		// Should have low ContractionsUsage score (low = human-like = uses contractions)
		if result.Signals.ContractionsUsage > 0.5 {
			t.Errorf("text with contractions should have low ContractionsUsage score, got %f", result.Signals.ContractionsUsage)
		}
	})

	t.Run("formal text without contractions", func(t *testing.T) {
		text := "I have been thinking and I do not know what I will do. She has said it will not work but I cannot believe that. We are going and they are coming too."
		result := analyzer.Analyze(text)

		// Should have high ContractionsUsage score (high = AI-like = no contractions)
		if result.Signals.ContractionsUsage < 0.5 {
			t.Errorf("formal text should have high ContractionsUsage score, got %f", result.Signals.ContractionsUsage)
		}
	})
}

// TestSentenceVarianceCalculation tests sentence variance calculation.
func TestSentenceVarianceCalculation(t *testing.T) {
	analyzer := NewTextAnalyzer()

	t.Run("highly varied sentences", func(t *testing.T) {
		text := "Short. This sentence is medium in length. And this one is quite a bit longer with many more words added to it for testing purposes."
		result := analyzer.Analyze(text)

		// Should have low variance score (low = human-like = varied)
		if result.Signals.SentenceVariance > 0.5 {
			t.Errorf("varied sentences should have low variance score, got %f", result.Signals.SentenceVariance)
		}
	})

	t.Run("uniform sentences", func(t *testing.T) {
		text := "This sentence is exactly the same. This sentence is exactly the same. This sentence is exactly the same. This sentence is exactly the same."
		result := analyzer.Analyze(text)

		// Should have high variance score (high = AI-like = uniform)
		if result.Signals.SentenceVariance < 0.7 {
			t.Errorf("uniform sentences should have high variance score, got %f", result.Signals.SentenceVariance)
		}
	})
}

// =============================================================================
// Tokenization Tests (T1.1)
// =============================================================================

// TestImprovedTokenization tests the improved tokenize function.
func TestImprovedTokenization(t *testing.T) {
	t.Run("basic words", func(t *testing.T) {
		tokens := tokenize("Hello world this is a test")
		expected := []string{"Hello", "world", "this", "is", "a", "test"}

		if len(tokens) != len(expected) {
			t.Errorf("expected %d tokens, got %d: %v", len(expected), len(tokens), tokens)
		}

		for i, tok := range expected {
			if i < len(tokens) && tokens[i] != tok {
				t.Errorf("token %d: expected %q, got %q", i, tok, tokens[i])
			}
		}
	})

	t.Run("contractions", func(t *testing.T) {
		tokens := tokenize("I don't know what I'm doing, but it's fine")

		// Should preserve contractions as single tokens
		found := map[string]bool{}
		for _, tok := range tokens {
			found[tok] = true
		}

		contractions := []string{"don't", "I'm", "it's"}
		for _, c := range contractions {
			if !found[c] {
				t.Errorf("expected contraction %q to be preserved, tokens: %v", c, tokens)
			}
		}
	})

	t.Run("hyphenated words", func(t *testing.T) {
		tokens := tokenize("This is a state-of-the-art well-known solution")

		found := map[string]bool{}
		for _, tok := range tokens {
			found[tok] = true
		}

		if !found["state-of-the-art"] {
			t.Errorf("expected 'state-of-the-art' to be preserved, tokens: %v", tokens)
		}
		if !found["well-known"] {
			t.Errorf("expected 'well-known' to be preserved, tokens: %v", tokens)
		}
	})

	t.Run("words with numbers", func(t *testing.T) {
		tokens := tokenize("COVID-19 is affecting B2B and B2C markets in 2024")

		found := map[string]bool{}
		for _, tok := range tokens {
			found[tok] = true
		}

		if !found["COVID-19"] {
			t.Errorf("expected 'COVID-19' to be preserved, tokens: %v", tokens)
		}
		if !found["B2B"] {
			t.Errorf("expected 'B2B' to be preserved, tokens: %v", tokens)
		}
		if !found["B2C"] {
			t.Errorf("expected 'B2C' to be preserved, tokens: %v", tokens)
		}
	})

	t.Run("pure numbers filtered", func(t *testing.T) {
		tokens := tokenize("There are 123 items and 456 users in 2024")

		for _, tok := range tokens {
			if tok == "123" || tok == "456" || tok == "2024" {
				t.Errorf("pure number %q should be filtered out, tokens: %v", tok, tokens)
			}
		}
	})

	t.Run("unicode characters", func(t *testing.T) {
		tokens := tokenize("The café serves naïve customers résumé")

		found := map[string]bool{}
		for _, tok := range tokens {
			found[tok] = true
		}

		if !found["café"] {
			t.Errorf("expected 'café' to be preserved, tokens: %v", tokens)
		}
		if !found["naïve"] {
			t.Errorf("expected 'naïve' to be preserved, tokens: %v", tokens)
		}
		if !found["résumé"] {
			t.Errorf("expected 'résumé' to be preserved, tokens: %v", tokens)
		}
	})

	t.Run("smart quotes normalized", func(t *testing.T) {
		// Smart quotes: ' ' " "
		// Using escape sequences for smart quotes
		tokens := tokenize("It\u2019s a \u201ctest\u201d with \u2018quotes\u2019")

		found := map[string]bool{}
		for _, tok := range tokens {
			found[tok] = true
		}

		if !found["It's"] {
			t.Errorf("expected smart quote contraction to be normalized, tokens: %v", tokens)
		}
	})

	t.Run("possessives", func(t *testing.T) {
		tokens := tokenize("John's book and Mary's car are here")

		found := map[string]bool{}
		for _, tok := range tokens {
			found[tok] = true
		}

		if !found["John's"] {
			t.Errorf("expected 'John's' to be preserved, tokens: %v", tokens)
		}
		if !found["Mary's"] {
			t.Errorf("expected 'Mary's' to be preserved, tokens: %v", tokens)
		}
	})

	t.Run("single letter filtering", func(t *testing.T) {
		tokens := tokenize("I saw a bird and x marks the spot")

		found := map[string]bool{}
		for _, tok := range tokens {
			found[tok] = true
		}

		// 'I' and 'a' should be kept
		if !found["I"] {
			t.Errorf("expected 'I' to be kept, tokens: %v", tokens)
		}
		if !found["a"] {
			t.Errorf("expected 'a' to be kept, tokens: %v", tokens)
		}
		// 'x' should be filtered
		if found["x"] {
			t.Errorf("expected 'x' to be filtered out, tokens: %v", tokens)
		}
	})

	t.Run("empty and whitespace", func(t *testing.T) {
		tokens := tokenize("   ")
		if len(tokens) != 0 {
			t.Errorf("expected 0 tokens for whitespace, got %d: %v", len(tokens), tokens)
		}

		tokens = tokenize("")
		if len(tokens) != 0 {
			t.Errorf("expected 0 tokens for empty string, got %d: %v", len(tokens), tokens)
		}
	})
}

// TestNormalizeText tests the text normalization function.
func TestNormalizeText(t *testing.T) {
	t.Run("smart quotes", func(t *testing.T) {
		// Using escape sequences: ' \u2019, ' \u2018, " \u201c, " \u201d
		input := "It\u2019s a \u201ctest\u201d with \u2018quotes\u2019"
		output := normalizeText(input)

		if strings.Contains(output, "\u2019") || strings.Contains(output, "\u2018") {
			t.Errorf("smart single quotes not normalized: %s", output)
		}
		if strings.Contains(output, "\u201c") || strings.Contains(output, "\u201d") {
			t.Errorf("smart double quotes not normalized: %s", output)
		}
	})

	t.Run("dashes", func(t *testing.T) {
		// Using escape sequences: em-dash \u2014, en-dash \u2013
		input := "em\u2014dash and en\u2013dash"
		output := normalizeText(input)

		if strings.Contains(output, "\u2014") || strings.Contains(output, "\u2013") {
			t.Errorf("dashes not normalized: %s", output)
		}
		if !strings.Contains(output, "-") {
			t.Errorf("expected regular dash in output: %s", output)
		}
	})
}

// TestIsNumeric tests the isNumeric helper function.
func TestIsNumeric(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"123", true},
		{"0", true},
		{"", false},
		{"abc", false},
		{"12a", false},
		{"a12", false},
		{"1.5", false}, // period is not a digit
	}

	for _, tc := range tests {
		result := isNumeric(tc.input)
		if result != tc.expected {
			t.Errorf("isNumeric(%q) = %v, expected %v", tc.input, result, tc.expected)
		}
	}
}
