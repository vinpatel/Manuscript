---
title: Text Detection
description: How Manuscript detects AI-generated text
---

# Text Detection

Manuscript uses statistical analysis to differentiate human-written text from AI-generated content.

## Detection Signals

### Sentence Length Variance

Human writing naturally varies in sentence length. AI tends to produce more uniform sentences.

| Pattern | Human | AI |
|---------|-------|-----|
| Short sentences | Common | Rare |
| Long sentences | Common | Common |
| Variation | High | Low |

**Weight:** 0.15

### Vocabulary Richness

Humans use personal vocabulary, including rare words, slang, and domain-specific terms. AI uses statistically "safe" common words.

**Weight:** 0.20

### Contraction Usage

Humans naturally use contractions ("don't", "I'm", "we'll"). AI often uses formal forms ("do not", "I am").

| Form | Human | AI |
|------|-------|-----|
| "don't" | Very common | Rare |
| "do not" | Rare (formal) | Common |
| "I'm" | Very common | Rare |
| "I am" | Rare (formal) | Common |

**Weight:** 0.10

### AI Phrase Detection

Certain phrases are strong indicators of AI generation:

- "As an AI..."
- "It's important to note..."
- "I don't have personal experiences..."
- "Let me break this down..."
- "That's a great question..."
- "In summary..."
- "Additionally..."

Manuscript maintains a database of 35+ such patterns.

**Weight:** 0.20

### Hedging Language

AI often uses excessive qualifiers and hedging:

- "It's possible that..."
- "Generally speaking..."
- "In most cases..."
- "It depends on..."

**Weight:** 0.10

### Punctuation Variety

Humans use diverse punctuation (!?;:—...). AI primarily uses periods and commas.

**Weight:** 0.10

### Repetition Patterns

AI tends to repeat structural patterns mechanically. Humans have organic callbacks.

**Weight:** 0.10

## Example Analysis

**Human-written text:**

```
I've been thinking about this for days. Can't shake the feeling that
something's off—you know what I mean? The data just... doesn't add up.
```

**Signals detected:**
- High sentence variance ✓
- Contractions used ("I've", "Can't") ✓
- Diverse punctuation (?, —, ...) ✓
- Natural hedging ✓

**Verdict:** Human (confidence: 0.92)

---

**AI-generated text:**

```
It is important to note that this analysis provides valuable insights.
Additionally, the data suggests several key findings. In summary, the
results demonstrate significant patterns that warrant further investigation.
```

**Signals detected:**
- Low sentence variance ✓
- No contractions ✓
- AI phrases detected ("It is important to note", "Additionally", "In summary") ✓
- Formulaic structure ✓

**Verdict:** AI (confidence: 0.94)

## API Usage

Submit text via `POST /verify` with a JSON body. See the [API Reference](/manuscript/docs/api/endpoints/) for full request/response details, batch processing, and SDK examples.

## Accuracy Benchmarks

| Metric | Value |
|--------|-------|
| Accuracy | 90.00% |
| Precision | 100.00% |
| Recall | 80.00% |
| F1 Score | 88.89% |

Tested on 100 samples (50 human, 50 AI) including GPT-4, Claude, Gemini, and Llama-3 content.

## Limitations

- **Short texts** (<100 words) have lower accuracy
- **Heavily edited AI content** may evade detection
- **Domain-specific jargon** can affect vocabulary analysis
- **Non-English** text support is limited

## Best Practices

1. **Minimum length:** Provide at least 100 words for reliable detection
2. **Original content:** Detection works best on unedited content
3. **Confidence threshold:** Consider results <0.7 as uncertain
4. **Multiple samples:** For important decisions, analyze multiple excerpts
