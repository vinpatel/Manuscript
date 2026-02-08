#!/bin/bash

# Manuscript Full Benchmark Script
# Runs benchmarks against all sample datasets and collects metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$BENCHMARK_DIR/datasets"
RESULTS_DIR="$BENCHMARK_DIR/results"
API_URL="${API_URL:-http://localhost:8080}"

mkdir -p "$RESULTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Manuscript Benchmark Suite"
echo "=========================================="
echo "API URL: $API_URL"
echo "Dataset Dir: $DATASET_DIR"
echo "Results Dir: $RESULTS_DIR"
echo ""

# Check API health
if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: API not available at $API_URL${NC}"
    exit 1
fi
echo -e "${GREEN}API is healthy${NC}"
echo ""

# Initialize results
TEXT_RESULTS="$RESULTS_DIR/text_results.csv"
IMAGE_RESULTS="$RESULTS_DIR/image_results.csv"
AUDIO_RESULTS="$RESULTS_DIR/audio_results.csv"

# CSV headers
echo "filename,expected,predicted,confidence,human,processing_time_ms" > "$TEXT_RESULTS"
echo "filename,expected,predicted,confidence,human,processing_time_ms" > "$IMAGE_RESULTS"
echo "filename,expected,predicted,confidence,human,processing_time_ms" > "$AUDIO_RESULTS"

# Counters
TEXT_TP=0 TEXT_TN=0 TEXT_FP=0 TEXT_FN=0 TEXT_TOTAL=0
IMAGE_TP=0 IMAGE_TN=0 IMAGE_FP=0 IMAGE_FN=0 IMAGE_TOTAL=0
AUDIO_TP=0 AUDIO_TN=0 AUDIO_FP=0 AUDIO_FN=0 AUDIO_TOTAL=0

# Function to analyze text
analyze_text() {
    local file="$1"
    local expected="$2"

    # Read file content and escape for JSON
    local content=$(cat "$file" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')

    local start_time=$(python3 -c "import time; print(int(time.time()*1000))")
    local response=$(curl -s -X POST "$API_URL/verify" \
        -H "Content-Type: application/json" \
        -d "{\"text\": $content}" 2>/dev/null)
    local end_time=$(python3 -c "import time; print(int(time.time()*1000))")
    local duration=$((end_time - start_time))

    local human=$(echo "$response" | jq -r '.human // false')
    local confidence=$(echo "$response" | jq -r '.confidence // 0')
    local predicted="ai"
    if [ "$human" = "true" ]; then
        predicted="human"
    fi

    local filename=$(basename "$file")
    echo "$filename,$expected,$predicted,$confidence,$human,$duration" >> "$TEXT_RESULTS"

    # Update counters (TP = correctly identified AI, TN = correctly identified human)
    if [ "$expected" = "ai" ] && [ "$predicted" = "ai" ]; then
        ((TEXT_TP++))
    elif [ "$expected" = "human" ] && [ "$predicted" = "human" ]; then
        ((TEXT_TN++))
    elif [ "$expected" = "human" ] && [ "$predicted" = "ai" ]; then
        ((TEXT_FP++))
    elif [ "$expected" = "ai" ] && [ "$predicted" = "human" ]; then
        ((TEXT_FN++))
    fi
    ((TEXT_TOTAL++))
}

# Function to analyze image
analyze_image() {
    local file="$1"
    local expected="$2"

    local start_time=$(python3 -c "import time; print(int(time.time()*1000))")
    local response=$(curl -s -X POST "$API_URL/verify" \
        -F "file=@$file" 2>/dev/null)
    local end_time=$(python3 -c "import time; print(int(time.time()*1000))")
    local duration=$((end_time - start_time))

    local human=$(echo "$response" | jq -r '.human // false')
    local confidence=$(echo "$response" | jq -r '.confidence // 0')
    local predicted="ai"
    if [ "$human" = "true" ]; then
        predicted="human"
    fi

    local filename=$(basename "$file")
    echo "$filename,$expected,$predicted,$confidence,$human,$duration" >> "$IMAGE_RESULTS"

    # Update counters
    if [ "$expected" = "ai" ] && [ "$predicted" = "ai" ]; then
        ((IMAGE_TP++))
    elif [ "$expected" = "human" ] && [ "$predicted" = "human" ]; then
        ((IMAGE_TN++))
    elif [ "$expected" = "human" ] && [ "$predicted" = "ai" ]; then
        ((IMAGE_FP++))
    elif [ "$expected" = "ai" ] && [ "$predicted" = "human" ]; then
        ((IMAGE_FN++))
    fi
    ((IMAGE_TOTAL++))
}

# Function to analyze audio
analyze_audio() {
    local file="$1"
    local expected="$2"

    local start_time=$(python3 -c "import time; print(int(time.time()*1000))")
    local response=$(curl -s -X POST "$API_URL/verify" \
        -F "file=@$file" 2>/dev/null)
    local end_time=$(python3 -c "import time; print(int(time.time()*1000))")
    local duration=$((end_time - start_time))

    local human=$(echo "$response" | jq -r '.human // false')
    local confidence=$(echo "$response" | jq -r '.confidence // 0')
    local predicted="ai"
    if [ "$human" = "true" ]; then
        predicted="human"
    fi

    local filename=$(basename "$file")
    echo "$filename,$expected,$predicted,$confidence,$human,$duration" >> "$AUDIO_RESULTS"

    # Update counters
    if [ "$expected" = "ai" ] && [ "$predicted" = "ai" ]; then
        ((AUDIO_TP++))
    elif [ "$expected" = "human" ] && [ "$predicted" = "human" ]; then
        ((AUDIO_TN++))
    elif [ "$expected" = "human" ] && [ "$predicted" = "ai" ]; then
        ((AUDIO_FP++))
    elif [ "$expected" = "ai" ] && [ "$predicted" = "human" ]; then
        ((AUDIO_FN++))
    fi
    ((AUDIO_TOTAL++))
}

# ===================
# TEXT BENCHMARK
# ===================
echo "=========================================="
echo "Running TEXT Benchmark"
echo "=========================================="

# Human text files
if [ -d "$DATASET_DIR/text/human" ]; then
    echo "Processing human text files..."
    for file in "$DATASET_DIR/text/human"/*.txt; do
        [ -f "$file" ] || continue
        analyze_text "$file" "human"
        printf "."
    done
    echo ""
fi

# AI text files
if [ -d "$DATASET_DIR/text/ai_generated" ]; then
    echo "Processing AI-generated text files..."
    for file in "$DATASET_DIR/text/ai_generated"/*.txt; do
        [ -f "$file" ] || continue
        analyze_text "$file" "ai"
        printf "."
    done
    echo ""
fi

echo "Text Results: TP=$TEXT_TP, TN=$TEXT_TN, FP=$TEXT_FP, FN=$TEXT_FN"

# ===================
# IMAGE BENCHMARK
# ===================
echo ""
echo "=========================================="
echo "Running IMAGE Benchmark"
echo "=========================================="

# Human images
if [ -d "$DATASET_DIR/image/human" ]; then
    echo "Processing human images..."
    for file in "$DATASET_DIR/image/human"/*.jpg "$DATASET_DIR/image/human"/*.png "$DATASET_DIR/image/human"/*.jpeg; do
        [ -f "$file" ] || continue
        analyze_image "$file" "human"
        printf "."
    done
    echo ""
fi

# AI images
if [ -d "$DATASET_DIR/image/ai_generated" ]; then
    echo "Processing AI-generated images..."
    for file in "$DATASET_DIR/image/ai_generated"/*.jpg "$DATASET_DIR/image/ai_generated"/*.png "$DATASET_DIR/image/ai_generated"/*.jpeg; do
        [ -f "$file" ] || continue
        analyze_image "$file" "ai"
        printf "."
    done
    echo ""
fi

echo "Image Results: TP=$IMAGE_TP, TN=$IMAGE_TN, FP=$IMAGE_FP, FN=$IMAGE_FN"

# ===================
# AUDIO BENCHMARK
# ===================
echo ""
echo "=========================================="
echo "Running AUDIO Benchmark"
echo "=========================================="

# Human audio
if [ -d "$DATASET_DIR/audio/human" ]; then
    echo "Processing human audio files..."
    for file in "$DATASET_DIR/audio/human"/*.flac "$DATASET_DIR/audio/human"/*.wav "$DATASET_DIR/audio/human"/*.mp3; do
        [ -f "$file" ] || continue
        analyze_audio "$file" "human"
        printf "."
    done
    echo ""
fi

# AI audio
if [ -d "$DATASET_DIR/audio/ai_generated" ]; then
    echo "Processing AI-generated audio files..."
    for file in "$DATASET_DIR/audio/ai_generated"/*.flac "$DATASET_DIR/audio/ai_generated"/*.wav "$DATASET_DIR/audio/ai_generated"/*.mp3; do
        [ -f "$file" ] || continue
        analyze_audio "$file" "ai"
        printf "."
    done
    echo ""
fi

echo "Audio Results: TP=$AUDIO_TP, TN=$AUDIO_TN, FP=$AUDIO_FP, FN=$AUDIO_FN"

# ===================
# CALCULATE METRICS
# ===================
echo ""
echo "=========================================="
echo "BENCHMARK SUMMARY"
echo "=========================================="

calculate_metrics() {
    local tp=$1 tn=$2 fp=$3 fn=$4 name=$5

    local total=$((tp + tn + fp + fn))
    if [ $total -eq 0 ]; then
        echo "$name: No samples processed"
        return
    fi

    # Accuracy
    local accuracy=$(python3 -c "print(f'{($tp + $tn) / $total * 100:.2f}')")

    # Precision (for detecting AI - avoid division by zero)
    local precision="0.00"
    if [ $((tp + fp)) -gt 0 ]; then
        precision=$(python3 -c "print(f'{$tp / ($tp + $fp) * 100:.2f}')")
    fi

    # Recall (for detecting AI)
    local recall="0.00"
    if [ $((tp + fn)) -gt 0 ]; then
        recall=$(python3 -c "print(f'{$tp / ($tp + $fn) * 100:.2f}')")
    fi

    # F1 Score
    local f1="0.00"
    local p_val=$(python3 -c "print($tp / ($tp + $fp) if ($tp + $fp) > 0 else 0)")
    local r_val=$(python3 -c "print($tp / ($tp + $fn) if ($tp + $fn) > 0 else 0)")
    if [ "$(python3 -c "print(1 if ($p_val + $r_val) > 0 else 0)")" = "1" ]; then
        f1=$(python3 -c "print(f'{2 * $p_val * $r_val / ($p_val + $r_val) * 100:.2f}')")
    fi

    echo "$name Detection:"
    echo "  Samples: $total (TP=$tp, TN=$tn, FP=$fp, FN=$fn)"
    echo "  Accuracy:  $accuracy%"
    echo "  Precision: $precision%"
    echo "  Recall:    $recall%"
    echo "  F1 Score:  $f1%"
    echo ""

    # Return values for report
    echo "$name,$total,$accuracy,$precision,$recall,$f1" >> "$RESULTS_DIR/summary.csv"
}

# Create summary header
echo "content_type,samples,accuracy,precision,recall,f1" > "$RESULTS_DIR/summary.csv"

calculate_metrics $TEXT_TP $TEXT_TN $TEXT_FP $TEXT_FN "Text"
calculate_metrics $IMAGE_TP $IMAGE_TN $IMAGE_FP $IMAGE_FN "Image"
calculate_metrics $AUDIO_TP $AUDIO_TN $AUDIO_FP $AUDIO_FN "Audio"

echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo "  - text_results.csv"
echo "  - image_results.csv"
echo "  - audio_results.csv"
echo "  - summary.csv"
echo "=========================================="
