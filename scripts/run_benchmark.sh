#!/bin/bash

# Manuscript Benchmark Runner
# Runs detection benchmarks against curated datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_DIR/benchmark"
DATASETS_DIR="$BENCHMARK_DIR/datasets"
RESULTS_DIR="$BENCHMARK_DIR/results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# API configuration
API_URL="${API_URL:-http://localhost:8080}"
TIMEOUT=30

# Create results directory
mkdir -p "$RESULTS_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to check if API is running
check_api() {
    if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
        echo -e "${RED}Error: Manuscript API not running at $API_URL${NC}"
        echo "Start the API with: make run"
        exit 1
    fi
    echo -e "${GREEN}API is running at $API_URL${NC}"
}

# Function to run detection on a single file
run_detection() {
    local file_path="$1"
    local content_type="$2"
    local expected="$3"  # "human" or "ai"

    local response
    if [ "$content_type" = "text" ]; then
        # For text files, send as JSON
        local text_content
        text_content=$(cat "$file_path")
        response=$(curl -s -X POST "$API_URL/verify" \
            -H "Content-Type: application/json" \
            -d "$(jq -n --arg text "$text_content" '{text: $text}')" \
            --max-time $TIMEOUT 2>/dev/null)
    else
        # For media files, upload as multipart
        response=$(curl -s -X POST "$API_URL/verify" \
            -F "file=@$file_path" \
            --max-time $TIMEOUT 2>/dev/null)
    fi

    if [ -z "$response" ]; then
        echo "error"
        return
    fi

    # Extract result
    local human
    human=$(echo "$response" | jq -r '.human // empty' 2>/dev/null)
    local confidence
    confidence=$(echo "$response" | jq -r '.confidence // 0' 2>/dev/null)

    if [ "$human" = "true" ]; then
        if [ "$expected" = "human" ]; then
            echo "correct,$confidence"
        else
            echo "fp,$confidence"  # False positive (predicted human, was AI)
        fi
    elif [ "$human" = "false" ]; then
        if [ "$expected" = "ai" ]; then
            echo "correct,$confidence"
        else
            echo "fn,$confidence"  # False negative (predicted AI, was human)
        fi
    else
        echo "error,0"
    fi
}

# Function to run benchmark for a content type
run_benchmark() {
    local content_type="$1"
    local human_dir="$DATASETS_DIR/$content_type/human"
    local ai_dir="$DATASETS_DIR/$content_type/ai_generated"
    local results_file="$RESULTS_DIR/${content_type}_${TIMESTAMP}.csv"

    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Running $content_type benchmark${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Check if dataset exists
    if [ ! -d "$human_dir" ] || [ ! -d "$ai_dir" ]; then
        echo -e "${YELLOW}Dataset not found for $content_type${NC}"
        echo "Run 'make download-benchmark-data' first"
        return
    fi

    # Count files
    local human_count=0
    local ai_count=0

    if [ -d "$human_dir" ]; then
        human_count=$(find "$human_dir" -type f \( -name "*.txt" -o -name "*.jpg" -o -name "*.png" -o -name "*.mp3" -o -name "*.wav" -o -name "*.mp4" -o -name "*.webm" \) 2>/dev/null | wc -l | tr -d ' ')
    fi

    if [ -d "$ai_dir" ]; then
        ai_count=$(find "$ai_dir" -type f \( -name "*.txt" -o -name "*.jpg" -o -name "*.png" -o -name "*.mp3" -o -name "*.wav" -o -name "*.mp4" -o -name "*.webm" \) 2>/dev/null | wc -l | tr -d ' ')
    fi

    echo "Human samples: $human_count"
    echo "AI samples: $ai_count"

    if [ "$human_count" -eq 0 ] && [ "$ai_count" -eq 0 ]; then
        echo -e "${YELLOW}No samples found. Skipping benchmark.${NC}"
        return
    fi

    # Initialize results
    echo "filename,expected,result,confidence,processing_time_ms" > "$results_file"

    local correct=0
    local total=0
    local fp=0  # False positives
    local fn=0  # False negatives
    local errors=0

    # Test human samples
    echo -e "\n${YELLOW}Testing human samples...${NC}"
    for file in "$human_dir"/*.txt "$human_dir"/*.jpg "$human_dir"/*.jpeg "$human_dir"/*.png "$human_dir"/*.gif "$human_dir"/*.webp "$human_dir"/*.mp3 "$human_dir"/*.wav "$human_dir"/*.flac "$human_dir"/*.mp4 "$human_dir"/*.webm "$human_dir"/*.mov; do
        [ -f "$file" ] || continue
        local filename
        filename=$(basename "$file")
        local start_time
        start_time=$(python3 -c 'import time; print(int(time.time() * 1000))')

        local result
        result=$(run_detection "$file" "$content_type" "human")

        local end_time
        end_time=$(python3 -c 'import time; print(int(time.time() * 1000))')
        local duration=$((end_time - start_time))

        local status
        status=$(echo "$result" | cut -d',' -f1)
        local confidence
        confidence=$(echo "$result" | cut -d',' -f2)

        echo "$filename,human,$status,$confidence,$duration" >> "$results_file"

        case "$status" in
            correct) ((correct++)) ;;
            fn) ((fn++)) ;;
            *) ((errors++)) ;;
        esac
        ((total++))

        printf "  %-40s %s\n" "$filename" "$status"
    done

    # Test AI samples
    echo -e "\n${YELLOW}Testing AI samples...${NC}"
    for file in "$ai_dir"/*.txt "$ai_dir"/*.jpg "$ai_dir"/*.jpeg "$ai_dir"/*.png "$ai_dir"/*.gif "$ai_dir"/*.webp "$ai_dir"/*.mp3 "$ai_dir"/*.wav "$ai_dir"/*.flac "$ai_dir"/*.mp4 "$ai_dir"/*.webm "$ai_dir"/*.mov; do
        [ -f "$file" ] || continue
        local filename
        filename=$(basename "$file")
        local start_time
        start_time=$(python3 -c 'import time; print(int(time.time() * 1000))')

        local result
        result=$(run_detection "$file" "$content_type" "ai")

        local end_time
        end_time=$(python3 -c 'import time; print(int(time.time() * 1000))')
        local duration=$((end_time - start_time))

        local status
        status=$(echo "$result" | cut -d',' -f1)
        local confidence
        confidence=$(echo "$result" | cut -d',' -f2)

        echo "$filename,ai,$status,$confidence,$duration" >> "$results_file"

        case "$status" in
            correct) ((correct++)) ;;
            fp) ((fp++)) ;;
            *) ((errors++)) ;;
        esac
        ((total++))

        printf "  %-40s %s\n" "$filename" "$status"
    done

    # Calculate metrics
    echo -e "\n${BLUE}Results for $content_type:${NC}"
    echo "----------------------------------------"

    if [ $total -gt 0 ]; then
        local accuracy
        accuracy=$(echo "scale=2; $correct * 100 / $total" | bc)

        local true_positives=$((correct - fn))
        local precision=0
        local recall=0

        if [ $((true_positives + fp)) -gt 0 ]; then
            precision=$(echo "scale=2; $true_positives * 100 / ($true_positives + $fp)" | bc)
        fi

        if [ $((true_positives + fn)) -gt 0 ]; then
            recall=$(echo "scale=2; $true_positives * 100 / ($true_positives + $fn)" | bc)
        fi

        local f1=0
        if [ $(echo "$precision + $recall > 0" | bc) -eq 1 ]; then
            f1=$(echo "scale=2; 2 * $precision * $recall / ($precision + $recall)" | bc)
        fi

        echo "Total samples:    $total"
        echo "Correct:          $correct"
        echo "False positives:  $fp"
        echo "False negatives:  $fn"
        echo "Errors:           $errors"
        echo ""
        echo -e "${GREEN}Accuracy:         ${accuracy}%${NC}"
        echo "Precision:        ${precision}%"
        echo "Recall:           ${recall}%"
        echo "F1 Score:         ${f1}%"
    else
        echo "No samples tested"
    fi

    echo ""
    echo "Results saved to: $results_file"
}

# Main
case "${1:-all}" in
    text)
        check_api
        run_benchmark "text"
        ;;
    image)
        check_api
        run_benchmark "image"
        ;;
    audio)
        check_api
        run_benchmark "audio"
        ;;
    video)
        check_api
        run_benchmark "video"
        ;;
    all)
        check_api
        run_benchmark "text"
        run_benchmark "image"
        run_benchmark "audio"
        run_benchmark "video"

        echo -e "\n${BLUE}========================================${NC}"
        echo -e "${BLUE}Benchmark Complete${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo "Results saved in: $RESULTS_DIR"
        ;;
    *)
        echo "Usage: $0 [text|image|audio|video|all]"
        exit 1
        ;;
esac
