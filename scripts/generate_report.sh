#!/bin/bash

# Manuscript Benchmark Report Generator
# Generates a summary report from benchmark results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_DIR/benchmark"
RESULTS_DIR="$BENCHMARK_DIR/results"
REPORT_FILE="$BENCHMARK_DIR/BENCHMARK_RESULTS.md"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Generating benchmark report...${NC}"

# Get the most recent results for each content type
get_latest_results() {
    local content_type="$1"
    ls -t "$RESULTS_DIR"/${content_type}_*.csv 2>/dev/null | head -1
}

# Calculate metrics from CSV file
calculate_metrics() {
    local csv_file="$1"

    if [ ! -f "$csv_file" ]; then
        echo "N/A,N/A,N/A,N/A,0"
        return
    fi

    # Skip header and calculate
    local total=0
    local correct=0
    local fp=0
    local fn=0

    while IFS=',' read -r filename expected result confidence duration; do
        [ "$filename" = "filename" ] && continue
        ((total++)) || true

        case "$result" in
            correct) ((correct++)) || true ;;
            fp) ((fp++)) || true ;;
            fn) ((fn++)) || true ;;
        esac
    done < "$csv_file"

    if [ $total -eq 0 ]; then
        echo "N/A,N/A,N/A,N/A,0"
        return
    fi

    local accuracy
    accuracy=$(echo "scale=1; $correct * 100 / $total" | bc)

    local precision="N/A"
    local recall="N/A"
    local f1="N/A"

    local tp=$((correct))  # Simplified for this context

    if [ $((tp + fp)) -gt 0 ]; then
        precision=$(echo "scale=1; $tp * 100 / ($tp + $fp)" | bc)
    fi

    if [ $((tp + fn)) -gt 0 ]; then
        recall=$(echo "scale=1; $tp * 100 / ($tp + $fn)" | bc)
    fi

    if [ "$precision" != "N/A" ] && [ "$recall" != "N/A" ]; then
        local p_val
        p_val=$(echo "$precision" | bc)
        local r_val
        r_val=$(echo "$recall" | bc)
        if [ $(echo "$p_val + $r_val > 0" | bc) -eq 1 ]; then
            f1=$(echo "scale=1; 2 * $p_val * $r_val / ($p_val + $r_val)" | bc)
        fi
    fi

    echo "$accuracy,$precision,$recall,$f1,$total"
}

# Generate the report
cat > "$REPORT_FILE" << 'EOF'
# Manuscript Benchmark Results

> Auto-generated benchmark report

**Generated:**
EOF

echo "$(date '+%Y-%m-%d %H:%M:%S UTC')" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'

---

## Summary

| Content Type | Accuracy | Precision | Recall | F1 Score | Samples |
|-------------|----------|-----------|--------|----------|---------|
EOF

# Add results for each content type
for content_type in text image audio video; do
    results_file=$(get_latest_results "$content_type")
    metrics=$(calculate_metrics "$results_file")

    accuracy=$(echo "$metrics" | cut -d',' -f1)
    precision=$(echo "$metrics" | cut -d',' -f2)
    recall=$(echo "$metrics" | cut -d',' -f3)
    f1=$(echo "$metrics" | cut -d',' -f4)
    samples=$(echo "$metrics" | cut -d',' -f5)

    echo "| ${content_type^} | ${accuracy}% | ${precision}% | ${recall}% | ${f1}% | $samples |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << 'EOF'

---

## Detailed Results

EOF

# Add detailed results for each content type
for content_type in text image audio video; do
    results_file=$(get_latest_results "$content_type")

    echo "### ${content_type^} Detection" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    if [ -f "$results_file" ]; then
        echo "**Results file:** \`$(basename "$results_file")\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"

        # Count results
        local total=0
        local correct=0
        local fp=0
        local fn=0
        local errors=0

        while IFS=',' read -r filename expected result confidence duration; do
            [ "$filename" = "filename" ] && continue
            ((total++)) || true
            case "$result" in
                correct) ((correct++)) || true ;;
                fp) ((fp++)) || true ;;
                fn) ((fn++)) || true ;;
                error) ((errors++)) || true ;;
            esac
        done < "$results_file"

        echo "| Metric | Value |" >> "$REPORT_FILE"
        echo "|--------|-------|" >> "$REPORT_FILE"
        echo "| Total Samples | $total |" >> "$REPORT_FILE"
        echo "| Correct Classifications | $correct |" >> "$REPORT_FILE"
        echo "| False Positives | $fp |" >> "$REPORT_FILE"
        echo "| False Negatives | $fn |" >> "$REPORT_FILE"
        echo "| Errors | $errors |" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"

        # Show sample of results
        echo "<details>" >> "$REPORT_FILE"
        echo "<summary>Sample Results</summary>" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "\`\`\`csv" >> "$REPORT_FILE"
        head -20 "$results_file" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        echo "</details>" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    else
        echo "*No results available. Run \`make benchmark-${content_type}\` to generate.*" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << 'EOF'
---

## Methodology

This benchmark evaluates Manuscript's detection capabilities using curated datasets of human-created and AI-generated content.

### Evaluation Metrics

- **Accuracy**: Overall percentage of correct classifications
- **Precision**: Proportion of AI detections that were actually AI-generated
- **Recall**: Proportion of AI-generated content that was correctly identified
- **F1 Score**: Harmonic mean of precision and recall

### Dataset Sources

See [DATASET_SOURCES.md](DATASET_SOURCES.md) for detailed information about the datasets used.

---

## Running the Benchmark

```bash
# Download datasets
make download-benchmark-data

# Start the API
make run

# Run benchmarks (in another terminal)
make benchmark-all

# Generate this report
make benchmark-report
```

---

*For more details, see the full [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)*
EOF

echo -e "${GREEN}Report generated: $REPORT_FILE${NC}"
