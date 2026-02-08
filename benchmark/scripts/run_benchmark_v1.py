#!/usr/bin/env python3
"""
Manuscript Benchmark Script - V1 Comparison
"""

import json
import os
import sys
import time
import csv
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

API_URL = os.environ.get("API_URL", "http://localhost:8080")
BENCHMARK_DIR = Path("/Users/vin/Development/Manuscript/benchmark")
DATASET_DIR = BENCHMARK_DIR / "datasets"
RESULTS_DIR = BENCHMARK_DIR / "results_v1"

@dataclass
class BenchmarkResult:
    filename: str
    expected: str
    predicted: str
    confidence: float
    human: bool
    processing_time_ms: int

@dataclass
class Metrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    results: List[BenchmarkResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total * 100

    @property
    def precision(self) -> float:
        if (self.tp + self.fp) == 0:
            return 0.0
        return self.tp / (self.tp + self.fp) * 100

    @property
    def recall(self) -> float:
        if (self.tp + self.fn) == 0:
            return 0.0
        return self.tp / (self.tp + self.fn) * 100

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision/100 * self.recall/100) / (self.precision/100 + self.recall/100) * 100

def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def analyze_text(file_path: Path) -> Tuple[bool, float, int]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    start_time = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_URL}/verify",
                json={"text": content},
                timeout=30
            )
            data = response.json()
            human = data.get("human", False)
            confidence = data.get("confidence", 0.0)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            print(f"\nError analyzing {file_path.name}: {e}")
            human = True
            confidence = 0.0

    time_ms = int((time.time() - start_time) * 1000)
    time.sleep(0.1)
    return human, confidence, time_ms

def analyze_file(file_path: Path) -> Tuple[bool, float, int]:
    start_time = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{API_URL}/verify",
                    files={"file": (file_path.name, f)},
                    timeout=60
                )
            data = response.json()
            human = data.get("human", False)
            confidence = data.get("confidence", 0.0)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            print(f"\nError analyzing {file_path.name}: {e}")
            human = True
            confidence = 0.0

    time_ms = int((time.time() - start_time) * 1000)
    time.sleep(0.1)
    return human, confidence, time_ms

def update_metrics(metrics: Metrics, expected: str, human: bool, confidence: float, filename: str, time_ms: int):
    predicted = "human" if human else "ai"
    result = BenchmarkResult(filename=filename, expected=expected, predicted=predicted, confidence=confidence, human=human, processing_time_ms=time_ms)
    metrics.results.append(result)

    if expected == "ai" and predicted == "ai":
        metrics.tp += 1
    elif expected == "human" and predicted == "human":
        metrics.tn += 1
    elif expected == "human" and predicted == "ai":
        metrics.fp += 1
    elif expected == "ai" and predicted == "human":
        metrics.fn += 1

def run_text_benchmark() -> Metrics:
    metrics = Metrics()
    human_dir = DATASET_DIR / "text" / "human"
    if human_dir.exists():
        files = sorted(human_dir.glob("*.txt"))
        print(f"Processing {len(files)} human text files...")
        for file in files:
            human, confidence, time_ms = analyze_text(file)
            update_metrics(metrics, "human", human, confidence, file.name, time_ms)
            print(".", end="", flush=True)
        print()

    ai_dir = DATASET_DIR / "text" / "ai_generated"
    if ai_dir.exists():
        files = sorted(ai_dir.glob("*.txt"))
        print(f"Processing {len(files)} AI-generated text files...")
        for file in files:
            human, confidence, time_ms = analyze_text(file)
            update_metrics(metrics, "ai", human, confidence, file.name, time_ms)
            print(".", end="", flush=True)
        print()
    return metrics

def run_image_benchmark() -> Metrics:
    metrics = Metrics()
    human_dir = DATASET_DIR / "image" / "human"
    if human_dir.exists():
        files = sorted(list(human_dir.glob("*.jpg")) + list(human_dir.glob("*.png")) + list(human_dir.glob("*.jpeg")))
        print(f"Processing {len(files)} human images...")
        for file in files:
            human, confidence, time_ms = analyze_file(file)
            update_metrics(metrics, "human", human, confidence, file.name, time_ms)
            print(".", end="", flush=True)
        print()

    ai_dir = DATASET_DIR / "image" / "ai_generated"
    if ai_dir.exists():
        files = sorted(list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png")) + list(ai_dir.glob("*.jpeg")))
        print(f"Processing {len(files)} AI-generated images...")
        for file in files:
            human, confidence, time_ms = analyze_file(file)
            update_metrics(metrics, "ai", human, confidence, file.name, time_ms)
            print(".", end="", flush=True)
        print()
    return metrics

def run_audio_benchmark() -> Metrics:
    metrics = Metrics()
    human_dir = DATASET_DIR / "audio" / "human"
    if human_dir.exists():
        files = sorted(list(human_dir.glob("*.flac")) + list(human_dir.glob("*.wav")) + list(human_dir.glob("*.mp3")))
        print(f"Processing {len(files)} human audio files...")
        for file in files:
            human, confidence, time_ms = analyze_file(file)
            update_metrics(metrics, "human", human, confidence, file.name, time_ms)
            print(".", end="", flush=True)
        print()

    ai_dir = DATASET_DIR / "audio" / "ai_generated"
    if ai_dir.exists():
        files = sorted(list(ai_dir.glob("*.flac")) + list(ai_dir.glob("*.wav")) + list(ai_dir.glob("*.mp3")))
        print(f"Processing {len(files)} AI-generated audio files...")
        for file in files:
            human, confidence, time_ms = analyze_file(file)
            update_metrics(metrics, "ai", human, confidence, file.name, time_ms)
            print(".", end="", flush=True)
        print()
    return metrics

def print_metrics(metrics: Metrics, name: str):
    print(f"{name} Detection:")
    print(f"  Samples: {metrics.total} (TP={metrics.tp}, TN={metrics.tn}, FP={metrics.fp}, FN={metrics.fn})")
    print(f"  Accuracy:  {metrics.accuracy:.2f}%")
    print(f"  Precision: {metrics.precision:.2f}%")
    print(f"  Recall:    {metrics.recall:.2f}%")
    print(f"  F1 Score:  {metrics.f1:.2f}%")
    print()

def main():
    print("=" * 50)
    print("Manuscript V1 (v0.1.0) Benchmark")
    print("=" * 50)
    print(f"API URL: {API_URL}")
    print()

    if not check_api_health():
        print(f"ERROR: API not available at {API_URL}")
        sys.exit(1)
    print("API is healthy\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    print("=" * 50)
    print("Running TEXT Benchmark")
    print("=" * 50)
    text_metrics = run_text_benchmark()
    all_metrics["Text"] = text_metrics
    print(f"Text Results: TP={text_metrics.tp}, TN={text_metrics.tn}, FP={text_metrics.fp}, FN={text_metrics.fn}")

    print("\n" + "=" * 50)
    print("Running IMAGE Benchmark")
    print("=" * 50)
    image_metrics = run_image_benchmark()
    all_metrics["Image"] = image_metrics
    print(f"Image Results: TP={image_metrics.tp}, TN={image_metrics.tn}, FP={image_metrics.fp}, FN={image_metrics.fn}")

    print("\n" + "=" * 50)
    print("Running AUDIO Benchmark")
    print("=" * 50)
    audio_metrics = run_audio_benchmark()
    all_metrics["Audio"] = audio_metrics
    print(f"Audio Results: TP={audio_metrics.tp}, TN={audio_metrics.tn}, FP={audio_metrics.fp}, FN={audio_metrics.fn}")

    print("\n" + "=" * 50)
    print("V1 BENCHMARK SUMMARY")
    print("=" * 50 + "\n")

    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["content_type", "samples", "accuracy", "precision", "recall", "f1"])
        for name, metrics in all_metrics.items():
            print_metrics(metrics, name)
            writer.writerow([name, metrics.total, f"{metrics.accuracy:.2f}", f"{metrics.precision:.2f}", f"{metrics.recall:.2f}", f"{metrics.f1:.2f}"])

    print(f"Results saved to: {RESULTS_DIR}/summary.csv")

if __name__ == "__main__":
    main()
