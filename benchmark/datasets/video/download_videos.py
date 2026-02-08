#!/usr/bin/env python3
"""
Video Dataset Downloader for Manuscript Benchmark

Downloads human-recorded and AI-generated video samples from various public sources.
Requires API keys for some sources (see DOWNLOAD_INSTRUCTIONS.md).

Usage:
    python download_videos.py --source pexels --api-key YOUR_KEY
    python download_videos.py --source pixabay --api-key YOUR_KEY
    python download_videos.py --source archive
    python download_videos.py --all --pexels-key KEY --pixabay-key KEY
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import time

# Base paths
SCRIPT_DIR = Path(__file__).parent
HUMAN_DIR = SCRIPT_DIR / "human"
AI_DIR = SCRIPT_DIR / "ai_generated"

# Metadata storage
METADATA_FILE = SCRIPT_DIR / "metadata.json"

def load_metadata():
    """Load existing metadata or create new."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"human": [], "ai_generated": [], "download_date": str(datetime.now())}

def save_metadata(metadata):
    """Save metadata to JSON file."""
    metadata["last_updated"] = str(datetime.now())
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def download_file(url, output_path, max_size_mb=50):
    """Download a file with size limit."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as response:
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                print(f"  Skipping {url}: File too large ({int(content_length) / 1024 / 1024:.1f}MB)")
                return False

            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
            return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False

def download_pexels_videos(api_key, count=50):
    """Download videos from Pexels API."""
    print(f"\n=== Downloading {count} videos from Pexels ===")

    if not api_key:
        print("Error: Pexels API key required. Get one at https://www.pexels.com/api/")
        return []

    downloaded = []
    queries = ["nature", "people", "city", "animals", "technology", "food", "travel", "sports", "ocean", "forest"]
    videos_per_query = count // len(queries) + 1

    for query in queries:
        if len(downloaded) >= count:
            break

        url = f"https://api.pexels.com/videos/search?query={query}&per_page={videos_per_query}&size=small"
        req = urllib.request.Request(url, headers={
            'Authorization': api_key,
            'User-Agent': 'Manuscript-Benchmark/1.0'
        })

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())

            for video in data.get('videos', []):
                if len(downloaded) >= count:
                    break

                # Find smallest video file under 50MB
                video_files = sorted(video.get('video_files', []),
                                    key=lambda x: x.get('width', 0))

                for vf in video_files:
                    if vf.get('file_type') == 'video/mp4':
                        video_url = vf.get('link')
                        idx = len(downloaded) + 1
                        filename = f"human_pexels_{query}_{idx:03d}.mp4"
                        output_path = HUMAN_DIR / filename

                        print(f"  Downloading: {filename}")
                        if download_file(video_url, output_path):
                            downloaded.append({
                                "file_name": filename,
                                "source_url": f"https://www.pexels.com/video/{video.get('id')}",
                                "generator": "human_recorded",
                                "duration_seconds": video.get('duration', 0),
                                "resolution": f"{vf.get('width', 0)}x{vf.get('height', 0)}",
                                "format": "mp4",
                                "content_description": f"Pexels stock video: {query}",
                                "license": "Pexels License (free for commercial use)"
                            })
                            time.sleep(0.5)  # Rate limiting
                        break

        except Exception as e:
            print(f"  Error fetching Pexels videos for '{query}': {e}")

    return downloaded

def download_pixabay_videos(api_key, count=50):
    """Download videos from Pixabay API."""
    print(f"\n=== Downloading {count} videos from Pixabay ===")

    if not api_key:
        print("Error: Pixabay API key required. Get one at https://pixabay.com/api/docs/")
        return []

    downloaded = []
    queries = ["nature", "people", "city", "animals", "technology", "food", "travel", "sports", "water", "landscape"]
    videos_per_query = count // len(queries) + 1

    for query in queries:
        if len(downloaded) >= count:
            break

        url = f"https://pixabay.com/api/videos/?key={api_key}&q={query}&per_page={videos_per_query}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())

            for video in data.get('hits', []):
                if len(downloaded) >= count:
                    break

                # Use small or tiny video
                video_url = video.get('videos', {}).get('small', {}).get('url') or \
                           video.get('videos', {}).get('tiny', {}).get('url')

                if video_url:
                    idx = len(downloaded) + 1
                    filename = f"human_pixabay_{query}_{idx:03d}.mp4"
                    output_path = HUMAN_DIR / filename

                    print(f"  Downloading: {filename}")
                    if download_file(video_url, output_path):
                        small_vid = video.get('videos', {}).get('small', {})
                        downloaded.append({
                            "file_name": filename,
                            "source_url": video.get('pageURL', ''),
                            "generator": "human_recorded",
                            "duration_seconds": video.get('duration', 0),
                            "resolution": f"{small_vid.get('width', 0)}x{small_vid.get('height', 0)}",
                            "format": "mp4",
                            "content_description": f"Pixabay stock video: {query}",
                            "license": "Pixabay License (free for commercial use)"
                        })
                        time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  Error fetching Pixabay videos for '{query}': {e}")

    return downloaded

def download_archive_videos(count=20):
    """Download public domain videos from Internet Archive."""
    print(f"\n=== Downloading {count} videos from Internet Archive ===")

    # Curated list of public domain video identifiers (short clips)
    archive_items = [
        {"id": "C-SPAN_20160101_000000_Open_Phones", "desc": "C-SPAN open phones segment"},
        {"id": "PublicDomainFootage-various", "desc": "Various public domain footage"},
        {"id": "newsreel_clip_001", "desc": "Historical newsreel"},
        {"id": "home_movie_1950s", "desc": "1950s home movie"},
        {"id": "educational_film_1960", "desc": "Educational film from 1960s"},
    ]

    downloaded = []
    print("  Note: Internet Archive requires specific item IDs. See DOWNLOAD_INSTRUCTIONS.md")
    print("  Placeholder entries created for documentation purposes.")

    # Create placeholder metadata for manual download
    for i, item in enumerate(archive_items[:count]):
        downloaded.append({
            "file_name": f"human_archive_{i+1:03d}.mp4",
            "source_url": f"https://archive.org/details/{item['id']}",
            "generator": "human_recorded",
            "duration_seconds": 0,  # To be filled after download
            "resolution": "unknown",
            "format": "mp4",
            "content_description": item['desc'],
            "license": "Public Domain",
            "status": "requires_manual_download"
        })

    return downloaded

def get_ai_video_sources():
    """Document AI video sources for the benchmark."""
    print("\n=== Documenting AI-Generated Video Sources ===")

    ai_sources = []

    # Sora samples (OpenAI)
    sora_samples = [
        {"desc": "Tokyo street scene", "url": "https://openai.com/sora", "type": "sora"},
        {"desc": "Woolly mammoth in snow", "url": "https://openai.com/sora", "type": "sora"},
        {"desc": "Cat waking owner", "url": "https://openai.com/sora", "type": "sora"},
        {"desc": "Drone through archway", "url": "https://openai.com/sora", "type": "sora"},
        {"desc": "Gold rush California", "url": "https://openai.com/sora", "type": "sora"},
    ]

    for i, sample in enumerate(sora_samples):
        ai_sources.append({
            "file_name": f"ai_sora_{i+1:03d}.mp4",
            "source_url": sample['url'],
            "generator": "sora",
            "duration_seconds": 0,
            "resolution": "1920x1080",
            "format": "mp4",
            "content_description": f"OpenAI Sora demo: {sample['desc']}",
            "status": "requires_manual_download",
            "notes": "Download from OpenAI Sora demos page when available"
        })

    # Runway ML samples
    runway_samples = [
        {"desc": "Text-to-video generation", "type": "runway_gen2"},
        {"desc": "Image animation", "type": "runway_gen2"},
        {"desc": "Style transfer video", "type": "runway_gen2"},
        {"desc": "Motion brush creation", "type": "runway_gen2"},
        {"desc": "Character animation", "type": "runway_gen2"},
    ]

    for i, sample in enumerate(runway_samples):
        ai_sources.append({
            "file_name": f"ai_runway_{i+1:03d}.mp4",
            "source_url": "https://runwayml.com/research/",
            "generator": sample['type'],
            "duration_seconds": 0,
            "resolution": "unknown",
            "format": "mp4",
            "content_description": f"Runway ML: {sample['desc']}",
            "status": "requires_manual_download",
            "notes": "Generate via Runway ML Gen-2 or download from research page"
        })

    # Pika Labs samples
    pika_samples = [
        {"desc": "Text-to-video scene", "type": "pika"},
        {"desc": "Image-to-video animation", "type": "pika"},
        {"desc": "Style transformation", "type": "pika"},
    ]

    for i, sample in enumerate(pika_samples):
        ai_sources.append({
            "file_name": f"ai_pika_{i+1:03d}.mp4",
            "source_url": "https://pika.art/",
            "generator": sample['type'],
            "duration_seconds": 0,
            "resolution": "unknown",
            "format": "mp4",
            "content_description": f"Pika Labs: {sample['desc']}",
            "status": "requires_manual_download"
        })

    # Stable Video Diffusion samples
    svd_samples = [
        {"desc": "Image-to-video generation", "type": "stable_video_diffusion"},
        {"desc": "Motion synthesis", "type": "stable_video_diffusion"},
        {"desc": "Scene animation", "type": "stable_video_diffusion"},
    ]

    for i, sample in enumerate(svd_samples):
        ai_sources.append({
            "file_name": f"ai_svd_{i+1:03d}.mp4",
            "source_url": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid",
            "generator": sample['type'],
            "duration_seconds": 0,
            "resolution": "unknown",
            "format": "mp4",
            "content_description": f"Stable Video Diffusion: {sample['desc']}",
            "status": "requires_manual_download"
        })

    # DeepFake samples (for detection research)
    deepfake_samples = [
        {"desc": "Face swap sample", "type": "deepfake", "dataset": "FaceForensics++"},
        {"desc": "Lip sync manipulation", "type": "deepfake", "dataset": "DFDC"},
        {"desc": "Expression transfer", "type": "deepfake", "dataset": "Celeb-DF"},
    ]

    for i, sample in enumerate(deepfake_samples):
        ai_sources.append({
            "file_name": f"ai_deepfake_{i+1:03d}.mp4",
            "source_url": f"https://github.com/deepfakes/faceswap",
            "generator": sample['type'],
            "duration_seconds": 0,
            "resolution": "unknown",
            "format": "mp4",
            "content_description": f"Deepfake sample from {sample['dataset']}",
            "status": "requires_registration",
            "dataset": sample['dataset'],
            "notes": "Academic access required for deepfake datasets"
        })

    # Fill remaining slots with various generators
    generators = ["kling", "minimax", "luma", "haiper", "genmo"]
    idx = len(ai_sources) + 1

    for gen in generators:
        for i in range(5):  # 5 samples per generator
            if len(ai_sources) >= 50:
                break
            ai_sources.append({
                "file_name": f"ai_{gen}_{i+1:03d}.mp4",
                "source_url": f"https://www.google.com/search?q={gen}+ai+video+generator",
                "generator": gen,
                "duration_seconds": 0,
                "resolution": "unknown",
                "format": "mp4",
                "content_description": f"{gen.capitalize()} AI video generation sample",
                "status": "requires_manual_download"
            })

    return ai_sources[:50]  # Limit to 50

def main():
    parser = argparse.ArgumentParser(description='Download video datasets for Manuscript benchmark')
    parser.add_argument('--source', choices=['pexels', 'pixabay', 'archive', 'ai', 'all'],
                       help='Video source to download from')
    parser.add_argument('--pexels-key', help='Pexels API key')
    parser.add_argument('--pixabay-key', help='Pixabay API key')
    parser.add_argument('--count', type=int, default=50, help='Number of videos to download per category')
    parser.add_argument('--init-metadata', action='store_true',
                       help='Initialize metadata with placeholder entries')

    args = parser.parse_args()

    # Ensure directories exist
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    if args.init_metadata:
        # Initialize with placeholder metadata for documentation
        print("Initializing metadata with placeholder entries...")

        # Human video placeholders
        human_entries = []
        for i in range(1, 51):
            source = "pexels" if i <= 20 else ("pixabay" if i <= 40 else "archive")
            human_entries.append({
                "file_name": f"human_{source}_{i:03d}.mp4",
                "source_url": f"https://{source}.com/video/{i}",
                "generator": "human_recorded",
                "duration_seconds": 0,
                "resolution": "unknown",
                "format": "mp4",
                "content_description": f"Human-recorded video from {source}",
                "status": "placeholder"
            })

        # AI video placeholders
        ai_entries = get_ai_video_sources()

        metadata["human"] = human_entries
        metadata["ai_generated"] = ai_entries
        save_metadata(metadata)
        print(f"Metadata initialized with {len(human_entries)} human and {len(ai_entries)} AI entries")
        return

    if args.source == 'pexels' or args.source == 'all':
        entries = download_pexels_videos(args.pexels_key, args.count)
        metadata["human"].extend(entries)

    if args.source == 'pixabay' or args.source == 'all':
        entries = download_pixabay_videos(args.pixabay_key, args.count)
        metadata["human"].extend(entries)

    if args.source == 'archive' or args.source == 'all':
        entries = download_archive_videos(min(args.count, 20))
        metadata["human"].extend(entries)

    if args.source == 'ai' or args.source == 'all':
        entries = get_ai_video_sources()
        metadata["ai_generated"] = entries

    save_metadata(metadata)
    print(f"\nMetadata saved to {METADATA_FILE}")
    print(f"Human videos: {len(metadata['human'])} entries")
    print(f"AI videos: {len(metadata['ai_generated'])} entries")

if __name__ == "__main__":
    main()
