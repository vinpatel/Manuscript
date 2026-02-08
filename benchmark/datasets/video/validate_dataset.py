#!/usr/bin/env python3
"""
Video Dataset Validator for Manuscript Benchmark

Validates downloaded videos against metadata requirements:
- Checks file existence and integrity
- Verifies video duration (3-30 seconds)
- Confirms file size limits (<50MB)
- Reports missing or invalid files

Usage:
    python validate_dataset.py
    python validate_dataset.py --verbose
    python validate_dataset.py --fix-metadata
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
HUMAN_DIR = SCRIPT_DIR / "human"
AI_DIR = SCRIPT_DIR / "ai_generated"
METADATA_FILE = SCRIPT_DIR / "metadata.json"

# Validation constants
MIN_DURATION = 3  # seconds
MAX_DURATION = 30  # seconds
MAX_SIZE_MB = 50
VALID_FORMATS = {".mp4", ".webm", ".mov"}


def get_video_info(video_path):
    """Get video duration and resolution using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)

        duration = float(data.get('format', {}).get('duration', 0))
        resolution = "unknown"

        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                resolution = f"{width}x{height}"
                break

        return {
            'duration': duration,
            'resolution': resolution,
            'format': video_path.suffix.lower().lstrip('.'),
            'valid': True
        }
    except FileNotFoundError:
        return {'error': 'ffprobe not found', 'valid': False}
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        return {'error': str(e), 'valid': False}


def validate_video_file(video_path, expected_metadata=None):
    """Validate a single video file."""
    issues = []

    if not video_path.exists():
        return {'exists': False, 'issues': ['File not found']}

    # Check file size
    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        issues.append(f"File too large: {size_mb:.1f}MB (max: {MAX_SIZE_MB}MB)")

    if size_mb == 0:
        issues.append("File is empty")
        return {'exists': True, 'size_mb': 0, 'issues': issues}

    # Check format
    if video_path.suffix.lower() not in VALID_FORMATS:
        issues.append(f"Invalid format: {video_path.suffix}")

    # Get video info
    info = get_video_info(video_path)

    if not info.get('valid', False):
        issues.append(f"Could not read video: {info.get('error', 'unknown error')}")
    else:
        # Check duration
        duration = info.get('duration', 0)
        if duration < MIN_DURATION:
            issues.append(f"Too short: {duration:.1f}s (min: {MIN_DURATION}s)")
        elif duration > MAX_DURATION:
            issues.append(f"Too long: {duration:.1f}s (max: {MAX_DURATION}s)")

    return {
        'exists': True,
        'size_mb': size_mb,
        'duration': info.get('duration', 0),
        'resolution': info.get('resolution', 'unknown'),
        'format': info.get('format', 'unknown'),
        'issues': issues
    }


def validate_dataset(verbose=False):
    """Validate the entire video dataset."""
    results = {
        'human': {'found': 0, 'valid': 0, 'missing': 0, 'invalid': []},
        'ai_generated': {'found': 0, 'valid': 0, 'missing': 0, 'invalid': []}
    }

    # Load metadata
    if not METADATA_FILE.exists():
        print("ERROR: metadata.json not found")
        return results

    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # Validate human videos
    print("\n=== Validating Human Videos ===")
    for entry in metadata.get('human', []):
        filename = entry.get('file_name')
        video_path = HUMAN_DIR / filename

        validation = validate_video_file(video_path)

        if not validation['exists']:
            results['human']['missing'] += 1
            if verbose:
                print(f"  MISSING: {filename}")
        elif validation['issues']:
            results['human']['found'] += 1
            results['human']['invalid'].append({
                'file': filename,
                'issues': validation['issues']
            })
            if verbose:
                print(f"  INVALID: {filename}")
                for issue in validation['issues']:
                    print(f"           - {issue}")
        else:
            results['human']['found'] += 1
            results['human']['valid'] += 1
            if verbose:
                print(f"  OK: {filename} ({validation['duration']:.1f}s, {validation['resolution']})")

    # Validate AI videos
    print("\n=== Validating AI-Generated Videos ===")
    for entry in metadata.get('ai_generated', []):
        filename = entry.get('file_name')
        video_path = AI_DIR / filename

        validation = validate_video_file(video_path)

        if not validation['exists']:
            results['ai_generated']['missing'] += 1
            if verbose:
                print(f"  MISSING: {filename}")
        elif validation['issues']:
            results['ai_generated']['found'] += 1
            results['ai_generated']['invalid'].append({
                'file': filename,
                'issues': validation['issues']
            })
            if verbose:
                print(f"  INVALID: {filename}")
                for issue in validation['issues']:
                    print(f"           - {issue}")
        else:
            results['ai_generated']['found'] += 1
            results['ai_generated']['valid'] += 1
            if verbose:
                print(f"  OK: {filename} ({validation['duration']:.1f}s, {validation['resolution']})")

    return results


def print_summary(results):
    """Print validation summary."""
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    for category in ['human', 'ai_generated']:
        r = results[category]
        total_expected = 50
        print(f"\n{category.upper().replace('_', ' ')} VIDEOS:")
        print(f"  Found:   {r['found']}/{total_expected}")
        print(f"  Valid:   {r['valid']}/{total_expected}")
        print(f"  Missing: {r['missing']}/{total_expected}")
        print(f"  Invalid: {len(r['invalid'])}")

        if r['invalid']:
            print("  Issues:")
            for item in r['invalid'][:5]:  # Show first 5
                print(f"    - {item['file']}: {', '.join(item['issues'])}")
            if len(r['invalid']) > 5:
                print(f"    ... and {len(r['invalid']) - 5} more")

    total_valid = results['human']['valid'] + results['ai_generated']['valid']
    total_expected = 100
    print(f"\n{'=' * 50}")
    print(f"TOTAL: {total_valid}/{total_expected} videos ready for benchmark")

    if total_valid == total_expected:
        print("STATUS: Dataset complete!")
        return 0
    else:
        print("STATUS: Dataset incomplete - see DOWNLOAD_INSTRUCTIONS.md")
        return 1


def update_metadata_from_files(dry_run=True):
    """Update metadata.json with actual values from downloaded files."""
    if not METADATA_FILE.exists():
        print("ERROR: metadata.json not found")
        return

    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    updates = 0

    # Update human videos
    for entry in metadata.get('human', []):
        video_path = HUMAN_DIR / entry['file_name']
        if video_path.exists():
            info = get_video_info(video_path)
            if info.get('valid'):
                if entry.get('duration_seconds', 0) == 0 or entry.get('status') == 'requires_download':
                    entry['duration_seconds'] = round(info['duration'], 1)
                    entry['resolution'] = info['resolution']
                    entry['format'] = info['format']
                    entry['status'] = 'downloaded'
                    updates += 1
                    print(f"Updated: {entry['file_name']}")

    # Update AI videos
    for entry in metadata.get('ai_generated', []):
        video_path = AI_DIR / entry['file_name']
        if video_path.exists():
            info = get_video_info(video_path)
            if info.get('valid'):
                if entry.get('duration_seconds', 0) == 0 or 'requires' in entry.get('status', ''):
                    entry['duration_seconds'] = round(info['duration'], 1)
                    entry['resolution'] = info['resolution']
                    entry['format'] = info['format']
                    entry['status'] = 'downloaded'
                    updates += 1
                    print(f"Updated: {entry['file_name']}")

    if updates > 0:
        if dry_run:
            print(f"\nDry run: Would update {updates} entries")
            print("Run with --fix-metadata to apply changes")
        else:
            metadata['last_validated'] = str(datetime.now())
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"\nUpdated {updates} entries in metadata.json")
    else:
        print("No updates needed")


def main():
    parser = argparse.ArgumentParser(description='Validate Manuscript video dataset')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show details for each video')
    parser.add_argument('--fix-metadata', action='store_true',
                        help='Update metadata.json with actual video info')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without making changes')

    args = parser.parse_args()

    # Ensure directories exist
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)

    if args.fix_metadata or args.dry_run:
        update_metadata_from_files(dry_run=not args.fix_metadata)
    else:
        results = validate_dataset(verbose=args.verbose)
        sys.exit(print_summary(results))


if __name__ == "__main__":
    main()
