#!/bin/bash
# Manuscript Local Setup Script
# Run this on your Mac to get started

set -e

echo "=========================================="
echo "  Manuscript Local Setup"
echo "=========================================="

# Create project directory
PROJECT_DIR="$HOME/manuscript"

if [ -d "$PROJECT_DIR" ]; then
    echo "⚠️  Directory $PROJECT_DIR already exists."
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm -rf "$PROJECT_DIR"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "📁 Created project at $PROJECT_DIR"

# Initialize Go module
echo "📦 Initializing Go module..."
go mod init github.com/manuscript/manuscript

# Create directory structure
mkdir -p cmd/api
mkdir -p internal/{config,handler,middleware,repository,service}
mkdir -p pkg/logger

echo "📂 Directory structure created."
echo ""
echo "=========================================="
echo "  Next Steps:"
echo "=========================================="
echo ""
echo "1. Copy the source files from Claude to:"
echo "   $PROJECT_DIR"
echo ""
echo "2. Or download from GitHub (once published):"
echo "   git clone https://github.com/manuscript/manuscript.git"
echo ""
echo "3. Run the server:"
echo "   cd $PROJECT_DIR"
echo "   go run cmd/api/main.go"
echo ""
echo "4. Test the API:"
echo "   curl http://localhost:8080/health"
echo ""
