#!/bin/bash

# Script to create the package and generate dist files
# Usage: ./package.sh

set -e

echo "=========================================="
echo "  OMNI42_AGENTS - Package Builder"
echo "=========================================="

# Clean previous builds
echo ""
echo "[1/4] Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build dependencies if needed
echo ""
echo "[2/4] Checking build dependencies..."
pip install --quiet --upgrade pip setuptools wheel build

# Build the package
echo ""
echo "[3/4] Building the package..."
python -m build

# Show results
echo ""
echo "[4/4] Build complete!"
echo ""
echo "Generated files in dist/:"
ls -la dist/

echo ""
echo "=========================================="
echo "  Package created successfully!"
echo "=========================================="
echo ""
echo "To upload to repository, use:"
echo "  twine upload --repository-url <REPO_URL> dist/*"
echo ""
