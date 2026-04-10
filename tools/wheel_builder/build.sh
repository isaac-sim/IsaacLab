#!/bin/bash
set -e

SELF_DIR="$(dirname "$(realpath "$0")")"
cd "$SELF_DIR/../.."

VERSION=$(cat VERSION)
BUILD_DIR=$SELF_DIR/build/stage
DIST_DIR=$SELF_DIR/build/dist

# Platform tags matching the official IsaacLab wheel
PYTHON_TAG="${PYTHON_TAG:-cp312}"
ABI_TAG="${ABI_TAG:-cp312}"
# Auto-detect platform
ARCH=$(uname -m)
case "$ARCH" in
    x86_64|AMD64)  PLATFORM_TAG="${PLATFORM_TAG:-manylinux_2_35_x86_64}" ;;
    aarch64|arm64) PLATFORM_TAG="${PLATFORM_TAG:-manylinux_2_35_aarch64}" ;;
    *)             PLATFORM_TAG="${PLATFORM_TAG:-linux_$ARCH}" ;;
esac

rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR/src/isaaclab"

# 1. Copy inventory (same as python_packages.toml inventory.includes.all)
cp -r apps "$BUILD_DIR/src/isaaclab/"
cp -r source "$BUILD_DIR/src/isaaclab/"

# Ensure apps/ is discovered as a Python sub-package (it has no __init__.py)
find "$BUILD_DIR/src/isaaclab/apps" -type d -exec touch {}/__init__.py \;

# Clean build artifacts that shouldn't be in the wheel
find "$BUILD_DIR/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/src" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/src" -name "*.pyc" -delete 2>/dev/null || true

# 2. Copy the custom res __init__.py and __main__.py
cp "$SELF_DIR/res/__init__.py" "$BUILD_DIR/src/isaaclab/"
cp "$SELF_DIR/res/__main__.py" "$BUILD_DIR/src/isaaclab/"

# 3. Generate pyproject.toml with dependencies from python_packages.toml
python3 "$SELF_DIR/gen_pyproject.py" "$SELF_DIR/res/python_packages.toml" "$BUILD_DIR/pyproject.toml" "$VERSION"

# 4. Build the wheel
cd "$BUILD_DIR"
pip install --break-system-packages build wheel
python -m build --wheel --outdir "$DIST_DIR/"

# 5. Retag the wheel to match official platform tags
cd "$DIST_DIR"
GENERIC_WHL=$(ls isaaclab-*.whl)
echo "Retagging $GENERIC_WHL -> $PYTHON_TAG-$ABI_TAG-$PLATFORM_TAG"
python3 -m wheel tags --python-tag "$PYTHON_TAG" --abi-tag "$ABI_TAG" --platform-tag "$PLATFORM_TAG" "$GENERIC_WHL"
# Remove the generic wheel (wheel tags creates a new file)
TAGGED_WHL=$(ls isaaclab-*"$PLATFORM_TAG"*.whl 2>/dev/null)
if [ "$GENERIC_WHL" != "$TAGGED_WHL" ] && [ -n "$TAGGED_WHL" ]; then
    rm -f "$GENERIC_WHL"
fi

echo ""
echo "[WHEEL BUILT]"
ls -lh isaaclab-*.whl
