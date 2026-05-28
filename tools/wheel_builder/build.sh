#!/bin/bash
set -e

SELF_DIR="$(dirname "$(realpath "$0")")"
cd "$SELF_DIR/../.."

VERSION=$(cat VERSION)
BUILD_DIR=$SELF_DIR/build/stage
DIST_DIR=$SELF_DIR/build/dist

# Compose a PEP 440 local version when CI metadata is provided so the wheel is
# traceable to a specific build and commit. With both env vars set the version
# becomes e.g. "3.0.0+build123.abc1234" (build number is monotonic, sha slug
# pins the source). If either is missing, fall back to the plain VERSION so
# local dev builds stay simple.
WHEEL_BUILD_NUMBER="${WHEEL_BUILD_NUMBER:-}"
WHEEL_SHA="${WHEEL_SHA:-}"
if [ -n "$WHEEL_BUILD_NUMBER" ] && [ -n "$WHEEL_SHA" ]; then
  SHA_SLUG="${WHEEL_SHA:0:7}"
  WHEEL_VERSION="${VERSION}+build${WHEEL_BUILD_NUMBER}.${SHA_SLUG}"
else
  WHEEL_VERSION="${VERSION}"
fi
echo "[WHEEL VERSION] $WHEEL_VERSION"

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

# Promote sub-packages (isaaclab_assets, isaaclab_rl, etc.) to top-level
# so they are importable as e.g. `import isaaclab_assets`.
# Each extension has the structure: source/isaaclab_FOO/isaaclab_FOO/ (Python pkg)
# plus sibling dirs like config/, data/. The __init__.py references ../config etc.
# We copy the inner Python package to src/ and also copy sibling resource dirs
# (config, data) into it so the relative-path lookups in __init__.py work.
for ext_dir in "$BUILD_DIR"/src/isaaclab/source/isaaclab_*; do
    pkg=$(basename "$ext_dir")
    inner="$ext_dir/$pkg"
    if [ -d "$inner" ] && [ -f "$inner/__init__.py" ]; then
        cp -r "$inner" "$BUILD_DIR/src/$pkg"
        # Copy resource dirs (config/, data/) into the Python package
        for res_dir in config data; do
            if [ -d "$ext_dir/$res_dir" ]; then
                cp -r "$ext_dir/$res_dir" "$BUILD_DIR/src/$pkg/$res_dir"
            fi
        done
        # Patch EXT_DIR: change '../' to '.' so __init__.py finds config/ inside
        # the package dir rather than one level up.
        sed -i 's|os\.path\.join(os\.path\.dirname(__file__), "\.\./"|os.path.join(os.path.dirname(__file__), ""|g' \
            "$BUILD_DIR/src/$pkg/__init__.py"
        # Remove the original from inside the isaaclab bundle to avoid duplication
        rm -rf "$ext_dir"
    fi
done

# Clean build artifacts that shouldn't be in the wheel
find "$BUILD_DIR/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/src" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR/src" -name "*.pyc" -delete 2>/dev/null || true

# 2. Copy the custom res __init__.py and __main__.py
cp "$SELF_DIR/res/__init__.py" "$BUILD_DIR/src/isaaclab/"
cp "$SELF_DIR/res/__main__.py" "$BUILD_DIR/src/isaaclab/"

# 3. Generate pyproject.toml with dependencies from python_packages.toml
python3 "$SELF_DIR/gen_pyproject.py" "$SELF_DIR/res/python_packages.toml" "$BUILD_DIR/pyproject.toml" "$WHEEL_VERSION"

# 4. Build the wheel
cd "$BUILD_DIR"
# Prefer --user to avoid polluting system Python; fall back to --break-system-packages
# for environments where --user is unsupported (e.g. Docker, ephemeral CI runners).
python3 -m pip install --user build wheel 2>/dev/null || python3 -m pip install --break-system-packages build wheel
python3 -m build --wheel --outdir "$DIST_DIR/"

# 5. Retag the wheel to match official platform tags
# cd "$DIST_DIR"
# GENERIC_WHL=$(ls isaaclab-*.whl)
# echo "Retagging $GENERIC_WHL -> $PYTHON_TAG-$ABI_TAG-$PLATFORM_TAG"
# python3 -m wheel tags --python-tag "$PYTHON_TAG" --abi-tag "$ABI_TAG" --platform-tag "$PLATFORM_TAG" "$GENERIC_WHL"
# # Remove the generic wheel (wheel tags creates a new file)
# TAGGED_WHL=$(ls isaaclab-*"$PLATFORM_TAG"*.whl 2>/dev/null)
# if [ "$GENERIC_WHL" != "$TAGGED_WHL" ] && [ -n "$TAGGED_WHL" ]; then
#     rm -f "$GENERIC_WHL"
# fi

echo ""
echo "[WHEEL BUILT]"
ls -lh $DIST_DIR/isaaclab-*.whl
