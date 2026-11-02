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

rm -rf "$BUILD_DIR" "$DIST_DIR"

# Stage the same aggregate source tree used by the PEP 517 Git-source backend.
uv run --no-project --python 3.12 python "$SELF_DIR/stage.py" "$BUILD_DIR" "$WHEEL_VERSION"

# Build in uv's isolated PEP 517 environment without installing tools into the host Python.
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-12}"
uv build --wheel --out-dir "$DIST_DIR" "$BUILD_DIR"

echo ""
echo "[WHEEL BUILT]"
ls -lh $DIST_DIR/isaaclab-*.whl
