#!/bin/sh
# setup-tessdata.sh
# Run once from project root to pre-cache Tesseract language data locally.
# Avoids CDN download on every boot.
#
# Usage: sh setup-tessdata.sh

set -e

DEST="./tessdata"
mkdir -p "$DEST"

GZ="$DEST/eng.traineddata.gz"

# Already done
if [ -f "$GZ" ]; then
  echo "✓ $GZ already exists, nothing to do."
  exit 0
fi

echo "Looking for eng.traineddata on this system..."

# Common locations across Linux distros and macOS
SEARCH_PATHS="
  /usr/share/tesseract-ocr/5/tessdata/eng.traineddata
  /usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata
  /usr/share/tessdata/eng.traineddata
  /usr/local/share/tessdata/eng.traineddata
  /opt/homebrew/share/tessdata/eng.traineddata
  /usr/local/Cellar/tesseract/*/share/tessdata/eng.traineddata
  /opt/local/share/tessdata/eng.traineddata
"

FOUND=""
for p in $SEARCH_PATHS; do
  # Handle glob expansion for Homebrew paths
  for expanded in $p; do
    if [ -f "$expanded" ]; then
      FOUND="$expanded"
      break 2
    fi
  done
done

if [ -n "$FOUND" ]; then
  echo "Found: $FOUND"
  cp "$FOUND" "$DEST/eng.traineddata"
  gzip -kf "$DEST/eng.traineddata"
  rm "$DEST/eng.traineddata"
  echo "✓ Written to $GZ"
  exit 0
fi

# Try tesseract CLI to locate its data dir
if command -v tesseract > /dev/null 2>&1; then
  TESS_DIR=$(tesseract --print-parameters 2>/dev/null | grep tessdata_dir | awk '{print $2}' || true)
  if [ -z "$TESS_DIR" ]; then
    # Fallback: ask tesseract where its data is via --list-langs output
    TESS_DIR=$(tesseract --list-langs 2>&1 | grep '"' | sed 's/.*"\(.*\)".*/\1/')
  fi
  if [ -n "$TESS_DIR" ] && [ -f "$TESS_DIR/eng.traineddata" ]; then
    echo "Found via tesseract CLI: $TESS_DIR/eng.traineddata"
    cp "$TESS_DIR/eng.traineddata" "$DEST/eng.traineddata"
    gzip -kf "$DEST/eng.traineddata"
    rm "$DEST/eng.traineddata"
    echo "✓ Written to $GZ"
    exit 0
  fi
fi

# Direct download from GitHub (tessdata_best — higher accuracy than fast)
echo "System tessdata not found. Downloading eng.traineddata from GitHub..."
URL="https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"

if command -v curl > /dev/null 2>&1; then
  curl -L --progress-bar "$URL" -o "$DEST/eng.traineddata"
elif command -v wget > /dev/null 2>&1; then
  wget -q --show-progress "$URL" -O "$DEST/eng.traineddata"
else
  echo "✗ Neither curl nor wget found. Install one and re-run, or manually download:"
  echo "  $URL"
  echo "  → save to $DEST/eng.traineddata, then run: gzip -k $DEST/eng.traineddata"
  exit 1
fi

gzip -kf "$DEST/eng.traineddata"
rm "$DEST/eng.traineddata"
echo "✓ Written to $GZ"