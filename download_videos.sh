#!/bin/bash
# Download YouTube videos (video only, no audio)

OUTPUT_DIR="/Users/aldenkling/Desktop/Personal Research/cv-player-tracking-all22/videos"
mkdir -p "$OUTPUT_DIR"

URLS=(
  "https://youtu.be/m-6yBBPd8Uk"
  "https://youtu.be/ynVIq2uRTS8"
  "https://youtu.be/4fLu9s70g34"
)

for url in "${URLS[@]}"; do
  echo "==> Downloading: $url"
  yt-dlp \
    -f "bestvideo[ext=mp4]" \
    -o "$OUTPUT_DIR/%(title)s.%(ext)s" \
    --no-audio \
    "$url"
  echo ""
done

echo "Done. Videos saved to: $OUTPUT_DIR"
