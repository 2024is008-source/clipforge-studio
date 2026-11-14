#!/usr/bin/env bash
set -euo pipefail

echo "=== ClipForge GitHub Publisher ==="
git remote -v

if command -v gh >/dev/null 2>&1; then
  gh auth status
  gh repo create 2024is008-source/clipforge-next --public --description "AI-powered video clipping and editing platform." || true
  gh repo edit 2024is008-source/clipforge-next --add-topic ai-video-editing,video-editing,video-clipping,ffmpeg,computer-vision,object-detection,yolo,sam2,python,typescript || true
fi

git push -u origin main
echo "Successfully pushed clean ClipForge history to origin/main!"
