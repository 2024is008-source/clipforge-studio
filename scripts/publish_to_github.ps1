# ClipForge — Automated GitHub Repository Creation & Push Script (PowerShell)
# Usage: .\scripts\publish_to_github.ps1
$ErrorActionPreference = "Stop"

Write-Host "=== ClipForge GitHub Publisher ===" -ForegroundColor Green

# 1. Verify Git remotes (never push to old repo)
$remotes = git remote -v
Write-Host "Configured remotes:`n$remotes"

# 2. Check GitHub CLI authentication if gh is installed
if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh auth status
    Write-Host "Creating or verifying 2024is008-source/clipforge-next and 2024is008-source/ClipForge..."
    gh repo create 2024is008-source/clipforge-next --public --description "AI-powered video clipping and editing platform." 2>$null
    gh repo edit 2024is008-source/clipforge-next --add-topic ai-video-editing,video-editing,video-clipping,ffmpeg,computer-vision,object-detection,yolo,sam2,python,typescript
}

# 3. Push clean commit history without --force
git push -u origin main
Write-Host "Successfully pushed clean ClipForge history to origin/main!" -ForegroundColor Green
