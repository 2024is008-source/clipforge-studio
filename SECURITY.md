# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `1.0.x` | :white_check_mark: |

## Security Architecture
- **Local-First Execution**: Speech transcription (`faster-whisper`), computer vision (`YuNet`, `YOLOv8`, `SAM 2`), and FFmpeg rendering execute locally on `127.0.0.1:8765`.
- **Bearer Token Protection**: All non-health endpoints on the local worker require `X-ClipForge-Token` or `Authorization: Bearer`.
- **Strict Path Validation**: `source_id` and `overlay_id` parameters reject path traversal sequences (`..`, `/`, `\`).

## Reporting a Vulnerability
Please report security vulnerabilities privately via GitHub Security Advisories on `https://github.com/2024is008-source/clipforge-studio/security/advisories` rather than opening a public issue.
