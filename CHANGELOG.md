# Changelog

All notable changes to **ClipForge** are documented in this file.

## [1.0.0] - 2026-11-14

### Added
- **Clean Modular Architecture**: Organized domain engines (`engines/video`, `engines/vision`, `engines/audio`, `engines/tracking`, `engines/rendering`), shared TypeScript contracts (`packages/contracts`), and unified test suites (`tests/unit`, `tests/integration`, `tests/e2e`).
- **GitHub Actions CI/CD Suite**: Automated workflows for frontend (`frontend.yml`), backend (`backend.yml`), pipeline QA (`tests.yml`), security scanning (`security.yml`), and release packaging (`release.yml`).
- **Brand Identity & Visual Assets**: Custom ClipForge Studio multi-size `favicon.ico`, `favicon.svg`, `apple-touch-icon.png`, and repository social preview banner.
- **Comprehensive Documentation**: Full architecture, API, rendering, AI model, and audit documentation with Mermaid diagrams.

### Fixed
- **FastAPI Router Lifespan Compatibility**: Added startup/shutdown event fallback in `local_worker/rendering/overlays.py` for compatibility across FastAPI versions.
- **Pydantic OverlayManifest Validation**: Accepted both `StrictFloat` and `StrictInt` for `fps` in `OverlayManifest`.
- **FFmpeg Subprocess Stdin Isolation**: Added `-nostdin` and `stdin=subprocess.DEVNULL` to `local_worker/rendering/renderer.py` to prevent non-interactive TTY stalls.
