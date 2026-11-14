# Contributing to ClipForge Studio

Thank you for investing time in contributing to ClipForge Studio! 🎬

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Branch Naming](#branch-naming)
5. [Commit Conventions](#commit-conventions)
6. [Pull Request Workflow](#pull-request-workflow)
7. [Code Style](#code-style)
8. [Testing Requirements](#testing-requirements)
9. [Issue Templates](#issue-templates)
10. [License Agreement](#license-agreement)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold it. Report unacceptable behaviour to the maintainers.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/clipforge-studio.git
   cd clipforge-studio
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/2024is008-source/clipforge-studio.git
   ```
4. Set up your development environment (see below).

---

## Development Setup

### Prerequisites

- **Node.js** ≥ 20 (or **Bun** ≥ 1.1)
- **Python** ≥ 3.11
- **FFmpeg** in PATH
- **CUDA 12** (optional, for GPU acceleration)
- A `.env` file copied from `.env.example` with your API keys filled in

### Frontend

```bash
bun install         # or: npm install
bun run dev         # TanStack Start dev server
```

### Backend (Local GPU Worker)

```bash
cd local_worker
python -m venv .venv
.venv\Scripts\activate        # Windows
# or: source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8765 --reload
```

---

## Branch Naming

Use lowercase kebab-case with a category prefix:

| Category | Pattern | Example |
|---|---|---|
| Feature | `feat/<short-description>` | `feat/active-speaker-tracking` |
| Bug fix | `fix/<short-description>` | `fix/caption-timing-off-by-one` |
| Docs | `docs/<short-description>` | `docs/update-setup-guide` |
| Refactor | `refactor/<short-description>` | `refactor/render-pipeline` |
| Chore | `chore/<short-description>` | `chore/update-dependencies` |
| Tests | `test/<short-description>` | `test/whisper-integration` |

Always branch from `main`:
```bash
git fetch upstream
git checkout -b feat/my-feature upstream/main
```

---

## Commit Conventions

This project uses **[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)**.

### Format

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

### Types

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code change that is neither fix nor feature |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `chore` | Build process, dependencies, tooling |
| `ci` | CI/CD configuration |
| `revert` | Reverts a previous commit |

### Scopes (optional)

`worker`, `frontend`, `render`, `captions`, `tracking`, `gemini`, `supabase`, `deps`, `repo`, `docs`

### Examples

```
feat(tracking): add SAM2 segmentation for precise subject isolation
fix(render): correct NVENC encoder fallback order on Windows
docs(worker): document Gemini API key rotation pool
chore(deps): upgrade faster-whisper to 1.1.0
```

### Rules

- Use the **imperative mood** in the summary: "add feature" not "added feature".
- Keep the summary **≤ 72 characters**.
- Reference issues in the footer: `Closes #42`, `Fixes #17`.
- Mark breaking changes: add `!` after the type, or `BREAKING CHANGE:` in the footer.

---

## Pull Request Workflow

1. **Keep PRs focused** — one feature or fix per PR.
2. **Rebase** (do not merge) your branch on top of `upstream/main` before opening a PR:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
3. **Fill out the PR template** completely. PRs with empty descriptions will be closed.
4. **Request review** from at least one maintainer.
5. **All CI checks must pass** before merging.
6. Maintainers merge using **Squash and Merge** to keep history clean.

---

## Code Style

### TypeScript / JavaScript

- **Formatter**: Prettier (config in `.prettierrc`). Run `bun run format`.
- **Linter**: ESLint (config in `eslint.config.js`). Run `bun run lint`.
- Prefer `const` over `let`; avoid `var`.
- Use explicit TypeScript types — avoid `any`.
- Keep components under 300 lines; extract logic into hooks or utilities.

### Python

- **Formatter**: `ruff format` (configured in `pyproject.toml`).
- **Linter**: `ruff check`.
- Follow **PEP 8** (enforced by ruff).
- Use **type hints** throughout.
- Prefer dataclasses or Pydantic models over raw dicts for structured data.
- Max line length: **100 characters**.

---

## Testing Requirements

### Frontend

Run the full test suite with:
```bash
bun run test
```

- Write Vitest unit tests for all utility functions in `src/lib/`.
- New React components should have at least smoke-test coverage.
- Do not lower overall coverage.

### Backend

Run with:
```bash
cd local_worker
pytest tests/ -v
```

Also run from the project root:
```bash
pytest tests/ -v
```

- New API endpoints require at least one integration test.
- New engine functions require unit tests.
- Tests must not make real network calls (use mocks/fixtures).

---

## Issue Templates

Use the GitHub issue templates:

- **Bug Report** — Provide steps to reproduce, expected vs. actual behaviour, environment info (OS, GPU, Python/Node versions).
- **Feature Request** — Describe the problem you're solving, your proposed solution, and alternatives considered.

For questions and discussions, please use **[GitHub Discussions](https://github.com/2024is008-source/clipforge-studio/discussions)** instead of Issues.

---

## License Agreement

By submitting a pull request, you agree that your contributions will be licensed
under the **MIT License** as described in [LICENSE](LICENSE).

Please note that this project uses **Ultralytics YOLOv8** (AGPL-3.0). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
