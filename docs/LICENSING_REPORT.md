# ClipForge — Comprehensive Licensing & Commercial Readiness Report

## Executive Summary
This report audits all external libraries, frameworks, machine learning models, fonts, and binaries integrated into or used by ClipForge. It defines the legal obligations, commercial deployment conditions, and copyleft boundaries for each component.

---

## 1. Component Audit Table

| Component | Source / Author | License | Commercial Use Allowed? | Copyleft / Source Disclosure? | Notes & Obligations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ultralytics YOLO** | Ultralytics Inc. | **AGPL-3.0** (or Enterprise) | Yes (with conditions) | **YES (Strong Copyleft)** | AGPL-3.0 requires source disclosure if modified or accessed over network. Commercial closed-source SaaS requires an Ultralytics Enterprise License. |
| **YOLOv8 Weights** | Ultralytics Inc. | **AGPL-3.0** | Yes (with license) | **YES** | Pretrained weights follow Ultralytics licensing terms. |
| **SAM 2 (Code)** | Meta AI | **Apache-2.0** | **YES** | NO | Permissive open source; requires license and copyright notice. |
| **SAM 2 Weights** | Meta AI | **Apache-2.0** | **YES** | NO | Official SAM 2.1 checkpoints are released under Apache-2.0. |
| **faster-whisper** | SYSTRAN / Guillaume Klein | **MIT** | **YES** | NO | Permissive; retain copyright notice. |
| **CTranslate2** | OpenNMT / SYSTRAN | **MIT** | **YES** | NO | Permissive; C++ inference engine. |
| **OpenCV** | OpenCV Team | **Apache-2.0** | **YES** | NO | Permissive (v4.5+); retain copyright notice. |
| **YuNet Model** | Shiqi Yu / OpenCV Zoo | **MIT** | **YES** | NO | Fast face detection model in ONNX format. |
| **FastAPI / Uvicorn** | Sebastián Ramírez / Encode | **MIT** | **YES** | NO | Standard permissive backend stack. |
| **ClipForge Core Engine** | Minuja Ranthika Amarasekara | **Proprietary / First-Party** | **YES** | NO | First-party Reframe Engine v2, Caption Studio, and Local Worker orchestration. |
| **Montserrat Font** | Julieta Ulanovsky | **OFL-1.1** | **YES** | NO | SIL Open Font License allows commercial bundling and embedding. |
| **FFmpeg (Installed)** | FFmpeg Team / Gyan.dev | **GPL-3.0** (full build) | **YES** (with source) | **YES (for redistribution)** | Distributing the GPL-compiled binary requires source redistribution. Using system FFmpeg via CLI does not make ClipForge GPL. |

---

## 2. Deep Dive: Ultralytics AGPL-3.0 Obligations

### The Legal Context
Ultralytics transitioned from GPL to **AGPL-3.0** (GNU Affero General Public License v3.0). Under Section 13 of the AGPL, if an application runs on a network server and interacts with users remotely, the complete corresponding source code must be made available to those users under AGPL-3.0 terms.

### Impact on ClipForge
1. **Current Local Architecture (Sidecar Process)**:
   In ClipForge's default architecture, the Python worker (`local_worker`) runs entirely on the **user's local workstation** as a sidecar utility. It is not hosted as a multi-tenant cloud service.
2. **Proprietary Cloud Commercialization**:
   If an organization chooses to host ClipForge as a commercial multi-tenant SaaS application without disclosing their proprietary frontend and business logic, they **must**:
   - Acquire an **Ultralytics Enterprise Commercial License** directly from Ultralytics; OR
   - Replace the YOLO object detector with an Apache-2.0 alternative such as **RT-DETR (Baidu/Paddle)** or **YOLO-NAS (Deci AI)** or **MediaPipe Object Detector**.

---

## 3. SAM 2 (Segment Anything Model 2) Terms

- Meta AI published Segment Anything 2 under the **Apache-2.0** license.
- Both the code repository (`facebookresearch/sam2`) and the official checkpoints (`sam2.1_hiera_*.pt`) are licensed under Apache-2.0.
- Commercial usage is fully permitted without royalty fees, provided the Apache-2.0 license and Meta copyright notice are included.
- In ClipForge, SAM 2 is implemented as an **isolated optional service**. If SAM 2 is not installed, ordinary video export and editing remain 100% operational.

---

## 4. FFmpeg Redistribution Requirements

- The currently installed FFmpeg binary (version 9.0 by Gyan.dev) is built with:
  `--enable-gpl --enable-version3 --enable-libx264 --enable-libass --enable-nvenc`
- **Subprocess Invocation**: ClipForge interacts with FFmpeg exclusively via command-line subprocesses (`subprocess.Popen` / `subprocess.run`), keeping the Python code legally independent from FFmpeg's internals.
- **Distribution**: If distributing ClipForge with FFmpeg pre-bundled:
  - If using a GPL build, the exact corresponding source code of FFmpeg and its GPL dependencies must be made available.
  - Alternatively, users can be instructed to install FFmpeg independently, or a non-GPL (LGPL) build can be distributed.

---

## 5. Summary Checklist for Commercial Production

- [x] All third-party attribution notices consolidated in `THIRD_PARTY_NOTICES.md`.
- [x] Font licenses verified (Montserrat OFL-1.1).
- [x] YOLO AGPL-3.0 boundary isolated within local worker.
- [x] SAM 2 optional dependency isolation verified.
- [x] FFmpeg CLI subprocess separation maintained.
- [x] Commercial path for SaaS documented (Ultralytics Enterprise or Apache-2.0 alternatives).

