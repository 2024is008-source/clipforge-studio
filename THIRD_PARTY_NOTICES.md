# Third-Party Notices

ClipForge Studio incorporates components from open-source projects. Their licenses
are listed below. Full license texts are available in each project's repository.

---

## ⚠️ Important — AGPL-3.0 Notice (Ultralytics YOLOv8)

**Ultralytics YOLOv8** is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
This has significant implications:

- If you distribute ClipForge Studio (modified or unmodified) **as a service over a network**, you
  are required to make the **complete corresponding source code** available to users.
- If you distribute binary copies, recipients must receive or be able to obtain the source code.
- Ultralytics offers a **commercial license** as an alternative. See: https://ultralytics.com/license

If you are deploying ClipForge Studio in a commercial or SaaS context, consult a legal professional
regarding your AGPL-3.0 obligations or obtain a commercial Ultralytics license.

---

## Frontend Dependencies

| Package | License | URL |
|---|---|---|
| React | MIT | https://github.com/facebook/react |
| TanStack Router / Start | MIT | https://github.com/TanStack/router |
| TanStack Query | MIT | https://github.com/TanStack/query |
| Tailwind CSS v4 | MIT | https://github.com/tailwindlabs/tailwindcss |
| Vite | MIT | https://github.com/vitejs/vite |
| TypeScript | Apache-2.0 | https://github.com/microsoft/TypeScript |
| @ffmpeg/ffmpeg | LGPL-2.1 | https://github.com/ffmpegwasm/ffmpeg.wasm |
| Supabase JS | Apache-2.0 | https://github.com/supabase/supabase-js |
| Radix UI (shadcn/ui) | MIT | https://github.com/radix-ui/primitives |
| Lucide React | ISC | https://github.com/lucide-icons/lucide |
| Framer Motion | MIT | https://github.com/framer/motion |
| Sonner | MIT | https://github.com/emilkowalski/sonner |
| Zod | MIT | https://github.com/colinhacks/zod |
| React Hook Form | MIT | https://github.com/react-hook-form/react-hook-form |
| Drizzle ORM | Apache-2.0 | https://github.com/drizzle-team/drizzle-orm |
| Vitest | MIT | https://github.com/vitest-dev/vitest |
| ESLint | MIT | https://github.com/eslint/eslint |
| Prettier | MIT | https://github.com/prettier/prettier |

---

## Backend Dependencies

| Package | License | URL |
|---|---|---|
| FastAPI | MIT | https://github.com/tiangolo/fastapi |
| Uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |
| Pydantic | MIT | https://github.com/pydantic/pydantic |
| faster-whisper | MIT | https://github.com/SYSTRAN/faster-whisper |
| OpenCV (opencv-python) | Apache-2.0 | https://github.com/opencv/opencv |
| **Ultralytics YOLOv8** | **AGPL-3.0** ⚠️ | https://github.com/ultralytics/ultralytics |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| FFmpeg | LGPL-2.1 / GPL-2.0+ | https://ffmpeg.org/legal.html |
| MediaPipe | Apache-2.0 | https://github.com/google-ai-edge/mediapipe |
| google-generativeai | Apache-2.0 | https://github.com/google-gemini/generative-ai-python |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| numpy | BSD-3-Clause | https://github.com/numpy/numpy |
| Pillow | HPND | https://github.com/python-pillow/Pillow |
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| pytest | MIT | https://github.com/pytest-dev/pytest |

---

## Notes on FFmpeg

FFmpeg is linked as an external binary and is **not statically compiled** into ClipForge Studio.
FFmpeg itself is licensed under **LGPL-2.1+** (with optional GPL components depending on build flags).
When distributing FFmpeg binaries, you must comply with the LGPL or GPL terms as applicable.
See: https://ffmpeg.org/legal.html

---

## Notes on LGPL Dependencies (@ffmpeg/ffmpeg, FFmpeg)

The LGPL-2.1 license permits use in proprietary software provided that:
- The LGPL-licensed component can be replaced by the user (dynamic linking or similar).
- Proper attribution is given.

---

*Last updated: 2026-11-14*
