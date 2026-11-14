"""
Hardware-aware faster-whisper transcription engine.
Detects real CUDA GPU availability, GPU name, and VRAM; selects optimal compute_type (int8_float16 on RTX 2050,
int8 on CPU fallback); caches loaded models; and never claims GPU acceleration unless verified at runtime.
"""
import importlib.util
import os
import shutil
import subprocess
from typing import Any, Callable, TypedDict
from .config import MODELS_DIR, WHISPER_MODELS
from .speaker_processing import check_diarization_available


class HardwareInfo(TypedDict):
    whisper_installed: bool
    cuda_available: bool
    gpu_name: str | None
    vram_total_mb: int | None
    vram_free_mb: int | None
    recommended_model: str
    recommended_device: str
    recommended_compute_type: str
    cpu_fallback_available: bool
    cpu_threads: int
    active_device: str | None
    active_compute_type: str | None
    active_model: str | None
    diarization: dict


def recommended_cpu_threads() -> int:
    """CTranslate2 defaults to 4 threads, which leaves most of a CPU-only machine idle.

    int8 GEMM scales with physical cores; counting hyperthreads past that mostly adds contention.
    psutil ships with ultralytics but is not a hard requirement, so half the logical count is the
    fallback estimate.
    """
    try:
        import psutil  # type: ignore
        physical = psutil.cpu_count(logical=False)
        if physical:
            return max(1, min(16, int(physical)))
    except Exception:
        pass
    return max(1, min(16, (os.cpu_count() or 4) // 2))


def probe_nvidia_gpu() -> tuple[str | None, int | None, int | None]:
    """Query nvidia-smi for GPU name, total VRAM (MB), and free VRAM (MB)."""
    if not shutil.which("nvidia-smi"):
        return None, None, None
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=True,
        )
        line = res.stdout.decode("utf-8", errors="replace").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            return parts[0], int(float(parts[1])), int(float(parts[2]))
    except Exception:
        pass
    return None, None, None


def check_ctranslate2_cuda() -> bool:
    """Check whether ctranslate2 is installed and reports at least 1 CUDA device."""
    try:
        import ctranslate2  # type: ignore
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class WhisperEngine:
    def __init__(self) -> None:
        self._model: Any = None
        self._active_model_name: str | None = None
        self._active_device: str | None = None
        self._active_compute_type: str | None = None
        self._last_load_error: str | None = None

    def is_faster_whisper_installed(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def get_hardware_info(self) -> HardwareInfo:
        installed = self.is_faster_whisper_installed()
        gpu_name, vram_total, vram_free = probe_nvidia_gpu()
        cuda_ok = check_ctranslate2_cuda()

        if cuda_ok and (vram_total is None or vram_total >= 2000):
            rec_device = "cuda"
            rec_compute = "int8_float16"
            rec_model = "small" if (vram_total is None or vram_total <= 5000) else "medium"
        else:
            rec_device = "cpu"
            rec_compute = "int8"
            rec_model = "small"

        return {
            "whisper_installed": installed,
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total,
            "vram_free_mb": vram_free,
            "recommended_model": rec_model,
            "recommended_device": rec_device,
            "recommended_compute_type": rec_compute,
            "cpu_fallback_available": True,
            "cpu_threads": recommended_cpu_threads(),
            "active_device": self._active_device,
            "active_compute_type": self._active_compute_type,
            "active_model": self._active_model_name,
            "diarization": check_diarization_available(),
        }

    def list_models_status(self) -> list[dict]:
        hw = self.get_hardware_info()
        vram = hw["vram_total_mb"] or 0
        out = []
        for mod_id, spec in WHISPER_MODELS.items():
            cached_dir = os.path.join(MODELS_DIR, f"models--Systran--faster-whisper-{mod_id}")
            is_cached = os.path.isdir(cached_dir)
            vram_ok = (not hw["cuda_available"]) or vram == 0 or vram >= spec["min_vram_mb"]
            out.append(
                {
                    **spec,
                    "cached": is_cached,
                    "vram_compatible": vram_ok,
                    "recommended": mod_id == hw["recommended_model"],
                }
            )
        return out

    def load_model(
        self,
        model_name: str = "small",
        preferred_device: str = "auto",
        preferred_compute_type: str = "auto",
    ) -> tuple[str, str]:
        """
        Load faster-whisper model onto CUDA if available & requested, otherwise automatically
        fall back to CPU (`int8`). Returns `(actual_device, actual_compute_type)`.
        """
        if model_name not in WHISPER_MODELS:
            raise ValueError(f"Unsupported Whisper model '{model_name}'. Allowed: {list(WHISPER_MODELS.keys())}")

        if not self.is_faster_whisper_installed():
            raise RuntimeError("faster-whisper is not installed. Run: pip install -r local_worker/requirements.txt")

        from faster_whisper import WhisperModel  # type: ignore

        hw = self.get_hardware_info()
        target_device = preferred_device
        if target_device == "auto":
            target_device = "cuda" if hw["cuda_available"] else "cpu"

        if (
            self._model is not None
            and self._active_model_name == model_name
            and self._active_device == target_device
        ):
            return self._active_device, self._active_compute_type or "int8"

        if target_device == "cuda":
            compute = "int8_float16" if preferred_compute_type in ("auto", "int8_float16") else preferred_compute_type
            try:
                self._model = WhisperModel(
                    model_name,
                    device="cuda",
                    compute_type=compute,
                    download_root=MODELS_DIR,
                )
                self._active_model_name = model_name
                self._active_device = "cuda"
                self._active_compute_type = compute
                self._last_load_error = None
                return "cuda", compute
            except Exception as cuda_err:
                self._last_load_error = f"CUDA load failed ({cuda_err}); fell back to CPU int8."

        # CPU fallback path
        cpu_compute = "int8" if preferred_compute_type in ("auto", "int8_float16", "int8") else preferred_compute_type
        self._model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=cpu_compute,
            download_root=MODELS_DIR,
            cpu_threads=recommended_cpu_threads(),
        )
        self._active_model_name = model_name
        self._active_device = "cpu"
        self._active_compute_type = cpu_compute
        return "cpu", cpu_compute

    def transcribe_file(
        self,
        wav_path: str,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        condition_on_previous_text: bool = True,
        initial_prompt: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict:
        """
        Transcribe a single normalized WAV file using faster-whisper and return genuine word-level timestamps.
        """
        actual_device, actual_compute = self.load_model(model_name, device, compute_type)
        lang_arg = None if (not language or language == "auto") else language

        segments_gen, info = self._model.transcribe(
            wav_path,
            beam_size=max(1, min(10, int(beam_size))),
            word_timestamps=bool(word_timestamps),
            vad_filter=bool(vad_filter),
            language=lang_arg,
            condition_on_previous_text=bool(condition_on_previous_text),
            initial_prompt=initial_prompt or None,
        )

        raw_words: list[dict] = []
        full_text_parts: list[str] = []

        for seg in segments_gen:
            if should_cancel and should_cancel():
                raise InterruptedError("Transcription cancelled by user")
            seg_text = (getattr(seg, "text", "") or "").strip()
            if seg_text:
                full_text_parts.append(seg_text)
            seg_words = getattr(seg, "words", None)
            if seg_words:
                for w_obj in seg_words:
                    w_txt = (getattr(w_obj, "word", "") or "").strip()
                    if not w_txt:
                        continue
                    raw_words.append(
                        {
                            "word": w_txt,
                            "start": getattr(w_obj, "start", None),
                            "end": getattr(w_obj, "end", None),
                            "probability": getattr(w_obj, "probability", None),
                        }
                    )
            elif seg_text:
                # If word_timestamps was disabled or segment had no word objects, record segment with start/end
                tokens = seg_text.split()
                for idx, tok in enumerate(tokens):
                    raw_words.append(
                        {
                            "word": tok,
                            "start": getattr(seg, "start", None) if idx == 0 else None,
                            "end": getattr(seg, "end", None) if idx == len(tokens) - 1 else None,
                        }
                    )

        return {
            "text": " ".join(full_text_parts).strip(),
            "words": raw_words,
            "detected_language": getattr(info, "language", lang_arg or "en"),
            "language_probability": float(getattr(info, "language_probability", 1.0) or 1.0),
            "device": actual_device,
            "compute_type": actual_compute,
            "model": model_name,
            "fallback_notice": self._last_load_error,
        }


ENGINE = WhisperEngine()
