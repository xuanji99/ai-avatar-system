"""
Speech-to-text service backed by faster-whisper.

The Whisper model is several hundred MB and takes 30–60 s to load on a cold
start. We defer loading until the first transcription so FastAPI's lifespan
hook stays fast and the /health endpoint becomes available promptly.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional, Union

import numpy as np
import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self):
        self.provider = settings.STT_PROVIDER
        self.model_name = settings.WHISPER_MODEL
        self.model = None
        # Lock ensures the model is loaded exactly once even under burst load.
        self._load_lock: Optional[asyncio.Lock] = None

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _build_model(self):
        """Synchronous model load — run inside a thread to avoid blocking the loop."""
        from faster_whisper import WhisperModel
        device = "cuda" if self._check_cuda() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Loading Whisper model {self.model_name!r} on {device} ({compute_type})…")
        model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded")
        return model

    async def initialize(self) -> None:
        """Eager warm-up. Optional — `transcribe` will load on first call too."""
        if self.model is not None or self.provider != "whisper":
            return
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            if self.model is None:
                self.model = await asyncio.to_thread(self._build_model)

    async def transcribe(self, audio_data: Union[bytes, str], language: str = "en") -> str:
        if self.provider != "whisper":
            raise ValueError(f"Unsupported STT provider: {self.provider}")
        if self.model is None:
            await self.initialize()
        return await asyncio.to_thread(self._transcribe_sync, audio_data, language)

    def _transcribe_sync(self, audio_data: Union[bytes, str], language: str) -> str:
        try:
            if isinstance(audio_data, bytes):
                audio, sample_rate = sf.read(io.BytesIO(audio_data))
            else:
                audio, sample_rate = sf.read(audio_data)

            # Mono mixdown
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample to 16 kHz (Whisper's expected rate)
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            audio = audio.astype(np.float32)

            assert self.model is not None  # for type checker
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            transcription = " ".join(seg.text for seg in segments).strip()
            logger.info(f"Transcribed {len(transcription)} chars (lang={info.language})")
            return transcription

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            raise


stt_service = STTService()
