import io
import logging
import threading

from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf

SPEAKERS = {"serena", "vivian", "aiden", "dylan", "eric", "ono_anna", "ryan", "sohee", "uncle_fu"}
DEFAULT_SPEAKER = "serena"

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                logging.getLogger().info(f"Loading Qwen3 TTS model on {device}...")
                _model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    device_map=device,
                    dtype=torch.float32,
                )
                logging.getLogger().info("Qwen3 TTS model loaded")
    return _model


def get_default_speaker(language: str) -> str:
    return DEFAULT_SPEAKER


def validate_speaker(speaker: str, language: str):
    if speaker not in SPEAKERS:
        raise ValueError(
            f"Speaker '{speaker}' is not available. "
            f"Available: {sorted(SPEAKERS)}"
        )


def synthesize(text: str, speaker: str, language: str) -> bytes:
    logging.getLogger().info(
        f"Synthesizing: language={language}, speaker={speaker}, text_length={len(text)}, text={text}"
    )

    model = get_model()

    wavs, sr = model.generate_custom_voice(
        text=text,
        language="Russian" if language == "ru" else "English",
        speaker=speaker,
    )

    wav_buf = io.BytesIO()
    sf.write(wav_buf, wavs[0], sr, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)

    segment = AudioSegment.from_wav(wav_buf)
    ogg_buf = io.BytesIO()
    segment.export(ogg_buf, format="ogg", codec="libopus", parameters=["-vbr", "on"])
    ogg_buf.seek(0)
    return ogg_buf.read()
