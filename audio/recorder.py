# audio/recorder.py
import sounddevice as sd
import numpy as np
import config


def record_audio(duration=None, samplerate=None):
    """
    Записывает аудио с микрофона по умолчанию.

    Параметры:
        duration: float — длительность записи в секундах.
                  Если None, используется config.WINDOW_SEC.
        samplerate: int — частота дискретизации.
                    Если None, используется config.SAMPLE_RATE.

    Возвращает:
        np.ndarray форма (N,) — аудиоданные с плавающей точкой в диапазоне [-1, 1].
    """
    if duration is None:
        duration = config.WINDOW_SEC
    if samplerate is None:
        samplerate = config.SAMPLE_RATE

    recording = sd.rec(int(duration * samplerate),
                       samplerate=samplerate,
                       channels=1,
                       dtype='float32')
    sd.wait()  # ждём окончания записи
    return recording.flatten()
