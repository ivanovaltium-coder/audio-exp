# audio/recorder.py
import sounddevice as sd
import numpy as np
import config


def record_audio(duration=None, samplerate=None, device=None):
    """
    Записывает аудио с микрофона.

    Параметры:
        duration: float — длительность записи в секундах.
                  Если None, используется config.WINDOW_SEC.
        samplerate: int — частота дискретизации.
                    Если None, используется config.SAMPLE_RATE.
        device: int или str — устройство ввода (по умолчанию None = системное по умолчанию).

    Возвращает:
        np.ndarray форма (N,) — аудиоданные с плавающей точкой в диапазоне [-1, 1].
    """
    if duration is None:
        duration = config.WINDOW_SEC
    if samplerate is None:
        samplerate = config.SAMPLE_RATE

    # Запись с микрофона
    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        device=device
    )
    sd.wait()  # ждём окончания записи
    return recording.flatten()


def list_devices():
    """Выводит список доступных аудиоустройств."""
    print(sd.query_devices())
    print("\nУстройство ввода по умолчанию:", sd.default.device[0])
