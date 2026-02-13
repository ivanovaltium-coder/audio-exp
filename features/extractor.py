# features/extractor.py
import librosa
import numpy as np
import config


def extract_features(audio, sr):
    """
    Извлекает MFCC + дельты и дельты-дельты из аудиосигнала.

    Параметры:
        audio: np.ndarray, форма (N,) — аудиоданные
        sr: int — частота дискретизации

    Возвращает:
        np.ndarray форма (39,) — объединённый вектор признаков
    """
    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)  # усредняем по времени

    # Дельта (скорость изменения)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Дельта-дельта (ускорение)
    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2, axis=1)

    # Объединяем
    features = np.concatenate([mfcc_mean, delta_mean, delta2_mean])
    return features
