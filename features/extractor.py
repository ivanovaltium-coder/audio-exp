# features/extractor.py
import librosa
import numpy as np
import config


def extract_features(audio, sr):
    """
    Извлекает MFCC, дельты и дельты-дельты из аудиосигнала.

    Параметры:
        audio: np.ndarray, форма (N,) — аудиоданные в диапазоне [-1, 1]
        sr: int — частота дискретизации (должна совпадать с config.SAMPLE_RATE)

    Возвращает:
        np.ndarray форма (39,) — объединённый вектор признаков:
            первые 13: MFCC (средние по времени)
            следующие 13: дельта MFCC
            последние 13: дельта-дельта MFCC
    """
    # Вычисляем MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    # Усредняем по времени (получаем 13 значений)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Дельта (первая производная по времени)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Дельта-дельта (вторая производная)
    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2, axis=1)

    # Объединяем все признаки в один вектор
    features = np.concatenate([mfcc_mean, delta_mean, delta2_mean])
    return features
