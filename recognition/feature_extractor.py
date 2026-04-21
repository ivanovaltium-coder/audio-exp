import numpy as np
import librosa
from typing import Optional, Tuple, List
import os


class FeatureExtractor:
    def __init__(self, sr: int = 96000, n_mfcc: int = 20):
        self.sr = sr
        self.n_mfcc = n_mfcc
        # Параметры окна для FFT
        self.n_fft = int(0.025 * sr)  # 25 мс
        self.hop_length = int(0.010 * sr)  # 10 мс

    def extract_features_from_array(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Извлекает признаки из numpy массива."""
        try:
            if len(y) < self.n_fft:
                return None

            # 1. MFCC (20 коэффициентов)
            mfccs = librosa.feature.mfcc(
                y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )

            # 2. Delta (20 коэффициентов)
            deltas = librosa.feature.delta(mfccs)

            # 3. Delta-Delta (20 коэффициентов)
            delta_deltas = librosa.feature.delta(mfccs, order=2)

            # Объединяем все в одну матрицу (60 строк x время)
            features = np.vstack((mfccs, deltas, delta_deltas))

            # Агрегация по времени: Mean и Std для каждого из 60 коэффициентов
            # Итого: 60 * 2 = 120 признаков
            mean_feat = np.mean(features, axis=1)
            std_feat = np.std(features, axis=1)

            final_vector = np.hstack((mean_feat, std_feat))
            return final_vector

        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return None

    def extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """Загружает файл и извлекает признаки."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True, res_type='kaiser_best')
            return self.extract_features_from_array(y)
        except Exception as e:
            print(f"Ошибка файла {audio_path}: {e}")
            return None

    def process_directory(self, folder_path: str) -> Tuple[np.ndarray, List[str]]:
        """Обрабатывает папку с файлами."""
        features_list = []
        valid_files = []

        if not os.path.exists(folder_path):
            return np.array([]), []

        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        print(f"Обработка папки {folder_path} ({len(files)} файлов)...")

        for i, file in enumerate(files):
            full_path = os.path.join(folder_path, file)
            feats = self.extract_features(full_path)
            if feats is not None:
                features_list.append(feats)
                valid_files.append(file)

        return np.array(features_list), valid_files
