import numpy as np
import librosa
import os
from typing import Tuple, List, Optional


class FeatureExtractor:
    """
    Модуль извлечения спектрально-временных признаков для акустической классификации.
    Оптимизирован для высокочастотных записей (96 кГц) и выделения гармоник БПЛА.
    """

    def __init__(self, sr: int = 96000, n_mfcc: int = 20, n_deltas: int = 10):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_deltas = n_deltas
        # Размер окна (25 мс) и шаг (10 мс) - стандарт для аудиоанализа
        self.n_fft = int(0.025 * sr)
        self.hop_length = int(0.010 * sr)

    def extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Извлекает комбинированный вектор признаков: MFCC + Delta + Delta-Delta.
        Возвращает усредненный по времени вектор признаков для всего файла.
        """
        try:
            # Загрузка аудио с принудительной моно-конвертацией и сохранением исходной частоты
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True, res_type='kaiser_best')

            if len(y) < self.n_fft:
                print(f"⚠️ Файл слишком короткий: {audio_path}")
                return None

            # 1. MFCC (Mel-frequency cepstral coefficients)
            # Описывают спектральную огибающую (тембр сигнала)
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )

            # 2. Delta (Скорость изменения MFCC)
            # Важно для детекции модуляции винтов (нестационарность)
            deltas = librosa.feature.delta(mfccs)

            # 3. Delta-Delta (Ускорение изменения MFCC)
            delta_deltas = librosa.feature.delta(mfccs, order=2)

            # Конкатенация всех признаков
            features = np.vstack((mfccs, deltas, delta_deltas))

            # Статистическое агрегирование (Mean & Std) по временной оси
            # Превращает матрицу (признаки × время) в вектор фиксированной длины
            mean_features = np.mean(features, axis=1)
            std_features = np.std(features, axis=1)

            final_vector = np.hstack((mean_features, std_features))

            return final_vector

        except Exception as e:
            print(f"❌ Ошибка обработки файла {os.path.basename(audio_path)}: {e}")
            return None

    def process_directory(self, folder_path: str) -> Tuple[np.ndarray, List[str]]:
        """Обрабатывает все WAV файлы в папке."""
        features_list = []
        valid_files = []

        if not os.path.exists(folder_path):
            print(f"❌ Папка не найдена: {folder_path}")
            return np.array([]), []

        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        print(f"📂 Обработка папки: {os.path.basename(folder_path)} ({len(files)} файлов)...")

        for i, file in enumerate(files):
            full_path = os.path.join(folder_path, file)
            feats = self.extract_features(full_path)

            if feats is not None:
                features_list.append(feats)
                valid_files.append(file)

            # Индикация прогресса каждые 100 файлов
            if (i + 1) % 100 == 0:
                print(f"   ... обработано {i + 1}/{len(files)}")

        return np.array(features_list), valid_files
