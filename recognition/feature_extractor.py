import numpy as np
import librosa
import os
from typing import Tuple, List


class FeatureExtractor:
    def __init__(self, n_mfcc=13, sr=96000):
        self.n_mfcc = n_mfcc
        self.sr = sr

    def extract_features(self, audio_path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            return mfccs_mean
        except Exception as e:
            print(f"Ошибка при обработке файла {audio_path}: {e}")
            return None

    def process_directory(self, folder_path: str) -> Tuple[List[np.ndarray], List[str]]:
        features_list = []
        valid_files = []

        if not os.path.exists(folder_path):
            print(f"Папка не найдена: {folder_path}")
            return [], []

        files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.WAV'))]
        print(f"Найдено файлов в {folder_path}: {len(files)}")

        for file in files:
            full_path = os.path.join(folder_path, file)
            feats = self.extract_features(full_path)

            if feats is not None:
                features_list.append(feats)
                valid_files.append(file)
            else:
                print(f"Пропущен файл: {file}")

        return features_list, valid_files
