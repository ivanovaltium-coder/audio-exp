import numpy as np
import os
import config
from models.classifier import DroneClassifier
from recognition.feature_extractor import FeatureExtractor


class DroneRecognizer:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model_path = model_path or config.MODEL_PATH
        self.scaler_path = scaler_path or config.SCALER_PATH

        # Инициализация экстрактора признаков (MFCC + Delta)
        self.extractor = FeatureExtractor(sr=config.SAMPLE_RATE)

        # Загрузка модели классификации
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Скалер не найден: {self.scaler_path}")

        self.classifier = DroneClassifier.load(self.model_path, self.scaler_path)
        print(f"✅ Модель загружена: {os.path.basename(self.model_path)}")

    def predict(self, audio_data: np.ndarray) -> tuple:
        """
        Принимает многоканальный аудио сигнал, обрабатывает первый канал
        и возвращает предсказание.
        """
        # Если данных много каналов, берем первый (или можно усреднить)
        if audio_data.ndim > 1:
            mono_signal = audio_data[:, 0]
        else:
            mono_signal = audio_data

        # 1. Извлечение признаков (через отдельный класс)
        features = self.extractor.extract_features_from_array(mono_signal)

        if features is None:
            raise ValueError("Не удалось извлечь признаки из сигнала")

        # 2. Классификация
        prediction, confidence = self.classifier.predict(features)

        return prediction, confidence
