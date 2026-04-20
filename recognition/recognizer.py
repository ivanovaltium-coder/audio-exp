import numpy as np
import joblib
import config
from models.classifier import DroneClassifier


class DroneRecognizer:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model_path = model_path or config.MODEL_PATH
        self.scaler_path = scaler_path or config.SCALER_PATH

        # Загрузка модели и скалера
        self.classifier = DroneClassifier.load(self.model_path, self.scaler_path)
        print(f"✅ Модель загружена: {self.model_path}")

    def predict(self, audio_data: np.ndarray) -> tuple:
        """
        Принимает аудио данные (numpy array), извлекает признаки и возвращает предсказание.
        audio_data: 1D массив (моно)
        """
        if audio_data.ndim != 1:
            raise ValueError("Ожидается моно аудио сигнал (1D массив)")

        # Извлечение признаков (MFCC и др.)
        features = self.classifier.extract_features(audio_data, config.SAMPLE_RATE)

        # Классификация
        prediction, probability = self.classifier.predict(features)

        return prediction, probability