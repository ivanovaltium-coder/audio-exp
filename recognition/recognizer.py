# recognition/recognizer.py
import numpy as np
import librosa
from features.extractor import extract_features
from models.classifier import DroneClassifier
import config


class DroneRecognizer:
    """
    Основной класс для распознавания дронов.
    Загружает обученную модель и scaler, может работать с файлами или микрофоном.
    """

    def __init__(self):
        """Инициализация: загрузка модели и scaler."""
        self.classifier = DroneClassifier.load(config.MODEL_PATH, config.SCALER_PATH)

    def recognize_file(self, file_path):
        """
        Распознаёт дрон в аудиофайле.

        Параметры:
            file_path: str — путь к WAV-файлу.

        Возвращает:
            class_name: str — название класса ('background' или 'drone')
            confidence: float — уверенность (0..1)
        """
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        features = extract_features(audio, sr).reshape(1, -1)
        preds, probs = self.classifier.predict(features)
        class_name = config.CLASSES[preds[0]]
        confidence = np.max(probs[0])
        return class_name, confidence

    def recognize_stream(self, callback, device=None):
        """
        Запускает непрерывное распознавание с микрофона.
        Для каждого окна длительностью config.WINDOW_SEC вызывает callback.

        Параметры:
            callback: функция, принимающая (class_name, confidence).
            device: индекс устройства микрофона (если нужно выбрать конкретный).
        """
        from audio.recorder import record_audio

        print(f"Прослушивание... (окно {config.WINDOW_SEC} сек, частота {config.SAMPLE_RATE} Гц)")
        if device is not None:
            print(f"Используется устройство ввода: {device}")
        print("Нажмите Ctrl+C для остановки.")

        try:
            while True:
                audio = record_audio(
                    duration=config.WINDOW_SEC,
                    samplerate=config.SAMPLE_RATE,
                    device=device
                )
                features = extract_features(audio, config.SAMPLE_RATE).reshape(1, -1)
                preds, probs = self.classifier.predict(features)
                class_name = config.CLASSES[preds[0]]
                confidence = np.max(probs[0])
                callback(class_name, confidence)
        except KeyboardInterrupt:
            print("\nОстановлено по запросу пользователя.")
        except Exception as e:
            print(f"\nОшибка при работе с микрофоном: {e}")

    def list_audio_devices(self):
        """Выводит список доступных аудиоустройств (для справки)."""
        import sounddevice as sd
        print(sd.query_devices())
        print("\nУстройство ввода по умолчанию:", sd.default.device[0])
