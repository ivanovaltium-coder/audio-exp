import os

# ----------------- НАСТРОЙКИ АУДИО -----------------
SAMPLE_RATE = 48000  # Частота дискретизации (Гц)
AUDIO_FORMAT = 16  # Разрядность (16 бит). PyAudio использует константы, но здесь для удобства число.
# В коде recorder.py это преобразуется в pyaudio.paInt16

# Количество каналов для записи
# Для Steinberg UR44C через ASIO доступно до 6-8 входов в зависимости от настроек драйвера
NUM_CHANNELS = 4  # Задел на будущее: 4 микрофона

# Использование одного канала для текущей версии проекта
USE_SINGLE_CHANNEL = True  # Если True, используется только первый канал (индекс 0)
ACTIVE_CHANNEL = 0  # Индекс активного канала (0-3)

# Длительность записи по умолчанию (секунды)
RECORD_DURATION = 5.0

# Размер буфера (влияет на задержку и стабильность)
BUFFER_SIZE = 1024  # Меньше = меньше задержка, но выше нагрузка на CPU

# Название устройства ASIO (как оно отображается в системе)
# Оставьте пустым, чтобы использовать первое найденное ASIO устройство
ASIO_DEVICE_NAME = "Steinberg UR44C"

# ----------------- НАСТРОЙКИ ПРОВЕРКИ МИКРОФОНА -----------------
MIC_CHECK_THRESHOLD_DB = -40.0  # Порог уровня сигнала в дБ для определения работы микрофона

# ----------------- НАСТРОЙКИ МОДЕЛИ -----------------
# Пути к файлам модели и скалера
MODEL_PATH = os.path.join("models", "saved_models", "drone_classifier.joblib")
SCALER_PATH = os.path.join("models", "saved_models", "scaler.joblib")

# Параметры для извлечения признаков (MFCC и др.)
N_MFCC = 13  # Количество коэффициентов MFCC
HOP_LENGTH = 512  # Шаг окна для анализа
N_FFT = 2048  # Размер окна БПФ

# ----------------- НАСТРОЙКИ ДАННЫХ -----------------
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Создание директорий при необходимости
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
