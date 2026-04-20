# config.py
"""
Файл конфигурации проекта.
Все основные параметры вынесены сюда для удобства.
"""

# Параметры аудио
SAMPLE_RATE = 48000  # Гц, частота дискретизации (рекомендуется 48000 для UR44C)
WINDOW_SEC = 3.0  # секунд, длина анализируемого окна
NUM_CHANNELS = 8  # количество каналов записи (8 для Steinberg UR44C: 4 микрофона + 2 линейных + 2 дубля)
USE_SINGLE_CHANNEL = True  # если True, используется только 1 канал (для первой версии проекта)
ACTIVE_CHANNEL = 0  # индекс активного канала (0-7), используется при USE_SINGLE_CHANNEL=True

# Параметры ASIO драйвера
USE_ASIO = True  # использовать ASIO драйвер для низкой задержки
ASIO_DEVICE_NAME = "Steinberg UR44C ASIO"  # имя устройства ASIO (точно как в системе)
ASIO_BUFFER_SIZE = 1024  # размер буфера ASIO (чем меньше, тем меньше задержка)

# Параметры MFCC
N_MFCC = 13  # количество коэффициентов MFCC
N_FFT = 2048  # размер окна БПФ
HOP_LENGTH = 512  # шаг между окнами

# Пути для сохранения модели
MODEL_PATH = 'models/saved_models/drone_classifier.joblib'
SCALER_PATH = 'models/saved_models/scaler.joblib'

# Имена классов (порядок важен!)
# 0 = фон, 1 = дрон
CLASSES = ['background', 'drone']

# Порог уровня сигнала для проверки работы микрофона (в дБ)
MIC_CHECK_THRESHOLD_DB = -40.0  # если сигнал громче этого порога, микрофон работает
