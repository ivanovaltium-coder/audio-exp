# Основные настройки
SAMPLE_RATE = 96000  # Нативная частота UR44C для высокого качества
NUM_CHANNELS = 8  # Полное количество каналов (4 микрофона + 2 линейных + дубли)
RECORD_DURATION = 5.0  # Длительность записи по умолчанию
BUFFER_SIZE = 1024  # Размер буфера (меньше = меньше задержка, но выше нагрузка)

# Настройки модели
MODEL_PATH = 'models/saved_models/drone_classifier.joblib'
SCALER_PATH = 'models/saved_models/drone_scaler.joblib'

# Настройки проверки микрофона
MIC_CHECK_THRESHOLD_DB = -40  # Порог в дБ

# Имя устройства (опционально, можно оставить пустым для автопоиска)
ASIO_DEVICE_NAME = "Steinberg"

# Формат аудио (для совместимости с numpy используем 16 бит в коде, но можно настроить)
AUDIO_FORMAT = 'int16'
