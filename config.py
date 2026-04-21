# Основные настройки аудио
SAMPLE_RATE = 96000  # Частота дискретизации (строго 96к для UR44C в ASIO)
NUM_CHANNELS = 8  # Количество каналов (4 микрофона + 2 линейных + дубли)
RECORD_DURATION = 5.0  # Длительность записи по умолчанию
BUFFER_SIZE = 4096  # Размер буфера (как в рабочем примере)

# Пути к моделям
MODEL_PATH = 'models/saved_models/drone_classifier.joblib'
SCALER_PATH = 'models/saved_models/drone_scaler.joblib'

# Порог проверки микрофона (дБ)
MIC_CHECK_THRESHOLD_DB = -40

# Имя устройства (используется для подсказок, основной поиск идет по ASIO API)
ASIO_DEVICE_NAME = "Steinberg"
