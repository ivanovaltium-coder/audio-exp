# config.py
"""
Файл конфигурации проекта.
Все основные параметры вынесены сюда для удобства.
"""

# Параметры аудио
SAMPLE_RATE = 22050          # Гц, частота дискретизации для загрузки и записи
WINDOW_SEC = 3.0             # секунд, длина анализируемого окна

# Параметры MFCC
N_MFCC = 13                  # количество коэффициентов MFCC
N_FFT = 2048                 # размер окна БПФ
HOP_LENGTH = 512             # шаг между окнами

# Пути для сохранения модели
MODEL_PATH = 'models/saved_models/drone_classifier.joblib'
SCALER_PATH = 'models/saved_models/scaler.joblib'

# Имена классов (порядок важен!)
# 0 = фон, 1 = дрон
CLASSES = ['background', 'drone']