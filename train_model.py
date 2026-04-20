"""
Скрипт для обучения модели классификации дронов.
Пока использует синтетические данные для проверки пайплайна.
В будущем здесь будет загрузка реальных датасетов.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Настройки
OUTPUT_DIR = "models/saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "drone_classifier.joblib")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

def generate_synthetic_data(samples=1000):
    """
    Генерирует синтетические данные для теста.
    Класс 0: Не дрон (шум, речь, ветер)
    Класс 1: Дрон (характерные частоты винтов)
    """
    print("Генерация синтетических данных...")
    np.random.seed(42)
    
    # Признаки: [mean_freq, std_freq, spectral_centroid, zero_crossing_rate, rms_energy]
    # Имитируем, что у дронов выше средняя частота и стабильнее спектр
    X_drones = np.random.normal(loc=[2000, 500, 2500, 100, 0.5], 
                                scale=[200, 100, 300, 20, 0.1], 
                                size=(samples // 2, 5))
    y_drones = np.ones(samples // 2)
    
    X_noise = np.random.normal(loc=[800, 800, 1000, 300, 0.3], 
                               scale=[300, 200, 400, 100, 0.2], 
                               size=(samples // 2, 5))
    y_noise = np.zeros(samples // 2)
    
    X = np.vstack([X_drones, X_noise])
    y = np.hstack([y_drones, y_noise])
    
    return X, y

def train():
    print("--- НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ ---")
    
    # 1. Подготовка данных
    X, y = generate_synthetic_data(samples=2000)
    
    # 2. Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Нормализация признаков (важно для аудио)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Обучение модели (Random Forest как базовый вариант)
    print("Обучение модели Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 5. Оценка качества
    y_pred = clf.predict(X_test_scaled)
    print("\nРезультаты на тестовой выборке:")
    print(classification_report(y_test, y_pred, target_names=['Not Drone', 'Drone']))
    
    # 6. Сохранение модели и скалера
    print(f"\nСохранение модели в {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    
    print(f"Сохранение скалера в {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)
    
    print("--- ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО ---")
    print("Теперь можно запустить: python main.py --check-mic")

if __name__ == "__main__":
    train()
