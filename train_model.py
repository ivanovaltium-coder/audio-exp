import os
import numpy as np
import sys

# Добавляем корень проекта в путь, чтобы работали импорты
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recognition.feature_extractor import FeatureExtractor
from models.classifier import DroneClassifier
import config


def main():
    print("--- НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ ---")

    raw_dir = 'data/raw'
    drone_dir = os.path.join(raw_dir, 'drone')
    not_drone_dir = os.path.join(raw_dir, 'not_drone')

    # Проверка наличия папок
    if not os.path.exists(drone_dir) or not os.path.exists(not_drone_dir):
        print(f"❌ Ошибка: Не найдены папки с данными!")
        print(f"   Создайте '{drone_dir}' и поместите туда записи дронов (.wav)")
        print(f"   Создайте '{not_drone_dir}' и поместите туда фоновые шумы (.wav)")
        print("\n⚠️ Запуск в режиме ожидания данных...")
        return

    extractor = FeatureExtractor(sr=config.SAMPLE_RATE, n_mfcc=20)
    classifier = DroneClassifier()

    print(f"\n📂 Обработка папки DRONE: {drone_dir}")
    X_drone_list, files_drone = extractor.process_directory(drone_dir)

    if len(X_drone_list) == 0:
        print("❌ Нет файлов дрона для обучения!")
        return
    X_drone = np.array(X_drone_list)
    y_drone = np.ones(len(X_drone))

    print(f"\n📂 Обработка папки NOT DRONE: {not_drone_dir}")
    X_not_drone_list, files_not_drone = extractor.process_directory(not_drone_dir)

    if len(X_not_drone_list) == 0:
        print("❌ Нет файлов шума для обучения!")
        return
    X_not_drone = np.array(X_not_drone_list)
    y_not_drone = np.zeros(len(X_not_drone))

    # Объединение данных
    X = np.vstack((X_drone, X_not_drone))
    y = np.hstack((y_drone, y_not_drone))

    print(f"\n✅ Всего образцов: {len(X)} (Дроны: {len(X_drone)}, Шумы: {len(X_not_drone)})")

    classifier.train(X, y)

    print("\n💾 Сохранение модели...")
    # Убедимся, что пути совпадают с теми, что в config.py
    classifier.save(config.MODEL_PATH, config.SCALER_PATH)

    print("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО ---")
    print(f"Теперь можно запустить: python main.py --device 1")


if __name__ == "__main__":
    main()
