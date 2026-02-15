# models/train.py    python -m models.train --data_dir data/raw/ запуск обучения
import os
import argparse
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from features.extractor import extract_features
from models.classifier import DroneClassifier
import config


def load_dataset(data_dir):
    """
    Загружает все WAV-файлы из подпапок no_drone/ и drone/,
    извлекает признаки и возвращает матрицу X и вектор меток y.
    """
    X = []
    y = []

    # Класс 0: фон (no_drone)
    no_drone_path = os.path.join(data_dir, 'no_drone', '*.wav')
    for file_path in glob.glob(no_drone_path):
        try:
            audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
            features = extract_features(audio, sr)
            X.append(features)
            y.append(0)
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")

    # Класс 1: дрон (drone)
    drone_path = os.path.join(data_dir, 'drone', '*.wav')
    for file_path in glob.glob(drone_path):
        try:
            audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
            features = extract_features(audio, sr)
            X.append(features)
            y.append(1)
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description='Обучение классификатора дронов')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Путь к папке с подпапками no_drone/ и drone/')
    args = parser.parse_args()

    print("Загрузка датасета...")
    X, y = load_dataset(args.data_dir)

    if len(X) == 0:
        print("Не найдено ни одного аудиофайла. Проверьте содержимое data_dir.")
        return

    n_no_drone = np.sum(y == 0)
    n_drone = np.sum(y == 1)
    print(f"Загружено {len(X)} сэмплов: {n_no_drone} фона, {n_drone} дронов.")

    # Разделение на обучающую и тестовую выборки (80/20) с сохранением пропорций классов
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучение классификатора
    clf = DroneClassifier()
    clf.train(X_train, y_train)

    # Оценка на тестовой выборке
    preds, probs = clf.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Точность на тестовой выборке: {accuracy:.3f}")

    # Дополнительно: выведем уверенность для правильно и неправильно классифицированных
    if len(y_test) > 0:
        print("\nПримеры предсказаний:")
        for i in range(min(5, len(y_test))):
            true_class = config.CLASSES[y_test[i]]
            pred_class = config.CLASSES[preds[i]]
            confidence = np.max(probs[i])
            print(f"  Истина: {true_class}, предсказание: {pred_class}, уверенность: {confidence:.3f}")

    # Создание папки для сохранения модели, если её нет
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # Сохранение модели и scaler
    clf.save(config.MODEL_PATH, config.SCALER_PATH)
    print(f"\nМодель сохранена в {config.MODEL_PATH}")
    print(f"Scaler сохранён в {config.SCALER_PATH}")


if __name__ == '__main__':
    main()
