# models/train.py
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
    Загружает все wav-файлы из папок no_drone/ и drone/,
    извлекает признаки и возвращает X, y.
    """
    X, y = [], []

    # класс 0: фон (no_drone)
    for file in glob.glob(os.path.join(data_dir, 'no_drone', '*.wav')):
        try:
            audio, sr = librosa.load(file, sr=config.SAMPLE_RATE)
            feats = extract_features(audio, sr)
            X.append(feats)
            y.append(0)
        except Exception as e:
            print(f"Ошибка при обработке {file}: {e}")

    # класс 1: дрон (drone)
    for file in glob.glob(os.path.join(data_dir, 'drone', '*.wav')):
        try:
            audio, sr = librosa.load(file, sr=config.SAMPLE_RATE)
            feats = extract_features(audio, sr)
            X.append(feats)
            y.append(1)
        except Exception as e:
            print(f"Ошибка при обработке {file}: {e}")

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description='Обучение классификатора дронов')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Путь к папке с подпапками no_drone/ и drone/')
    args = parser.parse_args()

    print("Загрузка датасета...")
    X, y = load_dataset(args.data_dir)
    print(f"Загружено {len(X)} сэмплов: {np.sum(y == 0)} фона, {np.sum(y == 1)} дронов.")

    if len(X) == 0:
        print("Нет данных для обучения. Проверьте путь к data_dir.")
        return

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучаем классификатор
    clf = DroneClassifier()
    clf.train(X_train, y_train)

    # Оценка на тесте
    _, probs = clf.predict(X_test)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_test)
    print(f"Точность на тестовой выборке: {accuracy:.3f}")

    # Сохраняем модель
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    clf.save(config.MODEL_PATH, config.SCALER_PATH)
    print(f"Модель сохранена в {config.MODEL_PATH}")
    print(f"Scaler сохранён в {config.SCALER_PATH}")


if __name__ == '__main__':
    main()
