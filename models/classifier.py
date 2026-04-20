import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
from typing import Tuple, List  # <--- ДОБАВЛЕНО


class DroneClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        if len(X) == 0:
            raise ValueError("Нет данных для обучения!")

        print(f"Нормализация данных ({len(X)} образцов)...")
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Обучение модели Random Forest...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        print("\nРезультаты на тестовой выборке:")
        print(classification_report(y_test, y_pred, target_names=['Not Drone', 'Drone']))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        return self.model

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        if not self.is_trained:
            raise RuntimeError("Модель не обучена!")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = np.max(self.model.predict_proba(features_scaled)[0])

        return prediction, probability

    def save(self, model_path: str, scaler_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Модель сохранена: {model_path}")
        print(f"Скалер сохранен: {scaler_path}")

    @staticmethod
    def load(model_path: str, scaler_path: str) -> 'DroneClassifier':
        obj = DroneClassifier()
        obj.model = joblib.load(model_path)
        obj.scaler = joblib.load(scaler_path)
        obj.is_trained = True
        return obj
