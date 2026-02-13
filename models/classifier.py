# models/classifier.py
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


class DroneClassifier:
    """
    Классификатор для детекции дронов на основе MFCC.
    Использует SVM с линейным ядром и стандартизацию признаков.
    """

    def __init__(self):
        self.model = SVC(kernel='linear', probability=True, class_weight='balanced')
        self.scaler = StandardScaler()

    def train(self, X, y):
        """
        Обучает классификатор.

        Параметры:
            X: np.ndarray, форма (n_samples, n_features) — признаки
            y: np.ndarray, форма (n_samples,) — метки (0 или 1)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        """
        Предсказывает класс и вероятности.

        Параметры:
            X: np.ndarray, форма (n_samples, n_features) — признаки

        Возвращает:
            preds: np.ndarray — предсказанные классы
            probs: np.ndarray — вероятности для каждого класса
        """
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        return preds, probs

    def save(self, model_path, scaler_path):
        """Сохраняет модель и scaler в файлы."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, model_path, scaler_path):
        """Загружает модель и scaler из файлов."""
        obj = cls()
        obj.model = joblib.load(model_path)
        obj.scaler = joblib.load(scaler_path)
        return obj
