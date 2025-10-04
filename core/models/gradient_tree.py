from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class GradientBoostTreeClassifier:
    def __init__(self, **kwargs):
        self.model = GradientBoostingClassifier(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        return acc, report

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path="models/gbdt_model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="models/gbdt_model.pkl"):
        self.model = joblib.load(path)
