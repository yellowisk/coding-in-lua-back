from sklearn.model_selection import train_test_split
from models.gradient_tree import GradientBoostTreeClassifier
from data.preprocessing import prepare_data

def train_gbdt(dataset_path: str):
    X, y = prepare_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gbdt = GradientBoostTreeClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    gbdt.train(X_train, y_train)
    acc, report = gbdt.evaluate(X_test, y_test)
    print(f"Accuracy: {acc:.4f}")
    print(report)

    gbdt.save()
    return gbdt
