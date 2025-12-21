from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_logestic_model():
    """
    Build and return the ML model.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )
    return model



def build_rf_model():
    """
    Build and return Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    return model
