import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import joblib
import os

def calculate_eer(y_true, y_scores):
    """
    Calculates the Equal Error Rate (EER) for binary classification.
    EER is the point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR).
    
    y_true: Ground truth labels (0 for Genuine, 1 for Deepfake)
    y_scores: Predicted probabilities of the positive class (1: Deepfake)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    try:
        # Find the intersection of FPR and FNR (where FPR = FNR)
        # We find x where x = fpr and y = fnr = 1 - tpr = x.
        # This is equivalent to finding where 1 - x - tpr(x) = 0.
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        # Fallback to closest point if interpolation fails
        idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        
    return eer

def train_classifier(X_train, y_train, n_estimators=100, max_depth=5, min_samples_split=6, min_samples_leaf=4, random_state=42):
    """
    Trains a regularized Random Forest Classifier to prevent overfitting.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluates the model on test data and returns key metrics:
    Accuracy, EER, F1-Score, Confusion Matrix, and Per-Class Accuracies.
    """
    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of Deepfake (class 1)
    
    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # 2. F1-Score
    f1 = f1_score(y_test, y_pred)
    
    # 3. EER
    eer = calculate_eer(y_test, y_prob)
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 5. Per-class Accuracy
    # Genuine accuracy (class 0)
    genuine_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Deepfake accuracy (class 1)
    deepfake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "eer": eer,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "per_class_accuracy": {
            "genuine": genuine_acc,
            "deepfake": deepfake_acc
        }
    }
    
    return metrics, y_pred, y_prob

def save_model(model, file_path):
    """
    Saves the trained model to a file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved successfully to: {file_path}")

def load_model(file_path):
    """
    Loads a trained model from a file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at: {file_path}")
    return joblib.load(file_path)

if __name__ == "__main__":
    # Test classifier training and EER calculation
    print("Testing model module...")
    X = np.random.randn(100, 20)
    # Class 0: Genuine, Class 1: Deepfake
    y = np.random.randint(0, 2, 100)
    
    model = train_classifier(X, y)
    metrics, _, _ = evaluate_classifier(model, X, y)
    
    print("Test metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")
    print(f"Per-Class - Genuine: {metrics['per_class_accuracy']['genuine']:.4f}, Deepfake: {metrics['per_class_accuracy']['deepfake']:.4f}")
    print("Model module works perfectly!")
