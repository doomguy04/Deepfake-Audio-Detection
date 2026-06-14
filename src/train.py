import os
import numpy as np
import pandas as pd
from src.features import extract_features
from src.model import train_classifier, evaluate_classifier, save_model

def load_dataset_features(data_dir):
    """
    Loads all WAV files in genuine/ and deepfake/ subfolders of data_dir,
    extracts features, and compiles features + labels.
    """
    X = []
    y = []
    
    classes = {"genuine": 0, "deepfake": 1}
    
    for class_name, label in classes.items():
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory not found: {class_path}")
            continue
            
        files = [f for f in os.listdir(class_path) if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]
        print(f"Extracting features from {len(files)} files in {class_name}...")
        
        for file_name in files:
            file_path = os.path.join(class_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

def run_pipeline(data_dir="demo_data", model_path="saved_models/detector_model.joblib"):
    """
    Runs the full training and evaluation pipeline.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    print("=== Step 1: Loading and Extracting Training Data ===")
    X_train, y_train = load_dataset_features(train_dir)
    print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    
    print("\n=== Step 2: Loading and Extracting Validation Data ===")
    X_val, y_val = load_dataset_features(val_dir)
    print(f"Validation set: X_val shape = {X_val.shape}, y_val shape = {y_val.shape}")
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("Error: No features extracted. Ensure that the dataset has been generated first.")
        return
        
    print("\n=== Step 3: Training Classifier ===")
    model = train_classifier(X_train, y_train)
    print("Training completed.")
    
    print("\n=== Step 4: Evaluating Classifier on Validation Set ===")
    metrics, y_pred, y_prob = evaluate_classifier(model, X_val, y_val)
    
    # Print beautiful performance report
    print("\n================ PERFORMANCE REPORT ================")
    print(f"Overall Accuracy : {metrics['accuracy'] * 100:.2f}% (Threshold: >= 80.00%)")
    print(f"Equal Error Rate : {metrics['eer'] * 100:.2f}% (Threshold: <= 12.00%)")
    print(f"F1-Score         : {metrics['f1_score'] * 100:.2f}% (Threshold: >= 80.00%)")
    print("\n--- Per-Class Accuracy ---")
    print(f"Genuine Speech   : {metrics['per_class_accuracy']['genuine'] * 100:.2f}% (Threshold: >= 75.00%)")
    print(f"Deepfake Speech  : {metrics['per_class_accuracy']['deepfake'] * 100:.2f}% (Threshold: >= 75.00%)")
    print("\n--- Confusion Matrix ---")
    print(f"True Genuine  (TN): {metrics['confusion_matrix']['tn']}")
    print(f"False Fake    (FP): {metrics['confusion_matrix']['fp']}")
    print(f"False Genuine (FN): {metrics['confusion_matrix']['fn']}")
    print(f"True Fake     (TP): {metrics['confusion_matrix']['tp']}")
    print("====================================================\n")
    
    # Save Model
    print("=== Step 5: Saving Trained Model ===")
    save_model(model, model_path)
    
    # Return metrics for validation checks
    return metrics

if __name__ == "__main__":
    run_pipeline()
