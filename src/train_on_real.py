import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from src.features import extract_features
from src.model import train_classifier, evaluate_classifier, save_model

def load_real_dataset(dataset_dir):
    """
    Scans a real-world dataset directory (e.g. Kaggle Fake-or-Real).
    Expected directory structure:
    dataset_dir/
      ├── real/ (or genuine/) -> containing real human speech WAV/MP3 files
      └── fake/ (or deepfake/) -> containing AI-generated speech WAV/MP3 files
    """
    X = []
    y = []
    
    # Check folder name mappings (Kaggle dataset uses 'real' and 'fake')
    class_mappings = {
        "real": 0, "genuine": 0,
        "fake": 1, "deepfake": 1, "spoof": 1
    }
    
    found_dirs = False
    for folder_name, label in class_mappings.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.exists(folder_path):
            found_dirs = True
            files = [f for f in os.listdir(folder_path) if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]
            print(f"Found {len(files)} audio files in: {folder_path} (Class: {'Genuine' if label==0 else 'Deepfake'})")
            
            # Extract features from a maximum of 1000 files per class for local resource safety
            # If the user wants to train on the entire dataset, they can remove this limit
            max_files = 1000
            files_to_process = files[:max_files]
            if len(files) > max_files:
                print(f" -> Capping at {max_files} files for training safety.")
                
            for idx, file_name in enumerate(files_to_process):
                file_path = os.path.join(folder_path, file_name)
                # Print progress every 100 files
                if idx % 100 == 0 and idx > 0:
                    print(f"   Processed {idx}/{len(files_to_process)} files...")
                    
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    
    if not found_dirs:
        print(f"Error: Could not find any standard class directories (real, fake, genuine, deepfake) inside: {dataset_dir}")
        return None, None
        
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="AcoustiShield AI: Train on Production Datasets")
    parser.add_argument("dataset_dir", help="Path to the extracted Kaggle Fake-or-Real or ASVspoof dataset directory")
    parser.add_argument("--model_out", default="saved_models/detector_model.joblib", help="Output path for the trained model")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set split ratio (default: 0.2)")
    
    args = parser.parse_args()
    
    print(f"=== Step 1: Processing Production Dataset in {args.dataset_dir} ===")
    X, y = load_real_dataset(args.dataset_dir)
    
    if X is None or len(X) == 0:
        print("Error: No features could be extracted. Training aborted.")
        return
        
    print(f"\nFeature extraction complete. Extracted feature matrix shape: {X.shape}")
    
    print("\n=== Step 2: Splitting Data into Train & Validation Sets ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=42, stratify=y
    )
    print(f"Training set size   : {X_train.shape[0]} samples")
    print(f"Validation set size : {X_val.shape[0]} samples")
    
    print("\n=== Step 3: Training Regularized Classifier ===")
    # Train our robust, regularized classifier to prevent overfitting
    model = train_classifier(X_train, y_train, max_depth=6, min_samples_split=8, min_samples_leaf=4)
    print("Training complete.")
    
    print("\n=== Step 4: Evaluating on Validation Set ===")
    metrics, _, _ = evaluate_classifier(model, X_val, y_val)
    
    # Print the real-world validation report
    print("\n================ PRODUCTION VAL REPORT ================")
    print(f"Overall Accuracy : {metrics['accuracy'] * 100:.2f}% (PS Benchmark: >= 80.00%)")
    print(f"Equal Error Rate : {metrics['eer'] * 100:.2f}% (PS Benchmark: <= 12.00%)")
    print(f"F1-Score         : {metrics['f1_score'] * 100:.2f}% (PS Benchmark: >= 80.00%)")
    print("\n--- Per-Class Accuracy ---")
    print(f"Genuine Speech   : {metrics['per_class_accuracy']['genuine'] * 100:.2f}%")
    print(f"Deepfake Speech  : {metrics['per_class_accuracy']['deepfake'] * 100:.2f}%")
    print("\n--- Confusion Matrix ---")
    print(f"True Genuine  (TN): {metrics['confusion_matrix']['tn']}")
    print(f"False Fake    (FP): {metrics['confusion_matrix']['fp']}")
    print(f"False Genuine (FN): {metrics['confusion_matrix']['fn']}")
    print(f"True Fake     (TP): {metrics['confusion_matrix']['tp']}")
    print("=======================================================\n")
    
    print("=== Step 5: Saving Production Model ===")
    save_model(model, args.model_out)
    print("Model successfully saved and deployed to dashboard.")

if __name__ == "__main__":
    main()
