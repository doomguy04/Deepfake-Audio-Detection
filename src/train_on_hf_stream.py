import os
import sys
import shutil
import numpy as np
from datasets import load_dataset, Audio
from src.features import extract_features
from src.model import train_classifier, evaluate_classifier, save_model

def train_on_stream(num_train_per_class=60, num_val_per_class=20):
    print("=== Step 1: Cleaning Up and Preparing Directories ===")
    # Clear old synthetic DSP directories to avoid mixing datasets
    base_dir = "demo_data"
    for split in ["train", "val"]:
        for cls in ["genuine", "deepfake"]:
            path = os.path.join(base_dir, split, cls)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            
    print("=== Step 2: Loading Hugging Face Dataset in Streaming Mode ===")
    ds = load_dataset("garystafford/deepfake-audio-detection", streaming=True)
    # Shuffle the stream to mix classes (Genuine/Deepfake) and avoid long skips
    ds = ds.shuffle(seed=42, buffer_size=500)
    # Cast audio column to decode=False to bypass torchcodec requirement
    ds = ds.cast_column("audio", Audio(decode=False))
    
    # We will gather training and validation datasets
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    # Track counts per class (0: Genuine, 1: Deepfake)
    counts = {0: 0, 1: 0}
    total_needed = (num_train_per_class + num_val_per_class) * 2
    
    print("=== Step 3: Streaming, Saving WAVs, and Extracting Features ===")
    train_iter = iter(ds["train"])
    processed_count = 0
    
    try:
        while True:
            item = next(train_iter)
            label = int(item["label"])
            
            # Check if we still need samples for this class
            limit = num_train_per_class + num_val_per_class
            if counts[label] >= limit:
                if counts[0] >= limit and counts[1] >= limit:
                    break
                continue
                
            # Extract audio bytes
            audio_data = item["audio"]
            audio_bytes = audio_data.get("bytes")
            original_path = audio_data.get("path", "sample.wav")
            
            if audio_bytes is None:
                continue
                
            # Determine split (train vs val)
            is_train = counts[label] < num_train_per_class
            split_name = "train" if is_train else "val"
            class_name = "genuine" if label == 0 else "deepfake"
            idx = counts[label] if is_train else counts[label] - num_train_per_class
            
            # Get extension
            ext = os.path.splitext(original_path)[1]
            if not ext:
                ext = ".wav"
                
            # Save the raw audio file to disk
            file_name = f"real_{idx}{ext}"
            file_path = os.path.join(base_dir, split_name, class_name, file_name)
            
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
                
            # Extract features from the saved file
            try:
                features = extract_features(file_path)
            except Exception as e:
                features = None
                
            if features is not None:
                if is_train:
                    X_train.append(features)
                    y_train.append(label)
                else:
                    X_val.append(features)
                    y_val.append(label)
                    
                counts[label] += 1
                processed_count += 1
                print(f"   [{processed_count}/{total_needed}] Saved and processed: {file_path}")
            else:
                # Cleanup if feature extraction failed
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    
            if counts[0] >= limit and counts[1] >= limit:
                break
                
    except StopIteration:
        print("Reached end of dataset stream before collecting all samples.")
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    
    print(f"\nTraining set size   : {X_train.shape[0]} samples")
    print(f"Validation set size : {X_val.shape[0]} samples")
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("Error: Could not extract features from files. Training aborted.")
        return
        
    print("\n=== Step 4: Training Regularized Classifier ===")
    model = train_classifier(X_train, y_train, max_depth=12, min_samples_split=2, min_samples_leaf=1)
    print("Training complete.")
    
    print("\n=== Step 5: Evaluating on Validation Set ===")
    metrics, _, _ = evaluate_classifier(model, X_val, y_val)
    
    # Print the real-world validation report
    print("\n================ HF REAL DATA VAL REPORT ================")
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
    print("=========================================================\n")
    
    print("=== Step 6: Saving Production Model ===")
    save_model(model, "saved_models/detector_model.joblib")
    print("Real-world trained model saved and active in the dashboard!")

if __name__ == "__main__":
    train_on_stream()
