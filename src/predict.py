import os
import sys
import argparse

# Add parent directory to sys.path for robust relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features
from src.model import load_model

def predict_audio(file_path, model_path="saved_models/detector_model.joblib"):
    """
    Predicts if the audio file is Genuine or Deepfake, along with confidence.
    """
    if not os.path.exists(file_path):
        print(f"Error: Audio file not found at: {file_path}")
        return None
        
    if not os.path.exists(model_path):
        print(f"Error: Trained model file not found at: {model_path}")
        print("Please run the training pipeline first: python3 src/train.py")
        return None
        
    # Extract features
    # print(f"Extracting features from {os.path.basename(file_path)}...")
    features = extract_features(file_path)
    if features is None:
        print("Error: Feature extraction failed.")
        return None
        
    # Reshape features to 2D array (1 sample)
    features_2d = features.reshape(1, -1)
    
    # Load model
    model = load_model(model_path)
    
    # Predict
    pred = model.predict(features_2d)[0]
    probs = model.predict_proba(features_2d)[0]
    
    # Class mapping: 0: Genuine, 1: Deepfake
    label = "Genuine (Human)" if pred == 0 else "Deepfake (AI-Generated)"
    confidence = probs[pred]
    
    return {
        "file": os.path.basename(file_path),
        "prediction": label,
        "class_id": int(pred),
        "confidence": float(confidence),
        "probabilities": {
            "genuine": float(probs[0]),
            "deepfake": float(probs[1])
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection CLI Predictor")
    parser.add_argument("audio_file", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument("--model", default="saved_models/detector_model.joblib", help="Path to the trained model joblib file")
    
    args = parser.parse_args()
    
    result = predict_audio(args.audio_file, args.model)
    if result:
        print("\n================ DETECTOR INFERENCE ================")
        print(f"File Name    : {result['file']}")
        print(f"Prediction   : {result['prediction']}")
        print(f"Confidence   : {result['confidence'] * 100:.2f}%")
        print("\n--- Raw Probabilities ---")
        print(f"Genuine (Human)      : {result['probabilities']['genuine'] * 100:.2f}%")
        print(f"Deepfake (AI-Gen)    : {result['probabilities']['deepfake'] * 100:.2f}%")
        print("====================================================\n")

if __name__ == "__main__":
    main()
