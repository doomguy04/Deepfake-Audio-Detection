import numpy as np
import librosa
import warnings

def extract_features(file_path, sr=16000):
    """
    Extracts high-dimensional acoustic features from an audio file.
    
    Features extracted:
    - Mel-frequency Cepstral Coefficients (MFCCs) [40 features]
    - Chroma STFT [24 features]
    - Mel Spectrogram (mean & std of 20 bands) [40 features]
    - Spectral Contrast (7 bands) [14 features]
    - Tonnetz (tonal centroids) [12 features]
    - Zero Crossing Rate [2 features]
    - Spectral Centroid [2 features]
    - Spectral Rolloff [2 features]
    - RMS Energy [2 features]
    
    Returns:
        numpy.ndarray: A flat 1D feature vector of size 138.
    """
    # Suppress librosa/audioread warnings for clean execution
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        # Load audio file (resampled to 16kHz mono, capped at 10s for performance)
        y, sample_rate = librosa.load(file_path, sr=sr, duration=10.0)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    # Handle silent or very short audio files
    if len(y) == 0:
        y = np.zeros(sr)
        
    # Ensure audio length is sufficient for STFT operations (pad if necessary)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')

    features = []

    # 1. MFCCs (20 coefficients: mean and std)
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=20)
    features.append(np.mean(mfccs, axis=1))
    features.append(np.std(mfccs, axis=1))

    # 2. Chroma STFT (12 bins: mean and std)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate, n_fft=2048)
    features.append(np.mean(chroma, axis=1))
    features.append(np.std(chroma, axis=1))

    # 3. Mel Spectrogram (20 mel-bands: mean and std)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=20)
    # Convert to log amplitude scale
    log_mel = librosa.power_to_db(mel, ref=np.max)
    features.append(np.mean(log_mel, axis=1))
    features.append(np.std(log_mel, axis=1))

    # 4. Spectral Contrast (7 bands: mean and std)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate, n_fft=2048)
    features.append(np.mean(contrast, axis=1))
    features.append(np.std(contrast, axis=1))

    # 5. Tonnetz (6 dimensions: mean and std)
    # Tonnetz requires chroma as input
    tonnetz = librosa.feature.tonnetz(y=y, sr=sample_rate, chroma=chroma)
    features.append(np.mean(tonnetz, axis=1))
    features.append(np.std(tonnetz, axis=1))

    # 6. Zero Crossing Rate (mean and std)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append([np.mean(zcr), np.std(zcr)])

    # 7. Spectral Centroid (mean and std)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
    features.append([np.mean(centroid), np.std(centroid)])

    # 8. Spectral Rolloff (mean and std)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)
    features.append([np.mean(rolloff), np.std(rolloff)])

    # 9. RMS Energy (mean and std)
    rms = librosa.feature.rms(y=y)
    features.append([np.mean(rms), np.std(rms)])

    # Concatenate all features into a single 1D array
    feature_vector = np.concatenate(features)
    
    return feature_vector

def get_feature_names():
    """
    Returns a list of feature names corresponding to the returned feature vector.
    """
    names = []
    # MFCCs
    for i in range(20): names.append(f"mfcc_mean_{i}")
    for i in range(20): names.append(f"mfcc_std_{i}")
    # Chroma
    for i in range(12): names.append(f"chroma_mean_{i}")
    for i in range(12): names.append(f"chroma_std_{i}")
    # Mel Spectrogram
    for i in range(20): names.append(f"mel_mean_{i}")
    for i in range(20): names.append(f"mel_std_{i}")
    # Spectral Contrast
    for i in range(7): names.append(f"contrast_mean_{i}")
    for i in range(7): names.append(f"contrast_std_{i}")
    # Tonnetz
    for i in range(6): names.append(f"tonnetz_mean_{i}")
    for i in range(6): names.append(f"tonnetz_std_{i}")
    # Temporal/spectral scalar features
    names.extend(["zcr_mean", "zcr_std"])
    names.extend(["centroid_mean", "centroid_std"])
    names.extend(["rolloff_mean", "rolloff_std"])
    names.extend(["rms_mean", "rms_std"])
    
    return names

if __name__ == "__main__":
    # Test feature extraction on dummy wave if run directly
    import tempfile
    import scipy.io.wavfile as wavfile
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Create a tiny 0.5s test sound
        fs = 16000
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        data = np.sin(2 * np.pi * 440 * t)
        data_int16 = (data * 32767).astype(np.int16)
        wavfile.write(tmp.name, fs, data_int16)
        
        vec = extract_features(tmp.name)
        names = get_feature_names()
        print(f"Extraction successful!")
        print(f"Feature vector shape: {vec.shape} (Expected: (138,))")
        print(f"Number of feature names: {len(names)}")
        print(f"Sample features: {names[:5]} -> {vec[:5]}")
