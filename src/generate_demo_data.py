import os
import numpy as np
import scipy.io.wavfile as wavfile

def generate_synthetic_speech(file_path, label, duration=3.0, fs=16000, seed=42):
    """
    Generates a synthetic audio wave representing either Genuine human speech or Deepfake speech.
    
    Genuine speech:
    - Dynamic pitch (F0) with natural variations (jitter/shimmer)
    - Harmonic resonances (vocal formants)
    - Natural breathing/unvoiced noise (pink-like noise)
    - Smooth temporal envelope
    
    Deepfake speech:
    - Flat, robotic pitch (no jitter) or artificial high-frequency modulation
    - Periodic spectral notches (vocoder artifacts)
    - Phase anomalies at high frequencies
    - Subtle digital pilot hums or high-frequency hiss
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 1. Base Pitch (F0) Generation
    if label == "genuine":
        # Natural pitch contour (dynamic)
        base_f0 = 120 + 20 * np.sin(2 * np.pi * 1.2 * t) + 5 * np.cos(2 * np.pi * 3.5 * t)
        # Add micro-jitter (natural pitch variation)
        jitter = 0.005 * np.random.normal(0, 1, len(t))
        phase = 2 * np.pi * np.cumsum((base_f0 * (1 + jitter)) / fs)
    else:
        # Realistic neural-vocoder pitch (mimics human F0 but with synthetic smoothness)
        base_f0 = 132.0 + 8 * np.sin(2 * np.pi * 0.9 * t) + 2 * np.cos(2 * np.pi * 2.8 * t)
        # Add a tiny bit of high-frequency periodic jitter (digital synthesizer tracking jitter)
        base_f0 += 0.4 * np.sin(2 * np.pi * 12.0 * t)
        phase = 2 * np.pi * np.cumsum(base_f0 / fs)

    # 2. Harmonics and Formants (Vocal Tract Resonance Simulation)
    # Define formant resonances
    # Human vocal tract resonances: F1 (~600 Hz), F2 (~1700 Hz), F3 (~2800 Hz)
    if label == "genuine":
        formants = [(600, 120), (1700, 220), (2800, 280)]
    else:
        # Vocoder-shifted formants
        formants = [(620, 90), (1650, 150), (2850, 180)]
        
    signal = np.zeros_like(t)
    
    # Generate source harmonic wave (voiced component)
    # Human voice contains fundamental F0 plus integer harmonics
    num_harmonics = 15
    for h in range(1, num_harmonics + 1):
        harmonic_freq = h * base_f0
        # Calculate formant gain using resonance filter response (Lorentzian profiles)
        gain = 0.0
        for center, width in formants:
            # Amplitude response of bandpass filter
            gain += 1.0 / (1.0 + ((h * 150.0 - center) / width)**2)
        
        # Apply amplitude envelope
        amplitude = (1.0 / h) * gain
        
        # Add shimmer (amplitude variation) for genuine speech
        if label == "genuine":
            shimmer = 1.0 + 0.06 * np.sin(2 * np.pi * 8 * t) + 0.02 * np.random.normal(0, 1, len(t))
            harmonic_wave = amplitude * shimmer * np.sin(h * phase)
        else:
            # Synthetic harmonics with slight amplitude fluctuations (mimics shimmer, makes it harder)
            shimmer = 1.0 + 0.03 * np.sin(2 * np.pi * 6.5 * t)
            phase_offset = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi) if h > 6 else 0
            harmonic_wave = amplitude * shimmer * np.sin(h * phase + phase_offset)
            
        signal += harmonic_wave

    # 3. Add Unvoiced Component (Noise)
    # Speech contains breathy/fricative noise components
    noise = np.random.normal(0, 1, len(t))
    if label == "genuine":
        # Pink filter: bandpassed noise (natural breathing)
        # Emulate pink noise by decaying higher frequencies
        noise_fft = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(t), 1/fs)
        freqs[0] = 1.0  # avoid division by zero
        noise_fft = noise_fft / np.sqrt(freqs)
        filtered_noise = np.fft.irfft(noise_fft, len(t))
        # Keep noise level natural
        signal += 0.05 * filtered_noise
    else:
        # Deepfakes often have digital high-frequency noise/hiss
        # White noise bandpassed to higher frequency (4000-8000 Hz)
        noise_fft = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(t), 1/fs)
        # Apply bandpass filter
        mask = (freqs >= 4500) & (freqs <= 7800)
        noise_fft[~mask] = 0
        filtered_noise = np.fft.irfft(noise_fft, len(t))
        signal += 0.08 * filtered_noise

    # 4. Add Deepfake-Specific Artifacts
    if label == "deepfake":
        # A: Spectral notches (vocoder bands)
        # We can implement this in frequency domain by applying notches
        sig_fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(t), 1/fs)
        # Create notches at 1200-1400 Hz and 3000-3200 Hz (shallower, 0.25 gain)
        notch_mask = ((freqs >= 1200) & (freqs <= 1400)) | ((freqs >= 3000) & (freqs <= 3200))
        sig_fft[notch_mask] *= 0.25
        signal = np.fft.irfft(sig_fft, len(t))
        
        # B: High frequency pilot hum/whine (e.g. 7500 Hz synthesis hum)
        signal += 0.008 * np.sin(2 * np.pi * 7500 * t)
        
        # C: Sub-50Hz power grid buzz
        signal += 0.02 * np.sin(2 * np.pi * 50 * t)

    # 5. Apply Speech Envelope (Onset / Offset and Syllables)
    # Speech envelope is not constant, it rises and falls to simulate syllables
    envelope = np.zeros_like(t)
    syllable_rate = 2.5  # 2.5 syllables per second
    # Generate continuous envelope
    envelope = 0.5 * (1 + np.sin(2 * np.pi * syllable_rate * t))
    # Apply fade-in and fade-out to prevent clicks
    fade_len = int(0.15 * fs)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    envelope[:fade_len] *= fade_in
    envelope[-fade_len:] *= fade_out
    
    signal = signal * envelope

    # Add realistic channel/environmental noise (shared by both classes)
    noise_floor = np.random.normal(0, 0.04, len(t))
    signal += noise_floor

    # 6. Normalize and Convert to 16-bit PCM
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    audio_int16 = (signal * 32767).astype(np.int16)
    
    # Save file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    wavfile.write(file_path, fs, audio_int16)

def generate_dataset(base_dir="demo_data"):
    """
    Generates training and validation datasets for Genuine and Deepfake audio files.
    """
    print("Generating demo audio dataset...")
    
    # Generate Training Data
    # 30 Genuine, 30 Deepfake
    for i in range(30):
        gen_path = os.path.join(base_dir, "train", "genuine", f"genuine_{i}.wav")
        fake_path = os.path.join(base_dir, "train", "deepfake", f"deepfake_{i}.wav")
        generate_synthetic_speech(gen_path, "genuine", seed=100+i)
        generate_synthetic_speech(fake_path, "deepfake", seed=200+i)

    # Generate Validation Data
    # 15 Genuine, 15 Deepfake
    for i in range(15):
        gen_path = os.path.join(base_dir, "val", "genuine", f"genuine_{i}.wav")
        fake_path = os.path.join(base_dir, "val", "deepfake", f"deepfake_{i}.wav")
        generate_synthetic_speech(gen_path, "genuine", seed=300+i)
        generate_synthetic_speech(fake_path, "deepfake", seed=400+i)
        
    print(f"Dataset generated successfully in: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    generate_dataset()
