import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Load original audio
y, sr = librosa.load('sample.wav', sr=None)  # sr=None keeps original rate
print(f"Original: {sr} Hz, {len(y)} samples, {len(y)/sr:.2f} seconds")

# Resample to different rates and save — LISTEN to each one in Audacity!
for target_sr in [8000, 16000, 48000]:
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sf.write(f'sample_{target_sr}hz.wav', y_resampled, target_sr)
    print(f"Saved sample_{target_sr}hz.wav")