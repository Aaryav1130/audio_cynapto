import librosa
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pyln

y, sr = librosa.load('sample.wav', sr=None)

# LUFS measurement
meter = pyln.Meter(sr)  # BS.1770 meter
loudness = meter.integrated_loudness(y)
print(f"Integrated LUFS: {loudness:.2f}")

# Sample peak
sample_peak = np.max(np.abs(y))
sample_peak_db = 20 * np.log10(sample_peak + 1e-10)
print(f"Sample Peak: {sample_peak:.4f} ({sample_peak_db:.2f} dBFS)")

# True peak (oversample by 4x, then find peak)
from scipy.signal import resample
y_oversampled = resample(y, len(y) * 4)
true_peak = np.max(np.abs(y_oversampled))
true_peak_db = 20 * np.log10(true_peak + 1e-10)
print(f"True Peak: {true_peak:.4f} ({true_peak_db:.2f} dBTP)")