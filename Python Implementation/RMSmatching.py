import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

y1, sr1 = librosa.load('audio1.wav', sr=None)
y2, sr2 = librosa.load('audio2.wav', sr=None)

# Compute RMS of each
rms1 = np.sqrt(np.mean(y1**2))
rms2 = np.sqrt(np.mean(y2**2))

print(f"Audio 1 RMS: {rms1:.4f}")
print(f"Audio 2 RMS: {rms2:.4f}")

# Scale audio2 to match audio1's loudness
scaling_factor = rms1 / rms2
y2_matched = y2 * scaling_factor

# Clip to avoid distortion
y2_matched = np.clip(y2_matched, -1.0, 1.0)

rms2_new = np.sqrt(np.mean(y2_matched**2))
print(f"Audio 2 RMS after matching: {rms2_new:.4f}")

sf.write('audio2_volume_matched.wav', y2_matched, sr2)