import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

# Extract pitch using librosa's pyin
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500,
                                              sr=sr, hop_length=512)
times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

plt.figure(figsize=(12, 4))
plt.plot(times, f0, 'o', markersize=2)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch (F0) Over Time')
plt.savefig('pitch_tracking.png', dpi=100)
plt.show()

# Summary stats
f0_valid = f0[~np.isnan(f0)]
print(f"Average pitch: {np.mean(f0_valid):.1f} Hz")
print(f"Pitch range: {np.min(f0_valid):.1f} - {np.max(f0_valid):.1f} Hz")