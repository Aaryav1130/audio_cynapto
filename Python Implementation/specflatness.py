import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

flatness = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=512)[0]
times = librosa.frames_to_time(np.arange(len(flatness)), sr=sr, hop_length=512)

plt.figure(figsize=(12, 4))
plt.plot(times, flatness)
plt.xlabel('Time (s)')
plt.ylabel('Spectral Flatness')
plt.title('Spectral Flatness (0=tonal, 1=noise)')
plt.ylim(0, 1)
plt.savefig('spectral_flatness.png', dpi=100)
plt.show()