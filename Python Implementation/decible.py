import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Raw amplitude
axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y)
axes[0].set_title('Raw Amplitude (Linear)')
axes[0].set_ylabel('Amplitude')

# dB scale (using absolute value + small epsilon to avoid log(0))
y_db = 20 * np.log10(np.abs(y) + 1e-10)
axes[1].plot(np.linspace(0, len(y)/sr, len(y)), y_db)
axes[1].set_title('Amplitude in dB')
axes[1].set_ylabel('dB')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('amplitude_vs_db.png', dpi=100)
plt.show()