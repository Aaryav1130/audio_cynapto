import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

# Compute STFT
n_fft = 2048       # frame size
hop_length = 512   # hop size

stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
# stft is complex-valued. Take magnitude and convert to dB.
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

plt.figure(figsize=(12, 5))
librosa.display.specshow(stft_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='log')  # log freq scale
plt.colorbar(format='%+2.0f dB')
plt.title('STFT Spectrogram (dB, Log Frequency Scale)')
plt.savefig('stft_spectrogram.png', dpi=100)
plt.show()