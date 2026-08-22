import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

# Compute Mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                          hop_length=512, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(12, 5))
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512,
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.savefig('mel_spectrogram.png', dpi=100)
plt.show()