import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

# Show frequency content
stft = np.abs(librosa.stft(y, n_fft=4096))
freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
avg_magnitude = np.mean(stft, axis=1)

plt.figure(figsize=(12, 4))
plt.semilogx(freqs[1:], 20 * np.log10(avg_magnitude[1:] + 1e-10))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Average Frequency Spectrum')

# Mark the ranges
ranges = [(60, 'Bass'), (250, 'Low-mid'), (500, 'Mid'), 
          (2000, 'Upper-mid'), (4000, 'High'), (8000, 'Very High')]
for freq, label in ranges:
    plt.axvline(x=freq, color='gray', linestyle=':', alpha=0.5)
    plt.text(freq, plt.ylim()[1]-5, label, fontsize=8, rotation=45)

plt.savefig('frequency_ranges.png', dpi=100)
plt.show()