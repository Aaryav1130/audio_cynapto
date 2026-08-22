import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

# Compute RMS energy per frame
frame_length = 2048
hop_length = 512
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# Also compute it manually to understand it
frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
rms_manual = np.sqrt(np.mean(frames**2, axis=0))

# Plot
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

plt.figure(figsize=(12, 4))
plt.plot(times, rms, label='RMS Energy')
plt.xlabel('Time (s)')
plt.ylabel('RMS')
plt.title('RMS Energy Over Time')
plt.legend()
plt.savefig('rms_energy.png', dpi=100)
plt.show()