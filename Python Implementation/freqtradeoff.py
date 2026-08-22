import librosa
import matplotlib.pyplot as plt
import numpy as np

# Visualize what frames look like
y, sr = librosa.load('sample.wav', sr=16000)

frame_size = 2048
hop_size = 512

# Show the first 3 frames overlapping
plt.figure(figsize=(12, 4))
colors = ['blue', 'red', 'green']
for i in range(3):
    start = i * hop_size
    end = start + frame_size
    frame = y[start:end]
    time = np.arange(start, end) / sr
    plt.plot(time, frame, color=colors[i], alpha=0.5, label=f'Frame {i+1}')
    plt.axvline(x=start/sr, color=colors[i], linestyle='--', alpha=0.3)

plt.title(f'Overlapping Frames (frame={frame_size}, hop={hop_size})')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.savefig('frames_visualization.png', dpi=100)
plt.show()