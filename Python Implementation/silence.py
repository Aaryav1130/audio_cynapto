import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('sample.wav', sr=None)

frame_length = 2048
hop_length = 512

# Compute RMS per frame
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# Set threshold (in RMS, not dB) — tweak this based on your audio
silence_threshold = 0.01

# Find silent frames
is_silent = rms < silence_threshold

# Convert frame indices to time ranges
silent_regions = []
in_silence = False
start = 0

for i, silent in enumerate(is_silent):
    if silent and not in_silence:
        start = i
        in_silence = True
    elif not silent and in_silence:
        start_time = librosa.frames_to_time(start, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        # Only count silences longer than 0.2 seconds
        if end_time - start_time > 0.2:
            silent_regions.append((start_time, end_time))
        in_silence = False

print("Silent regions found:")
for s, e in silent_regions:
    print(f"  {s:.2f}s - {e:.2f}s (duration: {e-s:.2f}s)")

# Visualize
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
plt.figure(figsize=(12, 4))
plt.plot(times, rms, label='RMS')
plt.axhline(y=silence_threshold, color='r', linestyle='--', label='Threshold')
for s, e in silent_regions:
    plt.axvspan(s, e, color='red', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('RMS')
plt.title('Silent Regions (red shaded)')
plt.legend()
plt.savefig('silent_regions.png', dpi=100)
plt.show()