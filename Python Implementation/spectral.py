# Try this to see the problem yourself
import numpy as np
import matplotlib.pyplot as plt

# Simulate: quiet speech (0.01) followed by loud speech (0.5)
quiet = 0.01 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, 16000))
loud = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, 16000))
signal = np.concatenate([quiet, loud])

plt.figure(figsize=(12, 3))
plt.plot(signal)
plt.title("Raw Amplitude — quiet part looks like silence (but it's not!)")
plt.show()