# Check channels
import soundfile as sf
import numpy as np

data, sr = sf.read('sample.wav')
if data.ndim == 1:
    print("Mono audio")
else:
    print(f"Audio has {data.shape[1]} channels")
    # Convert to mono by averaging channels
    mono = np.mean(data, axis=1)
    sf.write('sample_mono.wav', mono, sr)