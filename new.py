# Compare spectrogram of one Real file and one Fake file
# (Edit paths below if needed)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# ---- EDIT THESE PATHS ----
real_path = "/home/bs_thesis/datasets/OurDataset/SceneFake-Wild-Real/aud040_dev3.wav"
fake_path = "/home/bs_thesis/datasets/OurDataset/SceneFake-Wild-Fake2/aud040_dev3_Human_SNRmatched.wav"
# ---------------------------

def load_mono(path):
    audio, sr = sf.read(path)
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    return audio, sr

def plot_spec(audio, sr, title):
    f, t, Sxx = spectrogram(audio, sr, nperseg=512, noverlap=256)
    Sxx = 10 * np.log10(Sxx + 1e-10)

    plt.figure()
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="dB")
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)  # saves image

    plt.show()


# ---- Real ----
real_audio, real_sr = load_mono(real_path)
plot_spec(real_audio, real_sr, "Real Spectrogram")

# ---- Fake ----
fake_audio, fake_sr = load_mono(fake_path)
plot_spec(fake_audio, fake_sr, "Fake Spectrogram")
