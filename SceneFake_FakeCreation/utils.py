import csv
import random
import subprocess
from fake_gen_meta import TARGET_SR
import numpy as np
import torchaudio
import soundfile as sf
from pathlib import Path

def load_esc50_metadata(csv_path):
    meta = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["filename"]] = row["category"]
    return meta


def load_audio(path, target_sr=TARGET_SR):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(dim=0)
    return wav.numpy(), target_sr


def save_audio(path, wav, sr):
    wav = wav / (np.max(np.abs(wav)) + 1e-9)
    sf.write(path, wav, sr)


def loop_or_crop(x, length):
    if len(x) >= length:
        return x[:length]
    reps = int(np.ceil(length / len(x)))
    return np.tile(x, reps)[:length]

