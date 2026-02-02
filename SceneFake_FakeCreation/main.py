import numpy as np
import torchaudio
import soundfile as sf
import subprocess
import random
import tempfile
import csv
from pathlib import Path


TARGET_SR = 16000
SNR_CHOICES = [-5, 0, 5, 10, 15, 20]

ESC50_META_CSV = Path(
    "/home/bs_thesis/Documents/EnvDatasets/ESC-50-master/meta/esc50.csv"
)

ALLOWED_CATEGORIES = {
    "dog",
    "chirping_birds",
    "crow",
    "clapping",
    "mouse_click",
    "water_drops",
    "keyboard_typing",
    "wind",
    "footsteps",
    "car_horn",
    "rain",
    "insects",
    "laughing",
    "engine",
    "washing_machine",
    "door_wood_creaks",
    "crickets",
}


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


def spectral_tilt(x, alpha):
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = x[i] - alpha * x[i - 1]
    return y


def mix_at_snr(speech, bg, snr_db):
    sp = np.mean(speech ** 2)
    bp = np.mean(bg ** 2) + 1e-9
    scale = np.sqrt(sp / (bp * 10 ** (snr_db / 10)))
    return speech + bg * scale



def fake_scene_generation(ORIGINAL_FILES_DIR, DENOISED_OUT_DIR, NOISE_DIR, OUT_DIR, ESC_META,SNR_DB=SNR_CHOICES , MAX_FILES=None ):
    os.makedirs(OUT_DIR, exist_ok=True)
    inp = Path(ORIGINAL_FILES_DIR)
    out = Path(OUT_DIR)
    denoised_out_dir = Path(DENOISED_OUT_DIR)
    
    all_noise_files = list(Path(NOISE_DIR).rglob("*.wav"))

    noise_files = [
        p for p in all_noise_files
        if ESC_META.get(p.name) in ALLOWED_CATEGORIES
    ]

    speech_files = sorted(inp.rglob("*.wav"))
    if MAX_FILES is not None:
        speech_files = speech_files[MAX_FILES:]


    for wav in speech_files:
        FILE_NAME = wav.stem
        subprocess.run([
            "deepFilter",
            str(wav),
            "--output-dir", str(denoised_out_dir)
        ], check=True)

        print(f"DENOISED {wav.name}")

        DENOISED_WAV = denoised_out_dir / f"{FILE_NAME}_DeepFilterNet3.wav"
        speech, sr = load_audio(DENOISED_WAV)
        # noise_path = random.choice(noise_files)
        # noise_name = noise_path.stem
        noise_path = random.choice(noise_files)
        noise_filename = noise_path.name
        scene_name = ESC_META.get(noise_filename, "unknown")


        snr_db = random.choice(SNR_DB)
        noise, _ = load_audio(str(noise_path))

        noise = loop_or_crop(noise, len(speech))

        speech_col = spectral_tilt(speech, alpha=0.95)
        noise_col = spectral_tilt(noise, alpha=0.70)
        # output_wav = out / f"{FILE_NAME}_SNR{snr_db}_{scene_name}_fake.wav"
        output_wav = out / f"{FILE_NAME}_SNR{snr_db}_{scene_name}.wav"


        fake = mix_at_snr(speech_col, noise_col, snr_db)
        save_audio(output_wav, fake, sr)


        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                wav.name,
                output_wav.name,
                
                scene_name,
                snr_db
            ])



def is_allowed_file(path):
    """
    Exclude aud289_dev3.wav to aud348_dev3.wav (inclusive) - since this is the files with music background and the model doesnot work well music mainly the vocal part. 
    """
    name = path.stem  # audXXX_dev3
    try:
        idx = int(name.split("_")[0].replace("aud", ""))
    except ValueError:
        return False  # skip malformed names

    return not (289 <= idx <= 348)


# ------------------ MAIN ------------------

if __name__ == "__main__":

    speech_dir = Path("/home/bs_thesis/Documents/OurDataset/device3_iphone/")
    speech_files = sorted(
        p for p in speech_dir.glob("aud*_dev3.wav")
        if is_allowed_file(p)
    )

    esc50_meta = load_esc50_metadata(ESC50_META_CSV)

    noise_dir = Path("/home/bs_thesis/Documents/EnvDatasets/ESC-50-master/audio")
    denoised_out_dir = Path("/home/bs_thesis/Documents/scenefakegeneration_ours/FakeCreation/FakeScenesDenoised")
    out_dir = Path("/home/bs_thesis/Documents/scenefakegeneration_ours/FakeScenes")
    log_path = out_dir / "scene_metadata.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    denoised_out_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "original_file",
            "new_file",
            # "new_path",
            "scene_name",
            "snr_db"
        ])
    fake_scene_generation(
        ORIGINAL_FILES_DIR=speech_dir,
        DENOISED_OUT_DIR=denoised_out_dir,
        NOISE_DIR=noise_dir,ESC_META=esc50_meta,
        OUT_DIR=out_dir,
        SNR_DB=SNR_CHOICES ,MAX_FILES=100     )
