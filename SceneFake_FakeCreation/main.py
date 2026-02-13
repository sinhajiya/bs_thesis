import numpy as np
import torchaudio
import soundfile as sf
import subprocess
import random
import csv
from pathlib import Path
import os
import warnings

from fake_gen_meta import *
from utils import *
    
def random_burst_mask(length): #   gradual increase or decrease in the amplitude of the background sound at its start or end.
    mask = np.zeros(length)

    # Ensure first burst starts near beginning
    current_pos = 0
    num_events = random.randint(1, 4)

    for i in range(num_events):

        if i == 0:
            start = 0
        else:
            start = random.randint(current_pos, length - 1)

        dur = random.randint(length // 8, length // 3)
        end = min(length, start + dur)

        fade_len = random.randint(
            int(0.02 * length),     # small fade
            int(0.1 * length)       # large fade
        )
        fade_len = min(fade_len, (end - start) // 2)

        if fade_len > 0:
            ramp = np.linspace(0, 1, fade_len)

            mask[start:start+fade_len] += ramp
            mask[end-fade_len:end] += ramp[::-1]
            mask[start+fade_len:end-fade_len] += 1
        else:
            mask[start:end] += 1

        current_pos = end
        if current_pos >= length:
            break

    return np.clip(mask, 0, 1)



def time_varying_snr_mix(speech, noise):
    
    length = len(speech)
    segments = random.randint(2, 5)
    seg_len = length // segments
    output = np.zeros_like(speech)
    snr_values = []

    for i in range(segments):
        start = i * seg_len
        end = length if i == segments - 1 else (i + 1) * seg_len

        snr_db = random.choice(SNR_CHOICES)
        snr_values.append(snr_db)

        sp = np.mean(speech[start:end] ** 2)
        bp = np.mean(noise[start:end] ** 2) + 1e-9
        scale = np.sqrt(sp / (bp * 10 ** (snr_db / 10)))

        output[start:end] = speech[start:end] + noise[start:end] * scale

    return output, snr_values

def generate_complex_background(speech, noise_files):
    length = len(speech)
    combined = np.zeros(length)
    scene_labels = []

    num_sources = random.randint(1, 3)

    for _ in range(num_sources):
        noise_path = random.choice(noise_files)
        noise, _ = load_audio(str(noise_path))
        noise = loop_or_crop(noise, length)

        mask = random_burst_mask(length)
        noise = noise * mask

        combined += noise
        scene_labels.append(noise_path.stem)

    return combined, "+".join(scene_labels)



def generate_scene_coherent_background(denoised_audio, noise_files, esc_meta):
    length = len(denoised_audio)
    combined = np.zeros(length)

    scene_group_name = random.choice(list(SCENE_GROUPS.keys()))
    allowed_categories = SCENE_GROUPS[scene_group_name]
    selection_pool = allowed_categories + ["silence"]

    categories_added = []
    noise_labels = []

    current_pos = 0

    while current_pos < length:

        category = random.choice(selection_pool)

        # Silence event
        if category == "silence":
            gap = random.randint(int(0.05 * length), int(0.15 * length))
            current_pos += gap
            continue

        candidate_files = [
            p for p in noise_files
            if esc_meta.get(p.name) == category
        ]

        if not candidate_files:
            continue

        noise_path = random.choice(candidate_files)
        noise, _ = load_audio(str(noise_path))

        event_len = min(len(noise), length - current_pos)

        fade_len = min(int(0.05 * event_len), event_len // 2)
        ramp = np.linspace(0, 1, fade_len)

        segment = noise[:event_len].copy()

        if fade_len > 0:
            segment[:fade_len] *= ramp
            segment[-fade_len:] *= ramp[::-1]

        combined[current_pos:current_pos+event_len] += segment

        categories_added.append(category)
        noise_labels.append(noise_path.stem)

        overlap_shift = random.randint(
            int(0.5 * event_len),
            event_len
        )

        current_pos += overlap_shift

    if not categories_added:
        categories_added = ["silence"]
        noise_labels = ["none"]

    return (
        combined,
        scene_group_name,
        "+".join(noise_labels),
        "+".join(categories_added)
    )

def fake_scene_generation(
    ORIGINAL_FILES_DIR,
    DENOISED_OUT_DIR,
    NOISE_DIR,
    OUT_DIR,
    ESC_META,
    MAX_FILES=None,
):
    print("Starting fake scene generation...")
    all_noise_files = list((NOISE_DIR).rglob("*.wav"))

    original_files = sorted(ORIGINAL_FILES_DIR.rglob("*.wav"))

    if MAX_FILES:
        original_files = original_files[:MAX_FILES]
    print("Looping through original files...")
    for wav in original_files:
        FILE_NAME = wav.stem

        subprocess.run([
            "deepFilter",
            str(wav),
            "--output-dir", str(DENOISED_OUT_DIR)
        ], check=True)

        print("Denoised :", wav.name)
        denoised_path = DENOISED_OUT_DIR / f"{FILE_NAME}_DeepFilterNet3.wav"
        speech, sr = load_audio(denoised_path)

        print("Generating background for :", wav.name)

        background, scene_group, noise_labels, categories_added = \
    generate_scene_coherent_background(
        speech, all_noise_files, ESC_META
)


        # fake = time_varying_snr_mix(speech, background)
        fake, snr_values = time_varying_snr_mix(speech, background)


        output_wav = OUT_DIR / f"{FILE_NAME}_{scene_group}.wav"
        save_audio(output_wav, fake, sr)
        print("Saved fake scene:", output_wav.name)
        
        with open(SCENE_META_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
    wav.name,
    output_wav.name,
    scene_group,
    noise_labels,
    categories_added,
    "+".join(map(str, snr_values))
])




if __name__ == "__main__":
    os.makedirs(OUTPUT_DENOISED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FAKE_SCENES_DIR, exist_ok=True)

    with open(SCENE_META_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
    "original_file",
    "new_file",
    "scene_group",
    "noise_labels",
    "categories_added",
    "snr_segments"
])

    esc50_meta = load_esc50_metadata(ESC50_META_CSV)

    fake_scene_generation(
        ORIGINAL_FILES_DIR=INPUT_ORIGINAL_DIR,
        DENOISED_OUT_DIR=OUTPUT_DENOISED_DIR,
        NOISE_DIR=NOISE_DIR,
        OUT_DIR=OUTPUT_FAKE_SCENES_DIR,
        ESC_META=esc50_meta,
        MAX_FILES=None
    )
