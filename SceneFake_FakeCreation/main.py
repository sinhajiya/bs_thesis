import numpy as np
import subprocess
import random
import csv
from pathlib import Path
import os

from fake_gen_meta import *
from utils import *


# -------------------------------------------------
# Utility: Apply Fade In / Fade Out
# -------------------------------------------------

def apply_fade(signal, fade_ratio=0.05, fade_in=True, fade_out=True):
    length = len(signal)
    fade_len = int(fade_ratio * length)
    fade_len = min(fade_len, length // 2)

    if fade_len <= 0:
        return signal

    ramp = np.linspace(0, 1, fade_len)

    if fade_in:
        signal[:fade_len] *= ramp

    if fade_out:
        signal[-fade_len:] *= ramp[::-1]

    return signal

    ramp = np.linspace(0, 1, fade_len)
    signal[:fade_len] *= ramp
    signal[-fade_len:] *= ramp[::-1]

    return signal


# -------------------------------------------------
# Sequential Background Construction
# - First event starts at 0
# - Then shift by random amount (overlap allowed)
# - Continue until full audio is covered
# -------------------------------------------------

def add_background_events(speech, noise_files, esc_meta):
    length = len(speech)
    sr = SAMPLE_RATE

    combined_background = np.zeros(length)
    events_metadata = []

    scene_group_name = random.choice(list(SCENE_GROUPS.keys()))
    allowed_categories = SCENE_GROUPS[scene_group_name]
    selection_pool = allowed_categories + ["silence"]

    speech_power = np.mean(speech ** 2) + 1e-9

    current_pos = 0
    first_event = True

    MIN_EVENT_SAMPLES = int(1 * sr)  # minimum meaningful duration (0.5s)

    while current_pos < length:
        remaining = length - current_pos

        # If remaining duration is tiny → add one final overlapping event and exit
        if remaining <= MIN_EVENT_SAMPLES:
            category = random.choice(allowed_categories)  # avoid silence at tail

            candidate_files = [
                p for p in noise_files
                if esc_meta.get(p.name) == category
            ]

            if candidate_files:
                noise_path = random.choice(candidate_files)
                noise, _ = load_audio(str(noise_path))

                event_len = min(len(noise), remaining)
                noise = noise[:event_len].copy()

                noise = apply_fade(
                    noise,
                    fade_ratio=random.uniform(0.03, 0.1),
                    fade_in=True,
                    fade_out=True,
                )

                snr_db = random.choice(SNR_CHOICES)
                noise_power = np.mean(noise ** 2) + 1e-9
                scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
                noise *= scale

                combined_background[current_pos:length] += noise

                events_metadata.append({
                    "background_label": noise_path.stem,
                    "category": category,
                    "start_sec": round(current_pos / sr, 3),
                    "end_sec": round(length / sr, 3),
                    "snr_db": snr_db,
                })

            break

        category = random.choice(selection_pool)

        # Random event length relative to remaining duration
        event_len = random.randint(int(0.2 * length), int(0.5 * length))
        event_len = min(event_len, remaining)

        start = current_pos
        end = start + event_len

        # ----------------------
        # Silence Event
        # ----------------------
        if category == "silence":
            events_metadata.append({
                "background_label": "silence",
                "category": "silence",
                "start_sec": round(start / sr, 3),
                "end_sec": round(end / sr, 3),
                "snr_db": None,
            })
        else:
            candidate_files = [
                p for p in noise_files
                if esc_meta.get(p.name) == category
            ]

            if candidate_files:
                noise_path = random.choice(candidate_files)
                noise, _ = load_audio(str(noise_path))

                event_len = min(event_len, len(noise))
                noise = noise[:event_len].copy()

                # Skip fade-in for very first event
                if start == 0:
                    noise = apply_fade(
                        noise,
                        fade_ratio=random.uniform(0.03, 0.1),
                        fade_in=False,
                        fade_out=True,
                    )
                else:
                    noise = apply_fade(
                        noise,
                        fade_ratio=random.uniform(0.03, 0.1),
                        fade_in=True,
                        fade_out=True,
                    )

                snr_db = random.choice(SNR_CHOICES)
                noise_power = np.mean(noise ** 2) + 1e-9
                scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
                noise *= scale

                combined_background[start:start+event_len] += noise

                events_metadata.append({
                    "background_label": noise_path.stem,
                    "category": category,
                    "start_sec": round(start / sr, 3),
                    "end_sec": round((start + event_len) / sr, 3),
                    "snr_db": snr_db,
                })

        # Shift cursor (allows overlap)
        shift = random.randint(int(0.3 * event_len), event_len)
        current_pos += shift

    return combined_background, scene_group_name, events_metadata


# -------------------------------------------------
# Main Fake Scene Generation
# -------------------------------------------------

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

    for wav in original_files:
        FILE_NAME = wav.stem

        subprocess.run([
            "deepFilter",
            str(wav),
            "--output-dir", str(DENOISED_OUT_DIR)
        ], check=True)

        denoised_path = DENOISED_OUT_DIR / f"{FILE_NAME}_DeepFilterNet3.wav"
        speech, sr = load_audio(denoised_path)

        background, scene_group, events_metadata = add_background_events(
            speech,
            all_noise_files,
            ESC_META
        )

        fake = speech + background

        output_wav = OUT_DIR / f"{FILE_NAME}_{scene_group}.wav"
        save_audio(output_wav, fake, sr)

        with open(SCENE_META_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            for event in events_metadata:
                writer.writerow([
                    wav.name,
                    output_wav.name,
                    scene_group,
                    event["background_label"],
                    event["category"],
                    event["start_sec"],
                    event["end_sec"],
                    event["snr_db"],
                ])


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DENOISED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FAKE_SCENES_DIR, exist_ok=True)

    with open(SCENE_META_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "original_file",
            "new_file",
            "scene_group",
            "background_label",
            "category",
            "start_sec",
            "end_sec",
            "snr_db",
        ])

    esc50_meta = load_esc50_metadata(ESC50_META_CSV)

    fake_scene_generation(
        ORIGINAL_FILES_DIR=INPUT_ORIGINAL_DIR,
        DENOISED_OUT_DIR=OUTPUT_DENOISED_DIR,
        NOISE_DIR=NOISE_DIR,
        OUT_DIR=OUTPUT_FAKE_SCENES_DIR,
        ESC_META=esc50_meta,
        MAX_FILES=None,
    )
