import os
import soundfile as sf
import argparse


def total_hours(audio_path):
    total_seconds = 0
    for root, _, files in os.walk(audio_path):

        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                info = sf.info(path)
                total_seconds += info.frames / info.samplerate

    print(audio_path)
    print("Total seconds:", total_seconds )
    print("Total hours:", total_seconds / 3600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help="Path to audio folder")
    args = parser.parse_args()

    total_hours = total_hours(args.audio_path)