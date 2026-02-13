
from pathlib import Path


TARGET_SR = 16000
SNR_CHOICES = [-5, 0, 5, 10, 15, 20]

# background scenes info
SCENE_GROUPS = {
    "indoor_room": [
        "keyboard_typing",
        "mouse_click",
        
        "door_wood_creaks",
        "washing_machine",
        "door_knock"
    ],
    "nature": [
        "wind",
        "rain",
        "water_drops"
    ],
    "urban_outdoor": [
        "car_horn",
        "engine",
        "footsteps"
    ],
    "Animals": [
        "insects",
        "dog",
        "crow",
        "crickets",
        "chirping_birds",

    ],

    "Human": [
        "laughing",
        "breathing",
        "drinking_sipping",
        "coughing",
        "footsteps"
    ]
}

ESC50_META_CSV = Path(
    "/home/bs_thesis/Documents/EnvDatasets/ESC-50-master/meta/esc50.csv"
)


# path info
INPUT_ORIGINAL_DIR = Path("/home/bs_thesis/Documents/OurDataset/SceneFake-Wild-Real")
OUTPUT_DENOISED_DIR = Path("/home/bs_thesis/Documents/OurDataset/SceneFake-Wild-DenoisedOriginals")
OUTPUT_FAKE_SCENES_DIR = Path("/home/bs_thesis/Documents/OurDataset/SceneFake-Wild-Fake")
NOISE_DIR = Path("/home/bs_thesis/Documents/EnvDatasets/ESC-50-master/audio")

SCENE_META_CSV = Path("/home/bs_thesis/Documents/OurDataset/SceneFake-Wild-SceneMetadata.csv")

