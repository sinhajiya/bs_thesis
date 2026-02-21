import pandas as pd
from pathlib import Path

ROOT = Path("/home/bs_thesis/datasets/OurDataset")
REAL_CSV = ROOT / "SceneFake-Wild-Real-Info.csv"
REAL_AUDIO_DIR = ROOT / "SceneFake-Wild-Real"
SPLIT_DIR = ROOT / "generated_protocols_v12"
SPLIT_DIR.mkdir(exist_ok=True)
UNSEEN_SCENES = ["s07", "s12", "s08"]


df = pd.read_csv(REAL_CSV)

df = df.rename(columns={
    "File name": "file",
    "Speaker Id": "speaker",
    "Scene": "scene",
    "Time (s)": "duration"
})


df_unseen = df[df["scene"].isin(UNSEEN_SCENES)].copy()
df_seen   = df[~df["scene"].isin(UNSEEN_SCENES)].copy()

speaker_durations = (
    df_seen
    .groupby("speaker")["duration"]
    .sum()
    .reset_index()
)

speaker_durations = speaker_durations.sort_values(
    by="duration", ascending=False
).reset_index(drop=True)

total_duration_seen = speaker_durations["duration"].sum()

train_target = 0.75 * total_duration_seen
val_target   = 0.125 * total_duration_seen
test_target  = 0.125 * total_duration_seen

train_spk, val_spk, test_spk = set(), set(), set()
train_sum = val_sum = test_sum = 0

for _, row in speaker_durations.iterrows():

    spk = row["speaker"]
    dur = row["duration"]

    gaps = {
        "train": train_target - train_sum,
        "val": val_target - val_sum,
        "test": test_target - test_sum,
    }

    split_choice = max(gaps, key=gaps.get)

    if split_choice == "train":
        train_spk.add(spk)
        train_sum += dur
    elif split_choice == "val":
        val_spk.add(spk)
        val_sum += dur
    else:
        test_spk.add(spk)
        test_sum += dur

assert train_spk.isdisjoint(val_spk)
assert train_spk.isdisjoint(test_spk)
assert val_spk.isdisjoint(test_spk)

def assign_split(spk):
    if spk in train_spk:
        return "train"
    elif spk in val_spk:
        return "val"
    else:
        return "test_seen"

df_seen["split"] = df_seen["speaker"].apply(assign_split)
df_unseen["split"] = "test_unseen"

df_final = pd.concat([df_seen, df_unseen], ignore_index=True)

def write_protocol(split_name):
    subset = df_final[df_final["split"] == split_name]
    out_path = SPLIT_DIR / f"{split_name}.txt"

    with open(out_path, "w") as f:
        for _, row in subset.iterrows():
            path = REAL_AUDIO_DIR / row["file"]
            f.write(f"{path.resolve()} 0\n")

    print(f"{split_name}: {len(subset)} utterances written")

write_protocol("train")
write_protocol("val")
write_protocol("test_seen")
write_protocol("test_unseen")


def split_hours(split_name):
    subset = df_final[df_final["split"] == split_name]
    return subset["duration"].sum() / 3600

train_hours = split_hours("train")
val_hours = split_hours("val")
test_seen_hours = split_hours("test_seen")
test_unseen_hours = split_hours("test_unseen")

total_hours = df_final["duration"].sum() / 3600

print(f"Train hours        : {train_hours:.2f}")
print(f"Val hours          : {val_hours:.2f}")
print(f"Test (seen) hours  : {test_seen_hours:.2f}")
print(f"Test (unseen) hours: {test_unseen_hours:.2f}")
print(f"\nTotal hours (all)  : {total_hours:.2f}")
