import soundfile as sf
import random

PROTO = "/home/bs_thesis/datasets/SceneFakeDataset/protocols/scenefake_eval.txt"

TARGET_HOURS = 0.67
TARGET_SECONDS = TARGET_HOURS * 3600


real_files = []
fake_files = []

with open(PROTO) as f:
    for line in f:
        path, label = line.strip().split()
        if label == "0":
            real_files.append(path)
        else:
            fake_files.append(path)


def get_duration(path):
    return sf.info(path).duration

real_pool = [(f, get_duration(f)) for f in real_files]
fake_pool = [(f, get_duration(f)) for f in fake_files]

random.shuffle(real_pool)
random.shuffle(fake_pool)


folds = []
real_idx = 0
fake_idx = 0

while True:

    fold_real = []
    fold_fake = []

    real_time = 0
    fake_time = 0

    while real_idx < len(real_pool) and real_time < TARGET_SECONDS:
        file, dur = real_pool[real_idx]
        fold_real.append((file, dur))
        real_time += dur
        real_idx += 1

    while fake_idx < len(fake_pool) and fake_time < TARGET_SECONDS:
        file, dur = fake_pool[fake_idx]
        fold_fake.append((file, dur))
        fake_time += dur
        fake_idx += 1

    if real_time < TARGET_SECONDS or fake_time < TARGET_SECONDS:
        break

    folds.append((fold_real, fold_fake))

print("Total folds created:", len(folds))


for i, (fold_real, fold_fake) in enumerate(folds):

    total_real = sum(d for _, d in fold_real)
    total_fake = sum(d for _, d in fold_fake)

    print(f"\nFold {i}")
    print("Real hours:", total_real / 3600)
    print("Fake hours:", total_fake / 3600)

    with open(f"scenefake_eval_fold{i}.txt", "w") as f:
        for file, _ in fold_real:
            f.write(f"{file} 0\n")
        for file, _ in fold_fake:
            f.write(f"{file} 1\n")