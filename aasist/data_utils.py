import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader


def protocol_reader(protocol_path, is_eval=False):

    file_list = []
    labels = {}

    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue  
            parts = line.split()
            if not is_eval:
                label = int(parts[-1])
                path = " ".join(parts[:-1])
            else:
                path = parts[0]

            file_list.append(path)
            if not is_eval:
                labels[path] = label

    if is_eval:
        return file_list
    else:
        return labels, file_list
    
def select_class(files, labels, target_class):
    return [f for f in files if labels[f] == target_class]
def get_loader(seed, protocols, config):

    gen = torch.Generator()
    gen.manual_seed(seed)

    print("Reading all the protocol readers... (●'◡'●)")

    sf_train_labels, sf_train_files = protocol_reader(protocols["scenefake_train_protocol"])
    sf_val_labels, sf_val_files     = protocol_reader(protocols["scenefake_val_protocol"])
    sf_test_labels, sf_test_files   = protocol_reader(protocols["scenefake_test_protocol"])

    wild_train_labels, wild_train_files = protocol_reader(protocols["wild_train_protocol"])
    wild_val_labels, wild_val_files     = protocol_reader(protocols["wild_val_protocol"])
    wild_test_labels, wild_test_files   = protocol_reader(protocols["wild_test_protocol"])

    print("loaded all the protocols, now filtering out.. ")

    print("loading training file")

    train_datasets = []
    real_count = 0
    fake_count = 0

    if protocols["real_train"] in ["wild", "both"]:
        wild_real = select_class(wild_train_files, wild_train_labels, 0)
        train_datasets.append(
            SceneFakeWildTrainDataset(wild_real, {f: 0 for f in wild_real})
        )
        real_count += len(wild_real)
        print(f"Loaded training real from wild: {len(wild_real)}")

    if protocols["real_train"] in ["scenefake", "both"]:
        sf_real = select_class(sf_train_files, sf_train_labels, 0)
        train_datasets.append(
            SceneFakeTrainDataset(sf_real, {f: 0 for f in sf_real})
        )
        real_count += len(sf_real)
        print(f"Loaded training real from scenefake: {len(sf_real)}")

    fake_files = select_class(sf_train_files, sf_train_labels, 1)
    train_datasets.append(
        SceneFakeTrainDataset(fake_files, {f: 1 for f in fake_files})
    )
    fake_count += len(fake_files)

    print(f"Loaded fake classes for training: {fake_count}")

    train_set = ConcatDataset(train_datasets)

    print(f"Total training files: real={real_count}, fake={fake_count}")
    print("࣪ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ")



    print("loading validations file")

    dev_datasets = []
    val_real_count = 0
    val_fake_count = 0

    if protocols["real_val"] in ["wild", "both"]:
        wild_real_val = select_class(wild_val_files, wild_val_labels, 0)
        dev_datasets.append(
            SceneFakeWildEvalDataset(
                wild_real_val,
                {f: 0 for f in wild_real_val}
            )
        )
        val_real_count += len(wild_real_val)

    if protocols["real_val"] in ["scenefake", "both"]:
        sf_real_val = select_class(sf_val_files, sf_val_labels, 0)
        dev_datasets.append(
            SceneFakeEvalDataset(
                sf_real_val,
                {f: 0 for f in sf_real_val}
            )
        )
        val_real_count += len(sf_real_val)

    val_fake_files = select_class(sf_val_files, sf_val_labels, 1)
    dev_datasets.append(
        SceneFakeEvalDataset(
            val_fake_files,
            {f: 1 for f in val_fake_files}
        )
    )
    val_fake_count += len(val_fake_files)

    dev_set = ConcatDataset(dev_datasets)

    print(f"Total validation files: real={val_real_count}, fake={val_fake_count}")
    print("࣪ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ")




    print("loading testing file")

    eval_datasets = []
    test_real_count = 0
    test_fake_count = 0

    if protocols["real_test"] in ["wild", "both"]:
        wild_real_test = select_class(wild_test_files, wild_test_labels, 0)
        eval_datasets.append(
            SceneFakeWildEvalDataset(
                wild_real_test,
                {f: 0 for f in wild_real_test}
            )
        )
        test_real_count += len(wild_real_test)

    if protocols["real_test"] in ["scenefake", "both"]:
        sf_real_test = select_class(sf_test_files, sf_test_labels, 0)
        eval_datasets.append(
            SceneFakeEvalDataset(
                sf_real_test,
                {f: 0 for f in sf_real_test}
            )
        )
        test_real_count += len(sf_real_test)

    test_fake_files = select_class(sf_test_files, sf_test_labels, 1)
    eval_datasets.append(
        SceneFakeEvalDataset(
            test_fake_files,
            {f: 1 for f in test_fake_files}
        )
    )
    test_fake_count += len(test_fake_files)

    eval_set = ConcatDataset(eval_datasets)

    print(f"Total testing files: real={test_real_count}, fake={test_fake_count}")
    print("test dataset class created..")
    print("࣪ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ")

    total = real_count + fake_count
    class_weights = torch.tensor(
        [fake_count / total, real_count / total],
        dtype=torch.float32
    )

    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    print("dataloaders are ready ˚.🎀༘⋆")

    return trn_loader, dev_loader, eval_loader, class_weights



# -- unka scenefkae loader -- 

class SceneFakeTrainDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels
     
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        path = self.file_list[index]
        audio, _ = sf.read(path)

        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)

        # force fixed length
        cut = 64600
        if len(audio) >= cut:
            audio = audio[:cut]
        else:
            audio = np.pad(audio, (0, cut - len(audio)))

        label = self.labels[path]
        return torch.tensor(audio, dtype=torch.float32), label



class SceneFakeEvalDataset(Dataset):
  
    def __init__(self, file_list, labels=None,cut=64600):
        self.file_list = file_list
        self.labels = labels
        self.cut = cut

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path = self.file_list[index]
        audio, _ = sf.read(path)

        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)

        # FORCE FIXED LENGTH
        if len(audio) >= self.cut:
            audio = audio[:self.cut]
        else:
            audio = np.pad(audio, (0, self.cut - len(audio)))

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        if self.labels is None:
            return audio_tensor, path

        label = self.labels[path]
        return audio_tensor, label, path



# -- apna scenefake laoder -- 
class SceneFakeWildTrainDataset(Dataset):

    def __init__(self, file_list, labels, cut=64600, sr=16000):
        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr
        self.hop = cut 
        self.index_map = []

        self._build_index()

    def _build_index(self):
        for path in self.file_list:
            with sf.SoundFile(path) as f:
                total_len = len(f)

            start = 0
            while start < total_len:
                self.index_map.append((path, start))
                start += self.hop

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        path, start = self.index_map[index]

        with sf.SoundFile(path) as f:
            f.seek(start)
            audio = f.read(self.cut)

        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)

        # audio = audio.astype(np.float32)
        if len(audio) < self.cut:
            audio = np.pad(audio, (0, self.cut - len(audio)))

        audio = audio.astype(np.float32)

        label = self.labels[path]
        return torch.tensor(audio), label
    
class SceneFakeWildEvalDataset(Dataset):

    def __init__(self, file_list, labels=None, cut=64600, sr=16000):
        self.file_list = file_list
        self.labels = labels
        self.cut = cut
        self.sr = sr
        self.hop = cut # 1 sec overlap
        self.index_map = []

        self._build_index()

    def _build_index(self):
        for path in self.file_list:
            with sf.SoundFile(path) as f:
                total_len = len(f)

            start = 0
            while start < total_len:
                self.index_map.append((path, start))
                start += self.hop

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        path, start = self.index_map[index]

        with sf.SoundFile(path) as f:
            f.seek(start)
            audio = f.read(self.cut)

        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)

        if len(audio) < self.cut:
            audio = np.pad(audio, (0, self.cut - len(audio)))

        audio = audio.astype(np.float32)
        audio_tensor = torch.tensor(audio)

        if self.labels is None:
            return audio_tensor, path

        label = self.labels[path]
        return audio_tensor, label, path
