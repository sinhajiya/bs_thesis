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

def get_loader( seed, protocols, config):

    gen = torch.Generator()
    gen.manual_seed(seed)

    sf_train_labels, sf_train_files = protocol_reader(protocols["scenefake_train_protocol"])
    sf_val_labels, sf_val_files     = protocol_reader(protocols["scenefake_val_protocol"])
    sf_test_labels, sf_test_files   = protocol_reader(protocols["scenefake_test_protocol"])

    wild_train_labels, wild_train_files = protocol_reader(protocols["wild_train_protocol"])
    wild_val_labels, wild_val_files     = protocol_reader(protocols["wild_val_protocol"])
    wild_test_labels, wild_test_files     = protocol_reader(protocols["wild_test_protocol"])

    train_files = []
    train_labels = {}

    print("loaded all the protocols, now filtering out.. ")

    print("loading training file")
    if protocols["real_train"] == "wild":
        print("training k liye, apne real audios being used")
        real_files = select_class(wild_train_files, wild_train_labels, 0)
    elif protocols["real_train"] == "scenefake":
        print("training k liye, unke real audios being used")
        real_files = select_class(sf_train_files, sf_train_labels, 0)
    else:
        raise ValueError("Invalid real_train")

    print("loaded unke fake classes for training")
    fake_files = select_class(sf_train_files, sf_train_labels, 1)
   

    train_files += real_files + fake_files
    train_labels.update({f: 0 for f in real_files})
    train_labels.update({f: 1 for f in fake_files})
    print(f"totaal training files are {len(train_files)} jinme se {len(real_files)} are real and {len(fake_files)} are fake..")
    print("࣪ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ")

    print("loading validations file")
    dev_files = []
    dev_labels = {}

    if protocols["real_val"] == "wild":
        print("validation k liye, apne real audios being used")

        val_real_files = select_class(wild_val_files, wild_val_labels, 0)
    else:
        print("validation k liye, unke real audios being used")
        val_real_files = select_class(sf_val_files, sf_val_labels, 0)

    val_fake_files = select_class(sf_val_files, sf_val_labels, 1)
    print("loaded unke fake classes")

    dev_files += val_real_files + val_fake_files
    dev_labels.update({f: 0 for f in val_real_files})
    dev_labels.update({f: 1 for f in val_fake_files})
    print(f"totaal validation files are {len(dev_files)} jinme se {len(val_real_files)} are real and {len(val_fake_files)} are fake..")

    print("࣪ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ")
    print("loading testing file")

    eval_files = []
    eval_labels = {}

    if protocols["real_test"] == "wild":
        print("testing k liye, apne real audios being used")
        test_real_files = select_class(wild_test_files, wild_test_labels, 0)
        
    else:
        print("testing k liye, unke real audios being used")
        test_real_files = select_class(sf_test_files, sf_test_labels, 0)

    test_fake_files = select_class(sf_test_files, sf_test_labels, 1)

    eval_files += test_real_files + test_fake_files
    eval_labels.update({f: 0 for f in test_real_files})
    eval_labels.update({f: 1 for f in test_fake_files})
    print(f"totaal testing files are {len(eval_files)} jinme se {len(test_real_files)} are real and {len(test_fake_files)} are fake..")

    print("࣪ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡ ✌︎㋡")

    print("creating dataset class..")
    train_datasets = []

    if protocols["real_train"] == "wild":
        real_dataset = SceneFakeWildTrainDataset(real_files, {f:0 for f in real_files})
    else:
        real_dataset = SceneFakeTrainDataset(real_files, {f:0 for f in real_files})

    train_datasets.append(real_dataset)

    fake_dataset = SceneFakeTrainDataset(fake_files, {f:1 for f in fake_files})
    train_datasets.append(fake_dataset)

    train_set = ConcatDataset(train_datasets)

    print("training dataset class ccreated..")

    dev_datasets = []

    if protocols["real_val"] == "wild":
        real_dataset = SceneFakeWildEvalDataset(val_real_files, {f:0 for f in val_real_files})
    else:
        real_dataset = SceneFakeEvalDataset(val_real_files, {f:0 for f in val_real_files})
    dev_datasets.append(real_dataset)

    fake_dataset = SceneFakeEvalDataset(val_fake_files, {f:1 for f in val_fake_files})

    dev_datasets.append(fake_dataset)

    dev_set = ConcatDataset(dev_datasets)

    print("dev dataset class created..")

    eval_datasets = []

    if protocols["real_test"] == "wild":
        real_dataset = SceneFakeWildEvalDataset(test_real_files, {f:0 for f in test_real_files})
    else:
        real_dataset = SceneFakeEvalDataset(test_real_files, {f:0 for f in test_real_files})

    eval_datasets.append(real_dataset)

    fake_dataset = SceneFakeEvalDataset(test_fake_files, {f:1 for f in test_fake_files})
    eval_datasets.append(fake_dataset)

    eval_set = ConcatDataset(eval_datasets)
    print("test dataset class created..")

    labels_array = np.array(list(train_labels.values()))
    real = np.sum(labels_array == 0)
    fake = np.sum(labels_array == 1)
    total = real + fake

    class_weights = torch.tensor(
        [fake / total, real / total],
        dtype=torch.float32
    )

    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            num_workers=config.get("num_workers", 4),
                            pin_memory=True)

    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=config.get("num_workers", 4),
                            pin_memory=True)

    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             num_workers=config.get("num_workers", 4),
                             pin_memory=True)

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
        self.hop = cut - sr 
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
        self.hop = cut - sr  # 1 sec overlap
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
