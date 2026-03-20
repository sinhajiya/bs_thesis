"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import datetime
import datetime
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (protocol_reader, OurTrainDataset, OurEvalDataset)
from evaluation import evaluate_eer_utterance
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

def main(args: argparse.Namespace) -> None:

    print("Loading config from {}".format(args.config))

    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    # ---------------- DEVICE ----------------
    print("Checking device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cpu":
        raise ValueError("GPU required")

    # ---------------- CONFIG ----------------
    print("Experiment config:")
    for key, val in config.items():
        print("{}: {}".format(key, val))

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    set_seed(args.seed, config)

    # ---------------- EVAL MODE ----------------

    if args.eval:

        print("Evaluation mode")

        model = get_model(model_config, device)

        model.load_state_dict(
            torch.load(config["model_path"], map_location=device)
        )

        print("Loaded model:", config["model_path"])

        _,  seen_loader, unseen_loader = get_loader(
            args.seed,
            config,
     
        )
        print("SEEN DATA TESTING")
        seen_eer,_,_= evaluate_eer_utterance(seen_loader, model, device)
        print("unSEEN DATA TESTING")
        unseen_eer,_,_ = evaluate_eer_utterance(unseen_loader, model, device)

        sys.exit(0)
        
    # ---------------- OUTPUT DIR ----------------

    output_dir = Path(args.output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_tag = output_dir / f"{config['dataset_name']}_{config['suffix']}_{timestamp}"
    model_save_path = model_tag / "weights"
    model_save_path.mkdir(parents=True, exist_ok=True)
    copy(args.config, model_tag / "config.json")

    # ---------------- TRAINING ----------------

    trn_loader, seen_loader, unseen_loader = get_loader(
            args.seed,
            config
        )
    model = get_model(model_config, device)

    optim_config["steps_per_epoch"] = len(trn_loader)

    optimizer, scheduler = create_optimizer(
        model.parameters(),
        optim_config
    )

    optimizer_swa = SWA(optimizer)

    n_swa_update = 0

    for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs"):

        print("Start training epoch {:02d}".format(epoch))

        running_loss = train_epoch(
            trn_loader,
            model,
            optimizer,
            device,
            scheduler,
            config
        )
    
        print(
            "⊹ ࣪ ﹏𓊝﹏𓂁﹏⊹ ࣪ ˖ Loss:{:.5f} ⊹ ࣪ ﹏𓊝﹏𓂁﹏⊹ ࣪ ˖".format(
                running_loss,
                
            )
        )
    
        torch.save(
            model.state_dict(),
            model_save_path / f"MODEL_{epoch}.pth"
        )

        optimizer_swa.update_swa()
        n_swa_update += 1

        # -------- SWA EVAL --------

        print("Running SWA evaluation")

        if epoch > 0.75*config["num_epochs"] > 0:

            optimizer_swa.swap_swa_sgd()

            optimizer_swa.bn_update(
                trn_loader,
                model,
                device=device
            )

            print("Seen Data Evaluation....")
            _,_,_ = evaluate_eer_utterance(
                seen_loader,
                model,
                device
            )

            print("unSeen Data Evaluation....")

            _,_,_ = evaluate_eer_utterance(
                unseen_loader,
                model,
                device
            )
    print("\nTraining finished")



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_loader(seed: int, config: dict):

    gen = torch.Generator()
    gen.manual_seed(seed)
    pp = Path(config["protocol_path"])
    train_protocol = pp / "train.txt"
    unseen_protocol = pp / "unseen_test.txt"
    seen_protocol = pp/ "seen_test.txt"

    # -------- LOAD FULL TRAIN DATA --------
    labels, files = protocol_reader(train_protocol)
    
    print("total training files:", len(files))

    # -------- TRAIN DATASET --------
    train_set = OurTrainDataset(
        file_list=files,
        labels=labels
    )

    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=config.get("num_workers", 4),
        worker_init_fn=seed_worker,
        generator=gen
    )


    # -------- SEEN TEST DATASET --------
    seen_labels, seen_files = protocol_reader(seen_protocol)

    print("seen test files:", len(seen_files))

    seen_set = OurEvalDataset(
        file_list=seen_files,
        labels=seen_labels
    )

    seen_loader = DataLoader(
        seen_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.get("num_workers", 4)
    )

    # -------- UNSEEN TEST DATASET --------
    unseen_labels, unseen_files = protocol_reader(unseen_protocol)

    print("unseen test files:", len(unseen_files))

    unseen_set = OurEvalDataset(
        file_list=unseen_files,
        labels=unseen_labels
    )

    unseen_loader = DataLoader(
        unseen_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.get("num_workers", 4)
    )

    return trn_loader, seen_loader, unseen_loader

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
