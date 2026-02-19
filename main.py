"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
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
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import *

from evaluation import evaluate_eer, evaluate_wild_mean
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

from evaal import *
from tqdm import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)

def main(args: argparse.Namespace) -> None:

    # load experiment configurations
    print("Loading config from {}".format(args.config))

    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    print("Experiment config:")
    for key, val in config.items():
        print("{}: {}".format(key, val))  

    # make experiment reproducible
    set_seed(args.seed, config)

    # set device
    print("Checking device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {} 👨🏻‍💻".format(device))
    if device == "cpu":
        raise ValueError("GPU not no no dont go ahead full stoppppp ✋🏻🛑⛔️")

    print("Checking database...")

    output_dir = Path(args.output_dir)
    dataset_name = config["dataset_name"]

        # define model related paths    
    model_tag = f"{dataset_name}_{config['suffix']}_ep{config['num_epochs']}"

    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    metric_path = model_tag / "metrics"

    model_save_path.mkdir(parents=True, exist_ok=True)
    metric_path.mkdir(parents=True, exist_ok=True)

    copy(args.config, model_tag / "config.json")
    checkpoint_dir = model_tag / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(model_tag)
    
    # define model architecture
    print(f"Building the {model_config['architecture']} model...")
    model = get_model(model_config, device)

    # define dataloaders
    print("Loading the data...")
    print("Loading the data...")
    trn_loader, dev_loader, eval_loader, class_weights = get_loader(args.seed, config['protocols'],config)

 
    if args.eval:
        
        print("Evaluating the model...🕵🏻 ")
        checkpoint = torch.load(
            config["model_path"],
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        print("Model loaded : {}".format(config["model_path"]))
     
        eval_eer = evaluate_eer_utterance(eval_loader, model, device)

        print(f"Eval EER: {eval_eer:.3f}%")

        
        print("Evaluation finished 🗣️")

        sys.exit(0)


    print("Setting up optimizer and scheduler...")
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    optimizer_swa = SWA(optimizer)
    scaler = torch.cuda.amp.GradScaler()

    best_dev_eer = 100.
    best_eval_eer = 100.
    n_swa_update = 0  
    
    print(f"Model: {model_config['architecture']}, optimizer: {optim_config['optimizer']}, and scheduler: {optim_config['scheduler']} ready.")

    # Training
    print(f"Starting training for {config['num_epochs']} epochs ...")

    for epoch in range(config["num_epochs"]):

        print("Start training epoch {:03d}".format(epoch+1))

        running_loss = train_epoch(trn_loader, model, optimizer, device,scheduler, scaler, config, class_weights)

        print("Validating the model...")
        dev_eer = evaluate_eer_utterance(dev_loader, model, device)

        print(f"Epoch {epoch:03d} | Loss {running_loss:.4f} | Dev EER {dev_eer:.3f}%")
        if dev_eer == 0 or dev_eer == 100:
            print("Dev EER fir se 0 aur 100 k paas hain bcccc 😭😭😭😭😭😭😭 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️ 🗣️")
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
       
    # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Epoch {epoch+1} aagya... saving the checkpoints ⓹")
            save_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_eer": best_dev_eer,
                "best_eval_eer": best_eval_eer
            }, save_path)

            print(f"Saved!!  ")


        if dev_eer <= best_dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
          
            print("Evaluating on the best model...🕵🏻🕵🏻🕵🏻🕵🏻🕵🏻🕵🏻🕵🏻🕵🏻🕵🏻🕵🏻 ")
            eval_eer = evaluate_eer_utterance(eval_loader, model, device)

            print(f"Eval EER: {eval_eer:.3f}%")

            if eval_eer < best_eval_eer:
                best_eval_eer = eval_eer
                torch.save(model.state_dict(), model_save_path / "best.pth")

            print("Saving epoch {} for swa".format(epoch))

            optimizer_swa.update_swa()
            n_swa_update += 1

        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_eval_eer", best_eval_eer, epoch)

    print("Starting final evaluation")
    epoch += 1
    if n_swa_update > 0:
    
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
        torch.save(model.state_dict(), model_save_path / "swa_final.pth")

    print("Final evaluation...")


    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
        torch.save(model.state_dict(), model_save_path / "swa_final.pth")

    final_eer = evaluate_eer_utterance(eval_loader, model, device)
    print(f"Final Eval EER: {final_eer:.3f}%")



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def train_epoch(trn_loader, model, optimizer, device, scheduler, scaler, config, class_weights):

    
    model.train()
    running_loss = 0.0
    total = 0

    # weight = torch.tensor([0.1, 0.9], dtype=torch.float32).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    pbar = tqdm(trn_loader, desc="Training", leave=False)

    for batch_x, batch_y in pbar:

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.long().to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # _, output = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
            freq_aug = str_to_bool(config.get("freq_aug", "False"))
            _, output = model(batch_x, Freq_aug=freq_aug)

            loss = criterion(output, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()

        running_loss += loss.item() * batch_x.size(0)
        total += batch_x.size(0)

        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total



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
    

    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
     
    
    main(parser.parse_args())
