from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score


def evaluate_eer_utterance(loader, model, device):
# save logits per segment and then avg it to find the score and then find eer

    model.eval()

    logits_dict = defaultdict(list)
    labels_dict = {}

    with torch.no_grad():
        for batch_x, batch_y, batch_path in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, logits = model(batch_x, Freq_aug=False)

            for logit, label, path in zip(logits, batch_y, batch_path):
                logits_dict[path].append(logit.cpu())
                labels_dict[path] = label.item()

    final_scores = []
    final_labels = []

    for path in logits_dict:

        stacked = torch.stack(logits_dict[path])

        mean_logit = stacked.mean(dim=0)

        score = F.softmax(mean_logit, dim=0)[1].item()

        final_scores.append(score)
        final_labels.append(labels_dict[path])

    final_scores = np.array(final_scores)
    final_labels = np.array(final_labels)

    fpr, tpr, _ = roc_curve(final_labels, final_scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))] * 100
    print("Num utterances:", len(final_scores))
    print("Num real:", np.sum(final_labels == 0))
    print("Num fake:", np.sum(final_labels == 1))

    return eer



def evaluate_eer_utterance_avg_softmax(loader, model, device):
# save logits per segment and then avg it to find the score and then find eer
    model.eval()

    logits_dict = defaultdict(list)
    labels_dict = {}

    with torch.no_grad():
        for batch_x, batch_y, batch_path in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, logits = model(batch_x, Freq_aug=False)

            for logit, label, path in zip(logits, batch_y, batch_path):
                logits_dict[path].append(logit.cpu())
                labels_dict[path] = label.item()

    final_scores = []
    final_labels = []

    for path in logits_dict:
        stacked = torch.stack(logits_dict[path]) # [num_segments, 2]
    
        # Apply softmax to each segment first to get probabilities
        probs = F.softmax(stacked, dim=1)
    
        # Average the probability of "Fake" across all segments
        mean_fake_prob = probs[:, 1].mean().item()
    
        final_scores.append(mean_fake_prob)
        final_labels.append(labels_dict[path]) 

    final_scores = np.array(final_scores)
    final_labels = np.array(final_labels)

    fpr, tpr, _ = roc_curve(final_labels, final_scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))] * 100
    print("Num utterances:", len(final_scores))
    print("Num real:", np.sum(final_labels == 0))
    print("Num fake:", np.sum(final_labels == 1))

    return eer


def evaluate_confusion_utterance(loader, model, device):

    model.eval()

    logits_dict = defaultdict(list)
    labels_dict = {}

    with torch.no_grad():
        for batch_x, batch_y, batch_path in loader:

            batch_x = batch_x.to(device)

            _, logits = model(batch_x, Freq_aug=False)

            for logit, label, path in zip(logits, batch_y, batch_path):
                logits_dict[path].append(logit.cpu())
                labels_dict[path] = label.item()

    y_true = []
    y_pred = []

    # ---- aggregate segments -> utterance ----
    for path in logits_dict:

        stacked = torch.stack(logits_dict[path])
        mean_logit = stacked.mean(dim=0)

        score = F.softmax(mean_logit, dim=0)[1].item()
        pred = 1 if score >= 0.5 else 0

        y_true.append(labels_dict[path])
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred) * 100

    print("Num utterances:", len(y_true))
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.2f}%")

    return cm, acc, y_true, y_pred

def build_eval_loader_from_protocol(protocol_path, config, seed):

    from data_utils import get_loader

    config_copy = dict(config)
    config_copy["protocols"] = dict(config["protocols"])
    config_copy["protocols"]["scenefake_test_protocol"] = protocol_path

    _, _, eval_loader, _ = get_loader(seed, config_copy["protocols"], config_copy)

    return eval_loader


def evaluate_kfold_from_protocols(fold_protocols, config, args, model, device):

    print("\nStarting K-Fold Evaluation (Independent Splits)\n")

    all_y_true = []
    all_y_pred = []

    fold_results = []

    for i, proto in enumerate(fold_protocols):

        print(f"Evaluating Fold {i}")
        print(f"Protocol: {proto}")

        eval_loader = build_eval_loader_from_protocol(
            proto, config, args.seed
        )

        cm, acc, y_true, y_pred = evaluate_confusion_utterance(
            eval_loader, model, device
        )

        fold_results.append({
            "fold": i,
            "cm": cm,
            "acc": acc
        })

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)


    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    global_cm = confusion_matrix(all_y_true, all_y_pred)
    global_acc = accuracy_score(all_y_true, all_y_pred) * 100

    print("\nGlobal Confusion Matrix:\n", global_cm)
    print(f"Global Accuracy: {global_acc:.2f}%")


    with open("kfold_confusion_results.txt", "w") as f:

        for result in fold_results:
            f.write(f"\nFold {result['fold']}\n")
            f.write(f"Confusion Matrix:\n{result['cm']}\n")
            f.write(f"Accuracy: {result['acc']:.2f}%\n")


        f.write(f"Global Confusion Matrix:\n{global_cm}\n")
        f.write(f"Global Accuracy: {global_acc:.2f}%\n")

    print("\nK-Fold Evaluation Finished.\n")