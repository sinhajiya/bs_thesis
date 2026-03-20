from collections import defaultdict
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryEER
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


def evaluate_eer_utterance(loader, model, device):
    print("I evaluate EER per utterance, for each segment in the utterance, i have the logits, i do softmax ->  take the avg softmax and use the logit for predicting the label")

    model.eval()

    logits_dict = defaultdict(list)
    labels_dict = {}

    with torch.no_grad():
        for batch_x, batch_y, batch_path in tqdm(loader, desc="EVAL LOGS PER UTTERANCE.. 𐙚⋆°｡⋆♡", leave=False):
            batch_x = batch_x.to(device)
            _, logits = model(batch_x, Freq_aug=False)
            for logit, label, path in zip(logits, batch_y, batch_path):
                logits_dict[path].append(logit.cpu())
                labels_dict[path] = label.item()

    preds = []
    targets = []

    for path in logits_dict:
        stacked = torch.stack(logits_dict[path])   
        # print(f"stacked shape:{ stacked.shape}")
       
        probs = torch.softmax(stacked, dim=1)
        mean_prob = probs.mean(dim=0)

        score = mean_prob[1]   # probability of FAKE

        preds.append(score)
        targets.append(labels_dict[path])

    preds = torch.stack(preds).to(device)          # (N,)
    targets = torch.tensor(targets).to(device)     # (N,)

    metric = BinaryEER(normalization=None).to(device)
    eer = metric(preds, targets).item() * 100

    pred_labels = (preds >= 0.5).int().cpu().numpy()
    targets_np = targets.cpu().numpy()

    cm = confusion_matrix(targets_np, pred_labels)
    report = classification_report(
        targets_np,
        pred_labels,
        target_names=["real", "fake"]
    )

    print("Num utterances:", len(preds))
    print("Num real:", (targets_np == 0).sum())
    print("Num fake:", (targets_np == 1).sum())

    print("\nEER: {:.4f}".format(eer))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    return eer, cm, report
