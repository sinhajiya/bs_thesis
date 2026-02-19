from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve


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

#  36.667%