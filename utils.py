import torch
import random
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve


def set_seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_ood_score(net, ood_detector, test_loader, multi_class):
    ood_scores = []
    for i, (x_test, y_test) in enumerate(tqdm(test_loader)):
        x_test = x_test.cuda()
        with torch.no_grad():
            features = net.get_all_blocks(x_test)
            pred = ood_detector.eval()(features.detach())
            # KIRBY-R
            if multi_class:
                pred = torch.softmax(pred, dim=-1)[..., -1].view(-1, 1).detach().cpu().numpy()
            # KIRBY-B
            else:
                pred = torch.sigmoid(pred).view(-1, 1).detach().cpu().numpy()
            ood_scores.append(pred)
    ood_scores = np.concatenate(ood_scores, axis=0).flatten()
    return ood_scores


def get_ood_performance(_pos, _neg):
    ood_pred = _pos.flatten()
    ind_pred = _neg.flatten()
    ood_true = np.ones_like(ood_pred)
    ind_true = np.zeros_like(ind_pred)

    pred = np.concatenate([ood_pred, ind_pred], axis=0)
    true = np.concatenate([ood_true, ind_true], axis=0)
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-9)

    # calc ood metrics
    test_auroc = roc_auc_score(true, pred)

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true, pred)
    test_auprc_out = auc(recall, precision)

    # False positive rate at 95% recall
    th = thresholds[np.argmin(np.abs(recall - 0.95))]
    binary_pred = np.uint8(np.array(pred) >= th)
    negative_idx = np.where(np.array(true) == 0)[0]
    fp = np.sum(binary_pred[negative_idx]) / (len(negative_idx) + 1e-9)

    return test_auroc, test_auprc_out, fp


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone