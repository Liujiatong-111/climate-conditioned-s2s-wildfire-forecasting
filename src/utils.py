"""Brief implementation note."""
import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from typing import Dict


def set_seed(seed: int = 42):
    """Brief implementation note."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray = None, threshold: float = 0.5) -> Dict[str, float]:
    """Brief implementation note."""
    
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    
    if mask is not None:
        mask_flat = mask.reshape(-1)
        valid_mask = (mask_flat == 0)  
        preds_flat = preds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

    
    preds_binary = (preds_flat >= threshold).astype(int)
    targets_binary = targets_flat.astype(int)

    
    acc = accuracy_score(targets_binary, preds_binary)
    prec = precision_score(targets_binary, preds_binary, zero_division=0)
    rec = recall_score(targets_binary, preds_binary, zero_division=0)
    f1 = f1_score(targets_binary, preds_binary, zero_division=0)

    
    
    try:
        if len(np.unique(targets_binary)) > 1:  
            auroc = roc_auc_score(targets_binary, preds_flat)
            auprc = average_precision_score(targets_binary, preds_flat)
        else:
            
            auroc = 0.0
            auprc = 0.0
    except Exception as e:
        
        print(f"Warning: Could not compute AUROC/AUPRC: {e}")
        auroc = 0.0
        auprc = 0.0

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }


def create_output_dir(output_dir: str):
    """Brief implementation note."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Status")


def get_model_name(use_local: bool, use_global: bool, use_oci: bool) -> str:
    """Brief implementation note."""
    l = '1' if use_local else '0'
    g = '1' if use_global else '0'
    i = '1' if use_oci else '0'
    return f"L{l}_G{g}_I{i}"


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Brief implementation note."""
    print(f"{prefix}Metrics:")
    for key, value in metrics.items():
        
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


class AverageMeter:
    """Brief implementation note."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
