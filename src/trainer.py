"""
trainer.py — Training Loop, Evaluation, and Early Stopping
============================================================
Provides a unified Trainer class compatible with all three model
architectures. Implements:
  - Mixed-precision training (torch.cuda.amp)
  - Early stopping with configurable patience
  - Learning rate scheduling (ReduceLROnPlateau / linear warmup for XLM-R)
  - Per-epoch metric logging
  - McNemar's test for statistical significance between model pairs
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Stub torch.no_grad for environments without PyTorch
if not TORCH_AVAILABLE:
    import functools
    def _no_grad():
        """Returns a decorator that is a no-op (torch not available)."""
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw): return fn(*a, **kw)
            return wrapper
        return decorator
    class _FakeTorch:
        no_grad = staticmethod(_no_grad)
    torch = _FakeTorch()


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation F1 stops improving.

    Args:
        patience: epochs to wait before stopping (default 3, per proposal)
        min_delta: minimum improvement to qualify as progress
        mode: 'max' for F1/accuracy, 'min' for loss
    """
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf
        self.counter = 0
        self.best_state = None

    def step(self, metric: float, model) -> bool:
        """Returns True if training should stop."""
        improved = (
            metric > self.best + self.min_delta if self.mode == 'max'
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
            # Save best model weights (CPU copy to avoid GPU memory issues)
            if TORCH_AVAILABLE:
                import copy
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model):
        """Load the best saved weights back into the model."""
        if self.best_state is not None and TORCH_AVAILABLE:
            model.load_state_dict(self.best_state)


# ─── Metric Computation ───────────────────────────────────────────────────────

def compute_metrics(y_true: List[int], y_pred: List[int],
                    label_names: Optional[List[str]] = None) -> Dict:
    """
    Compute all evaluation metrics for a classification run.

    Returns dict with:
        accuracy, f1_macro, precision_macro, recall_macro,
        f1_per_class, classification_report_str, confusion_matrix
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    observed_labels = sorted(set(y_true) | set(y_pred))
    if label_names is not None:
        if len(label_names) > max(observed_labels):
            target_names = [label_names[i] for i in observed_labels]
        elif len(label_names) == len(observed_labels):
            target_names = label_names
        else:
            target_names = None
    else:
        target_names = None

    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 6),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 6),
        'precision': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 6),
        'recall': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 6),
        'f1_per_class': f1_score(y_true, y_pred, average=None, labels=observed_labels,
                                 zero_division=0).tolist(),
        'report': classification_report(
            y_true, y_pred, labels=observed_labels,
            target_names=target_names, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=observed_labels).tolist(),
    }
    return metrics


def mcnemar_test(preds_a: List[int], preds_b: List[int],
                 y_true: List[int]) -> Tuple[float, float]:
    """
    McNemar's test for statistical significance between two classifiers.

    Tests H0: both classifiers have the same error rate.
    Uses the continuity-corrected form (Edwards, 1948):
        χ² = (|b - c| - 1)² / (b + c)

    where b = A correct, B wrong; c = A wrong, B correct.

    Returns: (chi2_statistic, p_value)
    """
    from scipy.stats import chi2 as chi2_dist

    b, c = 0, 0
    for pa, pb, yt in zip(preds_a, preds_b, y_true):
        a_correct = (pa == yt)
        b_correct = (pb == yt)
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    if b + c == 0:
        return 0.0, 1.0  # indistinguishable

    # Edwards continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return round(chi2, 4), round(p_value, 6)


# ─── Unified Trainer ─────────────────────────────────────────────────────────

class Trainer:
    """
    Unified training and evaluation loop for TextCNN, BiLSTM-Attn, and XLM-R.

    Hyperparameters (from proposal §3.4):
        - TextCNN/BiLSTM: Adam, lr=1e-3, batch=64, epochs=30
        - XLM-R: AdamW, encoder_lr=1e-5, head_lr=1e-3, batch=32, epochs=10
        - Weight decay: 1e-4 (all models)
        - Early stopping patience: 3
        - Mixed precision: enabled on CUDA
    """

    DEFAULT_HP = {
        'TextCNN':      {'lr': 1e-3, 'batch_size': 64, 'epochs': 30, 'weight_decay': 1e-4},
        'BiLSTM-Attn':  {'lr': 1e-3, 'batch_size': 64, 'epochs': 30, 'weight_decay': 1e-4},
        'XLM-R':        {'lr': 1e-5, 'batch_size': 32, 'epochs': 10, 'weight_decay': 1e-2},
    }

    def __init__(self, model, model_name: str, device: str = 'auto',
                 class_weights=None, label_names=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for Trainer.")

        self.model = model
        self.model_name = model_name
        self.label_names = label_names

        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        # Loss function (with optional class weighting for imbalance)
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Mixed precision scaler (no-op on CPU)
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')

        self.history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    def _get_optimizer(self, hp: Dict):
        """Build optimizer appropriate to model type."""
        if self.model_name == 'XLM-R' and hasattr(self.model, 'get_optimizer_groups'):
            param_groups = self.model.get_optimizer_groups(
                lr_encoder=hp['lr'], lr_head=hp['lr'] * 100
            )
            return torch.optim.AdamW(param_groups, weight_decay=hp['weight_decay'])
        return torch.optim.Adam(
            self.model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
        )

    def _train_epoch(self, loader, optimizer) -> Tuple[float, float]:
        """Single training epoch. Returns (avg_loss, macro_f1)."""
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for batch in loader:
            optimizer.zero_grad()
            labels = batch['label'].to(self.device)

            with autocast(enabled=self.device.type == 'cuda'):
                if self.model_name == 'XLM-R':
                    logits = self.model(
                        batch['input_ids'].to(self.device),
                        batch['attention_mask'].to(self.device)
                    )
                else:
                    logits = self.model(batch['input_ids'].to(self.device))

                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            # Gradient clipping (important for RNNs — prevents exploding gradients)
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return avg_loss, f1

    @torch.no_grad()
    def _eval_epoch(self, loader) -> Tuple[float, float, List, List]:
        """Evaluation pass. Returns (avg_loss, macro_f1, all_preds, all_labels)."""
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        for batch in loader:
            labels = batch['label'].to(self.device)
            if self.model_name == 'XLM-R':
                logits = self.model(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
            else:
                logits = self.model(batch['input_ids'].to(self.device))

            loss = self.criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return avg_loss, f1, all_preds, all_labels

    def fit(self, train_loader, val_loader,
            hp: Optional[Dict] = None) -> Dict:
        """
        Full training loop with early stopping.

        Returns training history dict with per-epoch metrics.
        """
        if hp is None:
            hp = self.DEFAULT_HP.get(self.model_name, self.DEFAULT_HP['BiLSTM-Attn'])

        optimizer = self._get_optimizer(hp)
        # LR scheduler: reduce on plateau (patience=2, factor=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=2, factor=0.5
        )
        early_stopper = EarlyStopping(patience=3, mode='max')

        print(f"    Training on {self.device} | "
              f"lr={hp['lr']:.0e} batch={hp['batch_size']} epochs={hp['epochs']}")

        t_start = time.time()
        for epoch in range(1, hp['epochs'] + 1):
            train_loss, train_f1 = self._train_epoch(train_loader, optimizer)
            val_loss, val_f1, _, _ = self._eval_epoch(val_loader)
            scheduler.step(val_f1)

            self.history['train_loss'].append(round(train_loss, 5))
            self.history['val_loss'].append(round(val_loss, 5))
            self.history['train_f1'].append(round(train_f1, 5))
            self.history['val_f1'].append(round(val_f1, 5))

            if epoch % 5 == 0 or epoch == 1:
                print(f"      Epoch {epoch:3d} | "
                      f"Train F1={train_f1:.4f} Loss={train_loss:.4f} | "
                      f"Val F1={val_f1:.4f} Loss={val_loss:.4f}")

            if early_stopper.step(val_f1, self.model):
                print(f"      Early stopping at epoch {epoch} "
                      f"(best val F1={early_stopper.best:.4f})")
                early_stopper.restore_best(self.model)
                break

        self.history['train_time'] = round(time.time() - t_start, 1)
        return self.history

    def evaluate(self, test_loader) -> Dict:
        """Full evaluation on test set with all metrics."""
        t_inf = time.time()
        _, _, preds, labels = self._eval_epoch(test_loader)
        inference_time = (time.time() - t_inf) / len(labels)

        metrics = compute_metrics(labels, preds, self.label_names)
        metrics['inference_time'] = round(inference_time, 6)
        metrics['train_time'] = self.history.get('train_time', 0)
        return metrics

    def save_checkpoint(self, path: str):
        """Save model weights and training history."""
        if TORCH_AVAILABLE:
            torch.save({
                'model_state': self.model.state_dict(),
                'history': self.history,
                'model_name': self.model_name,
            }, path)
            print(f"    [Saved] Checkpoint → {path}")


# ─── Learning Rate Scheduler for XLM-R ───────────────────────────────────────

class LinearWarmupScheduler:
    """
    Linear warmup + linear decay scheduler for transformer fine-tuning.
    Commonly used with BERT-family models (Devlin et al., 2019).

    Warmup phase: lr increases linearly from 0 to peak_lr over warmup_steps
    Decay phase:  lr decreases linearly from peak_lr to 0 over remaining steps
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        s = self.current_step
        if s <= self.warmup_steps:
            scale = s / self.warmup_steps
        else:
            scale = max(0.0, (self.total_steps - s) / (self.total_steps - self.warmup_steps))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale
