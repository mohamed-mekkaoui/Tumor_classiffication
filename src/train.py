"""
Training Module
───────────────
Dataset, training loop, MIL aggregation, and evaluation for the
STRTransformer on random-walk sequences.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import config
from model import STRTransformer


# ──────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────

class STRSequenceDataset(Dataset):
    """One item = one random-walk sequence of embeddings + its label."""

    def __init__(self, rw_meta, rw_paths, feats, split):
        self.meta = rw_meta[rw_meta["split"] == split].reset_index(drop=True)
        self.paths = rw_paths
        self.feats = feats

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path_id = int(row["path_id"])
        path = self.paths[path_id]

        X = np.array(self.feats[path], dtype=np.float32)  # (L, D)
        y = int(row["label_id"])
        wsi_id = row["wsi_id"]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long), wsi_id


def collate_pad(batch):
    """Pads variable-length sequences to the longest in the batch."""
    Xs, ys, wsi_ids = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in Xs], dtype=torch.long)
    D = Xs[0].shape[1]
    Lmax = int(lengths.max())

    Xpad = torch.zeros(len(Xs), Lmax, D, dtype=torch.float32)
    for i, x in enumerate(Xs):
        Xpad[i, : x.shape[0]] = x

    y = torch.stack(ys)
    return Xpad, lengths, y, list(wsi_ids)


# ──────────────────────────────────────────────
# 2. MIL WSI-level aggregation
# ──────────────────────────────────────────────

def agg_topk_mean(values, k=None):
    """Mean of the top-k values (for WSI-level probability aggregation)."""
    k = k or config.TOPK_AGG
    v = np.asarray(values, dtype=np.float64)
    if len(v) <= k:
        return float(v.mean())
    topk = np.partition(v, -k)[-k:]
    return float(topk.mean())


# ──────────────────────────────────────────────
# 3. Class weights for imbalanced data
# ──────────────────────────────────────────────

def compute_class_weights(rw_meta, split="train", num_classes=None):
    """Inverse-frequency weights for CrossEntropyLoss."""
    nc = num_classes or config.NUM_CLASSES
    sub = rw_meta[rw_meta["split"] == split]
    counts = sub["label_id"].value_counts().sort_index()

    weights = torch.zeros(nc, dtype=torch.float32)
    total = len(sub)
    for label_id, count in counts.items():
        weights[label_id] = (total / (nc * count)) ** 0.5  # sqrt atténue les extrêmes

    # Classes not present in train get weight 0
    print("Class weights:", weights.tolist())
    return weights


def _filter_and_remap(rw_meta):
    """Remove walks whose label is in EXCLUDED_LABELS and remap label_ids
    to a contiguous range 0 .. N-1 so the model has no unused output neurons.

    Returns:
        rw_meta      : filtered DataFrame with remapped label_id
        inv_label    : {new_id: class_name}
        num_classes  : number of remaining classes
    """
    excluded_ids = {
        config.LABEL_MAP[l]
        for l in config.EXCLUDED_LABELS
        if l in config.LABEL_MAP
    }
    n_before = len(rw_meta)
    rw_meta = rw_meta[~rw_meta["label_id"].isin(excluded_ids)].reset_index(drop=True)
    n_after = len(rw_meta)
    print(f"Filtered excluded labels: {n_before} -> {n_after} walks "
          f"(removed {n_before - n_after} for {config.EXCLUDED_LABELS})")

    # Remap to contiguous ids
    unique_ids = sorted(rw_meta["label_id"].unique())
    remap = {old: new for new, old in enumerate(unique_ids)}
    rw_meta["label_id"] = rw_meta["label_id"].map(remap)

    orig_inv = {v: k for k, v in config.LABEL_MAP.items()}
    inv_label = {new: orig_inv[old] for old, new in remap.items()}
    num_classes = len(unique_ids)

    print(f"Active classes ({num_classes}): {inv_label}")
    return rw_meta, inv_label, num_classes


# ──────────────────────────────────────────────
# 4. Training / evaluation epoch
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, train=True):
    """Runs one training or evaluation epoch.

    Returns:
        avg_loss, all_y, all_pred, all_probs, all_wsi
    """
    model.train() if train else model.eval()

    all_y, all_pred, all_probs, all_wsi = [], [], [], []
    total_loss = 0.0

    for X, lengths, y, wsi_ids in tqdm(loader, leave=False):
        X = X.to(config.DEVICE, non_blocking=True)
        lengths = lengths.to(config.DEVICE, non_blocking=True)
        y = y.to(config.DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(X, lengths)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if config.GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item() * X.size(0)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.tolist())
        all_probs.append(probs)
        all_wsi.extend(wsi_ids)

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    return avg_loss, all_y, all_pred, all_probs, all_wsi


# ──────────────────────────────────────────────
# 5. WSI-level aggregation
# ──────────────────────────────────────────────

def aggregate_wsi(all_y, all_probs, all_wsi, k=None):
    """Aggregates walk-level predictions to WSI-level via top-k mean.

    Returns a DataFrame with columns: wsi_id, y_true, y_pred, + prob per class.
    """
    k = k or config.TOPK_AGG
    df = pd.DataFrame({
        "wsi_id": all_wsi,
        "y_true": all_y,
    })
    # Add per-class probabilities
    for c in range(all_probs.shape[1]):
        df[f"prob_{c}"] = all_probs[:, c]

    prob_cols = [f"prob_{c}" for c in range(all_probs.shape[1])]

    rows = []
    for wsi_id, grp in df.groupby("wsi_id"):
        # WSI true label = most common walk label
        y_true = Counter(grp["y_true"]).most_common(1)[0][0]
        # Aggregate probabilities per class via top-k mean
        agg_probs = [agg_topk_mean(grp[pc].values, k=k) for pc in prob_cols]
        y_pred = int(np.argmax(agg_probs))
        row = {"wsi_id": wsi_id, "y_true": y_true, "y_pred": y_pred}
        for c, p in enumerate(agg_probs):
            row[f"prob_{c}"] = p
        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 6. Main training loop
# ──────────────────────────────────────────────

def run(model_name=None, experiment_name=None):
    """Full training pipeline.

    Args:
        model_name: embedding model name (default config.EMBEDDING_MODEL)
        experiment_name: sub-folder name for results (e.g. "uni_d256_lr2e-4").
                         If None, saves directly in MODELS_DIR.
    """
    from embedding_extractor import load_embeddings
    from walk_generator import load_walks

    model_name = model_name or config.EMBEDDING_MODEL

    # Output directory for this experiment
    if experiment_name:
        exp_dir = os.path.join(config.MODELS_DIR, experiment_name)
    else:
        exp_dir = config.MODELS_DIR
    os.makedirs(exp_dir, exist_ok=True)

    # Load data
    all_paths, rw_meta, index_df = load_walks()
    rw_meta, inv_label, num_classes = _filter_and_remap(rw_meta)
    feats, embed_dim = load_embeddings(model_name)

    # Datasets & loaders
    train_ds = STRSequenceDataset(rw_meta, all_paths, feats, "train")
    val_ds = STRSequenceDataset(rw_meta, all_paths, feats, "val")
    test_ds = STRSequenceDataset(rw_meta, all_paths, feats, "test")

    if getattr(config, "OVERFIT_TEST", False):
        n = config.OVERFIT_N_SAMPLES
        meta = train_ds.meta
        rng = np.random.default_rng(42)
        per_class = max(1, n // num_classes)
        indices = []
        for lid in sorted(meta["label_id"].unique()):
            pool = meta.index[meta["label_id"] == lid].tolist()
            chosen = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
            indices.extend(chosen.tolist())
        remaining = [i for i in range(len(train_ds)) if i not in set(indices)]
        if len(indices) < n and remaining:
            extra = rng.choice(remaining, size=min(n - len(indices), len(remaining)), replace=False)
            indices.extend(extra.tolist())
        indices = indices[:n]

        train_ds = Subset(train_ds, indices)
        val_ds   = train_ds
        test_ds  = train_ds

        dist = meta.iloc[indices]["label"].value_counts().to_dict()
        print(f"\n{'!'*60}")
        print(f"  OVERFIT TEST — {len(indices)} samples stratifies (train=val=test)")
        print(f"  Distribution : {dist}")
        print(f"  Attendu : train_acc -> 1.0 en quelques epochs")
        print(f"  Si non : bug dans data/model/loss")
        print(f"{'!'*60}\n")

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_pad,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_pad,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_pad,
    )

    print(f"Sequences: train={len(train_ds)}, val={len(val_ds)}, "
          f"test={len(test_ds)}")

    # Model
    net = STRTransformer(in_dim=embed_dim, num_classes=num_classes).to(config.DEVICE)
    print(net)

    # Loss (with optional class weights)
    if config.USE_CLASS_WEIGHTS:
        weights = compute_class_weights(rw_meta, num_classes=num_classes).to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Training
    best_path = os.path.join(exp_dir, "best_model.pt")
    best_val_f1m = -1.0   # early stopping sur F1 macro (pas acc)
    bad_epochs = 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*50}  Epoch {epoch}/{config.EPOCHS}  {'='*50}")

        # Train
        tr_loss, tr_y, tr_pred, _, _ = run_epoch(
            net, train_loader, criterion, optimizer, train=True
        )
        tr_acc = accuracy_score(tr_y, tr_pred)
        tr_f1w = f1_score(tr_y, tr_pred, average="weighted", zero_division=0)

        # Validate
        va_loss, va_y, va_pred, va_probs, va_wsi = run_epoch(
            net, val_loader, criterion, train=False
        )
        va_acc = accuracy_score(va_y, va_pred)
        va_f1w = f1_score(va_y, va_pred, average="weighted", zero_division=0)
        va_f1m = f1_score(va_y, va_pred, average="macro", zero_division=0)
        va_bacc = balanced_accuracy_score(va_y, va_pred)

        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss, "tr_acc": tr_acc, "tr_f1w": tr_f1w,
            "va_loss": va_loss, "va_acc": va_acc,
            "va_f1w": va_f1w, "va_f1m": va_f1m, "va_bacc": va_bacc,
        })

        print(
            f"  train  loss={tr_loss:.4f}  acc={tr_acc:.3f}  "
            f"f1w={tr_f1w:.3f}\n"
            f"  val    loss={va_loss:.4f}  acc={va_acc:.3f}  "
            f"f1w={va_f1w:.3f}  f1m={va_f1m:.3f}  bacc={va_bacc:.3f}"
        )

        # Early stopping sur F1 macro (robuste au déséquilibre de classes)
        if va_f1m > best_val_f1m:
            best_val_f1m = va_f1m
            torch.save(net.state_dict(), best_path)
            bad_epochs = 0
            print(f"  -> saved best model (val f1m={best_val_f1m:.3f})")
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print("  Early stopping triggered.")
                break

    # ── Evaluate on test ─────────────────────────
    print(f"\n{'='*50}  TEST  {'='*50}")
    net.load_state_dict(torch.load(best_path, map_location=config.DEVICE, weights_only=True))

    te_loss, te_y, te_pred, _, _ = run_epoch(
        net, test_loader, criterion, train=False
    )
    te_acc = accuracy_score(te_y, te_pred)
    te_f1w = f1_score(te_y, te_pred, average="weighted", zero_division=0)
    te_f1m = f1_score(te_y, te_pred, average="macro", zero_division=0)
    te_bacc = balanced_accuracy_score(te_y, te_pred)
    print(f"Test walk-level  loss={te_loss:.4f}  acc={te_acc:.3f}  "
          f"f1w={te_f1w:.3f}  f1m={te_f1m:.3f}  bacc={te_bacc:.3f}")

    # Walk-level report
    target_names = [inv_label.get(i, str(i)) for i in sorted(set(te_y + te_pred))]
    print("\nWalk-level classification report:")
    print(classification_report(te_y, te_pred, target_names=target_names,
                                zero_division=0))

    # ── Plots ─────────────────────────────────────
    from visualize import plot_test_results, plot_history
    hist_df = pd.DataFrame(history)
    plot_history(exp_dir, history_df=hist_df)
    plot_test_results(te_y, te_pred, exp_dir, inv_label=inv_label)
