"""
Training Module
───────────────
Dataset, training loop, MIL aggregation, and evaluation for the
STRTransformer on random-walk sequences.
"""

import logging
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
# 0. Logger setup
# ──────────────────────────────────────────────

def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    """Creates a file-only logger (no console output).

    Everything (DEBUG and above) is written to <log_dir>/train.log
    (overwritten each run). Rien n'est affiché dans la console pour
    ne pas polluer la sortie du notebook.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()          # avoid duplicate handlers if run() is called twice

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.propagate = False          # n'écrit pas via le root logger (console)

    logger.info(f"Log file: {log_path}")
    return logger


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
        path = np.asarray(self.paths[path_id], dtype=np.int64)

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
# 2. Class weights for imbalanced data
# ──────────────────────────────────────────────

def compute_class_weights(rw_meta, split="train", num_classes=None):
    """Inverse-frequency weights for CrossEntropyLoss."""
    nc = num_classes or config.NUM_CLASSES
    sub = rw_meta[rw_meta["split"] == split]
    counts = sub["label_id"].value_counts().sort_index()

    weights = torch.zeros(nc, dtype=torch.float32)
    total = len(sub)
    mode = getattr(config, "WEIGHT_MODE", "sqrt")
    for label_id, count in counts.items():
        raw = total / (nc * count)
        weights[label_id] = raw if mode == "balanced" else raw ** 0.5

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
# 4. LR Scheduler factory
# ──────────────────────────────────────────────

def _build_scheduler(optimizer, epochs):
    """Builds the LR scheduler from config.SCHEDULER.

    Returns None if config.SCHEDULER is None or "none".
    """
    name = getattr(config, "SCHEDULER", None)
    if not name:
        return None

    eta_min = getattr(config, "SCHEDULER_ETA_MIN", 1e-6)

    if name == "cosine":
        T_max = getattr(config, "SCHEDULER_T_MAX", None) or epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=getattr(config, "SCHEDULER_PLATEAU_PATIENCE", 3),
            factor=getattr(config, "SCHEDULER_PLATEAU_FACTOR", 0.5),
        )
    if name == "cosine_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(config, "SCHEDULER_T_0", 10),
            T_mult=getattr(config, "SCHEDULER_T_MULT", 1),
            eta_min=eta_min,
        )
    raise ValueError(
        f"Unknown SCHEDULER={name!r}. "
        "Choose from 'cosine', 'plateau', 'cosine_restart', or None."
    )


# ──────────────────────────────────────────────
# 5. Training / evaluation epoch
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, train=True,
              logger: logging.Logger = None, epoch: int = 0):
    """Runs one training or evaluation epoch.

    Args:
        logger : optional logger — if provided, traces gradient events,
                 clipping, and a sample of predictions each batch
        epoch  : current epoch number (used in log messages)

    Returns:
        avg_loss, all_y, all_pred, all_probs, all_wsi
    """
    model.train() if train else model.eval()
    phase = "TRAIN" if train else "EVAL "

    all_y, all_pred, all_probs, all_wsi = [], [], [], []
    total_loss = 0.0

    for batch_idx, (X, lengths, y, wsi_ids) in enumerate(tqdm(loader, leave=False)):
        X = X.to(config.DEVICE, non_blocking=True)
        lengths = lengths.to(config.DEVICE, non_blocking=True)
        y = y.to(config.DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(X, lengths)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()

                # ── Gradient diagnostics (null / NaN) ─────────────────
                if logger is not None:
                    null_params, nan_params = [], []
                    for pname, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if param.grad is None:
                            null_params.append(pname)
                        elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            nan_params.append(pname)

                    if null_params:
                        logger.warning(
                            f"E{epoch:02d} {phase} B{batch_idx:04d} | "
                            f"NULL gradients — {len(null_params)} params"
                            f" (ex: {null_params[0]})"
                        )
                    if nan_params:
                        logger.error(
                            f"E{epoch:02d} {phase} B{batch_idx:04d} | "
                            f"NaN/Inf gradients — {len(nan_params)} params"
                            f" (ex: {nan_params[0]})"
                        )

                # ── Gradient clipping ─────────────────────────────────
                if config.GRAD_CLIP > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.GRAD_CLIP
                    )
                    if logger is not None and float(total_norm) > config.GRAD_CLIP:
                        logger.info(
                            f"E{epoch:02d} {phase} B{batch_idx:04d} | "
                            f"CLIP triggered — grad_norm={float(total_norm):.4f}"
                            f" → clipped to {config.GRAD_CLIP}"
                        )

                optimizer.step()

        total_loss += loss.item() * X.size(0)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        # ── Log first-batch prediction sample ─────────────────────────
        if logger is not None and batch_idx == 0:
            y_np = y.cpu().numpy()
            n_show = min(8, len(y_np))
            pairs = "  ".join(
                f"{y_np[i]}→{preds[i]}" for i in range(n_show)
            )
            logger.debug(
                f"E{epoch:02d} {phase} B0000 | "
                f"pred sample (true→pred): {pairs}"
            )

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.tolist())
        all_probs.append(probs)
        all_wsi.extend(wsi_ids)

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    return avg_loss, all_y, all_pred, all_probs, all_wsi


# ──────────────────────────────────────────────
# 5. Main training loop
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

    logger = setup_logger(exp_dir, name=experiment_name or "train")
    logger.info(f"=== Experiment: {experiment_name or 'default'} ===")
    logger.info(
        f"Config — backbone={config.BACKBONE}  d_model={config.D_MODEL}"
        f"  nhead={config.NHEAD}  layers={config.NUM_LAYERS}"
        f"  aggregation={getattr(config,'AGGREGATION','concat')}"
        f"  embedding={model_name}"
    )
    logger.info(
        f"Training — lr={config.LEARNING_RATE}  wd={config.WEIGHT_DECAY}"
        f"  bs={config.BATCH_SIZE}  epochs={config.EPOCHS}"
        f"  patience={config.PATIENCE}  grad_clip={config.GRAD_CLIP}"
        f"  scheduler={getattr(config,'SCHEDULER',None)}"
    )

    # Load data
    all_paths, rw_meta, index_df = load_walks()
    rw_meta, inv_label, num_classes = _filter_and_remap(rw_meta)
    feats, embed_dim = load_embeddings(model_name)

    # ── Sanity check des embeddings (détecte zéros / désalignement tôt) ──
    _sample = np.asarray(
        feats[np.linspace(0, len(feats) - 1, min(len(feats), 5000), dtype=np.int64)],
        dtype=np.float32,
    )
    _zero_frac = float((_sample == 0).all(axis=1).mean())
    _norms = np.linalg.norm(_sample, axis=1)
    logger.info(
        f"Embeddings — dim={embed_dim}  L2_norm(moy={_norms.mean():.3f}"
        f"  min={_norms.min():.3f}  max={_norms.max():.3f})"
        f"  vecteurs_nuls={_zero_frac*100:.2f}%"
    )
    if _zero_frac > 0.001:
        logger.warning(
            f"{_zero_frac*100:.2f}% des embeddings sont nuls — "
            f"marches non discriminantes probables. Re-extraire les embeddings."
        )

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

        dist = meta.iloc[indices]["label_id"].value_counts().to_dict()
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

    logger.info(
        f"Dataset — train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
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
    scheduler = _build_scheduler(optimizer, config.EPOCHS)
    if scheduler is not None:
        print(f"Scheduler: {type(scheduler).__name__}")

    # Training
    best_path = os.path.join(exp_dir, "best_model.pt")
    best_val_f1m = -1.0   # early stopping sur F1 macro (pas acc)
    bad_epochs = 0
    history = []

    # ── Deux phases : geler l'encoder en Phase 1 ──────────────────────────
    freeze_ep = getattr(config, 'FREEZE_ENCODER_EPOCHS', 0)
    if freeze_ep > 0:
        for param in net.encoder.parameters():
            param.requires_grad_(False)
        logger.info(
            f"Phase 1 ({freeze_ep} epochs) : encoder GELÉ — "
            f"entraîne input_norm + proj + cls_head uniquement"
        )

    for epoch in range(1, config.EPOCHS + 1):

        # ── Transition Phase 1 → Phase 2 ──────────────────────────────────
        if freeze_ep > 0 and epoch == freeze_ep + 1:
            for param in net.encoder.parameters():
                param.requires_grad_(True)
            factor = getattr(config, 'PHASE2_LR_FACTOR', 0.1)
            for pg in optimizer.param_groups:
                pg['lr'] *= factor
            logger.info(
                f"E{epoch:02d} Phase 2 : encoder DÉGELÉ — "
                f"LR réduit à {optimizer.param_groups[0]['lr']:.2e}"
            )
        print(f"\n{'='*50}  Epoch {epoch}/{config.EPOCHS}  {'='*50}")

        # Train
        tr_loss, tr_y, tr_pred, _, _ = run_epoch(
            net, train_loader, criterion, optimizer, train=True,
            logger=logger, epoch=epoch,
        )
        tr_acc = accuracy_score(tr_y, tr_pred)
        tr_f1w = f1_score(tr_y, tr_pred, average="weighted", zero_division=0)

        # Validate
        va_loss, va_y, va_pred, va_probs, va_wsi = run_epoch(
            net, val_loader, criterion, train=False,
            logger=logger, epoch=epoch,
        )
        va_acc = accuracy_score(va_y, va_pred)
        va_f1w = f1_score(va_y, va_pred, average="weighted", zero_division=0)
        va_f1m = f1_score(va_y, va_pred, average="macro", zero_division=0)
        va_bacc = balanced_accuracy_score(va_y, va_pred)

        # Step scheduler — log LR change if any
        lr_before = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            logger.info(
                f"E{epoch:02d} LR    | scheduler stepped:"
                f" {lr_before:.2e} → {lr_after:.2e}"
            )

        current_lr = lr_after
        history.append({
            "epoch": epoch,
            "lr": current_lr,
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

        # ── Log epoch summary with prediction distribution ─────────────
        logger.info(
            f"E{epoch:02d} TRAIN | loss={tr_loss:.4f}  acc={tr_acc:.3f}"
            f"  f1w={tr_f1w:.3f}  lr={current_lr:.2e}"
        )
        logger.info(
            f"E{epoch:02d} VAL   | loss={va_loss:.4f}  acc={va_acc:.3f}"
            f"  f1w={va_f1w:.3f}  f1m={va_f1m:.3f}  bacc={va_bacc:.3f}"
        )
        logger.debug(
            f"E{epoch:02d} TRAIN | true_dist={dict(sorted(Counter(tr_y).items()))}"
            f"  pred_dist={dict(sorted(Counter(tr_pred).items()))}"
        )
        logger.debug(
            f"E{epoch:02d} VAL   | true_dist={dict(sorted(Counter(va_y).items()))}"
            f"  pred_dist={dict(sorted(Counter(va_pred).items()))}"
        )

        # Early stopping sur F1 macro (robuste au déséquilibre de classes)
        if va_f1m > best_val_f1m:
            best_val_f1m = va_f1m
            torch.save(net.state_dict(), best_path)
            bad_epochs = 0
            print(f"  -> saved best model (val f1m={best_val_f1m:.3f})")
            logger.info(f"E{epoch:02d} BEST  | saved (val f1m={best_val_f1m:.3f})")
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print("  Early stopping triggered.")
                logger.info(f"E{epoch:02d} STOP  | early stopping after {bad_epochs} bad epochs")
                break

    # ── Evaluate on test ─────────────────────────
    print(f"\n{'='*50}  TEST  {'='*50}")
    logger.info("=" * 60)
    logger.info("TEST phase — best model reloaded")
    net.load_state_dict(torch.load(best_path, map_location=config.DEVICE, weights_only=True))

    te_loss, te_y, te_pred, _, _ = run_epoch(
        net, test_loader, criterion, train=False,
        logger=logger, epoch=0,
    )
    te_acc = accuracy_score(te_y, te_pred)
    te_f1w = f1_score(te_y, te_pred, average="weighted", zero_division=0)
    te_f1m = f1_score(te_y, te_pred, average="macro", zero_division=0)
    te_bacc = balanced_accuracy_score(te_y, te_pred)
    print(f"Test walk-level  loss={te_loss:.4f}  acc={te_acc:.3f}  "
          f"f1w={te_f1w:.3f}  f1m={te_f1m:.3f}  bacc={te_bacc:.3f}")
    logger.info(
        f"TEST | loss={te_loss:.4f}  acc={te_acc:.3f}"
        f"  f1w={te_f1w:.3f}  f1m={te_f1m:.3f}  bacc={te_bacc:.3f}"
    )
    logger.debug(
        f"TEST | true_dist={dict(sorted(Counter(te_y).items()))}"
        f"  pred_dist={dict(sorted(Counter(te_pred).items()))}"
    )

    # Walk-level report
    target_names = [inv_label.get(i, str(i)) for i in sorted(set(te_y + te_pred))]
    print("\nWalk-level classification report:")
    report = classification_report(te_y, te_pred, target_names=target_names,
                                   zero_division=0)
    print(report)
    logger.info("Walk-level classification report:\n" + report)

    # Full predictions log (one line per sample)
    logger.debug("TEST | all predictions — format: idx  true  pred")
    for idx, (yt, yp) in enumerate(zip(te_y, te_pred)):
        true_name = inv_label.get(yt, str(yt))
        pred_name = inv_label.get(yp, str(yp))
        logger.debug(f"  {idx:05d}  {yt} ({true_name:<22})  →  {yp} ({pred_name})")

    # ── Plots ─────────────────────────────────────
    from visualize import plot_test_results, plot_history
    hist_df = pd.DataFrame(history)
    plot_history(exp_dir, history_df=hist_df)
    plot_test_results(te_y, te_pred, exp_dir, inv_label=inv_label)
