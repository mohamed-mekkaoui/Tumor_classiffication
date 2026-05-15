"""
Embedding Extractor Module
───────────────────────────
Reads pre-extracted patches from a memory-mapped array, passes them
through a configurable vision encoder (DINOv2, ViT, UNI, ...), and
saves the resulting embeddings as a memory-mapped numpy array.

Requires: patch_extractor.run() to have been executed first.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
import timm.layers
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import config


# ──────────────────────────────────────────────
# 1. Model factory
# ──────────────────────────────────────────────

def get_model(model_name=None):
    """Returns (model, transform, embed_dim) for the requested encoder.

    Supported names: "dinov2", "vit", "uni".
    Add new models by extending the if/elif chain.
    """

    model_name = model_name or config.EMBEDDING_MODEL
    if model_name not in config.EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(config.EMBEDDING_REGISTRY.keys())}"
        )

    timm_name, embed_dim = config.EMBEDDING_REGISTRY[model_name]

    if model_name == "uni2-h":
        # UNI2-h requires explicit architecture kwargs
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True,
        }
        model = timm.create_model(timm_name, pretrained=True, **timm_kwargs)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == "uni":
        print("on est la")
        # UNI (v1) requires init_values for LayerScale
        model = timm.create_model(timm_name, pretrained=True,
                                  init_values=1e-5, dynamic_img_size=True,num_classes=0)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == "dinov2":
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        transform = None  # built below
    else:
        model = timm.create_model(timm_name, pretrained=True)
        if hasattr(model, "head"):
            model.head = nn.Identity()
        elif hasattr(model, "fc"):
            model.fc = nn.Identity()
        transform = None  # built below

    model = model.to(config.DEVICE).eval()

    # Build generic transform if the model branch didn't set one
    if transform is None:
        cfg = model.default_cfg
        img_size = (cfg["input_size"][1], cfg["input_size"][2])
        transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

    # Determine input size for verification
    cfg = model.default_cfg if hasattr(model, 'default_cfg') else model.pretrained_cfg
    img_size = (cfg["input_size"][1], cfg["input_size"][2])

    # Verify embed_dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size[0], img_size[1]).to(config.DEVICE)
        out = model(dummy)
        actual_dim = out.shape[-1]

    if actual_dim != embed_dim:
        print(f"Warning: expected embed_dim={embed_dim}, got {actual_dim}. "
              f"Using {actual_dim}.")
        embed_dim = actual_dim

    print(f"Loaded model '{model_name}' ({timm_name})")
    print(f"  Input size : {img_size}")
    print(f"  Embed dim  : {embed_dim}")

    return model, transform, embed_dim


# ──────────────────────────────────────────────
# 2. Patch dataset (reads from pre-extracted memmap)
# ──────────────────────────────────────────────

class PatchMemmapDataset(Dataset):
    """Reads patches from a pre-extracted memmap array.

    Each item corresponds to one row in the node index.
    Since patches are stored as uint8 numpy arrays, this is
    fork-safe and supports num_workers > 0.
    """

    def __init__(self, patches_memmap, transform, done_mask=None):
        """
        Args:
            patches_memmap : np.memmap (N, H, W, 3) uint8
            transform      : torchvision transform
            done_mask      : optional np.ndarray to filter already-embedded patches
        """
        self.patches = patches_memmap
        self.transform = transform
        self.N = len(patches_memmap)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Read from memmap (fast, parallel-safe)
        patch_np = self.patches[idx]  # (H, W, 3) uint8
        patch = Image.fromarray(patch_np)

        if self.transform:
            patch = self.transform(patch)
        return patch, idx


# ──────────────────────────────────────────────
# 3. Extraction loop with checkpoint
# ──────────────────────────────────────────────

def extract_embeddings(model_name=None, batch_size=None, num_workers=None):
    """Full extraction pipeline: loads pre-extracted patches, runs model.

    Requires patch_extractor.run() to have been called first.

    Saves:
        features.npy  — memmap float16 (N, embed_dim)
        done.npy      — boolean mask for checkpoint/resume
    """
    model_name = model_name or config.EMBEDDING_MODEL
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.EMBEDDING_NUM_WORKERS

    # Output paths (model-specific sub-folder)
    out_dir = os.path.join(config.EMBEDDINGS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    feats_path = os.path.join(out_dir, "features.npy")
    done_path = os.path.join(out_dir, "done.npy")

    # Load index
    index_path = os.path.join(config.WALKS_DIR, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run walk generation first."
        )
    index_df = pd.read_csv(index_path)
    N = len(index_df)

    # Load model
    model, transform, embed_dim = get_model(model_name)

    # Load pre-extracted patches
    from patch_extractor import load_patches
    patches_memmap, patches_done = load_patches()
    # if not patches_done.all():
    #     undone = int((patches_done == 0).sum())
    #     raise RuntimeError(
    #         f"{undone} patches not yet extracted. "
    #         f"Run patch_extractor.run() first."
    #     )

    # Init or resume memmap + done mask
    if os.path.exists(done_path) and os.path.exists(feats_path):
        done = np.load(done_path)
        feats = np.memmap(feats_path, dtype=np.float16, mode="r+",
                          shape=(N, embed_dim))
        already = int(done.sum())
        print(f"Resuming: {already}/{N} already done")
    else:
        done = np.zeros(N, dtype=np.uint8)
        feats = np.memmap(feats_path, dtype=np.float16, mode="w+",
                          shape=(N, embed_dim))
        np.save(done_path, done)
        print(f"Fresh start: {N} patches to embed")

    if done.all():
        print("All embeddings already extracted.")
        return feats_path, done_path, embed_dim

    # Dataset & loader — num_workers > 0 is now safe (no OpenSlide)
    ds = PatchMemmapDataset(patches_memmap, transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    processed = 0
    with torch.no_grad():
        for X, idxs in tqdm(loader, desc=f"Extracting ({model_name})"):
            idxs = idxs.numpy()
            mask = done[idxs] == 0
            if not mask.any():
                continue

            X_keep = X[mask].to(config.DEVICE, non_blocking=True)
            idx_keep = idxs[mask]

            z = model(X_keep).detach().cpu().numpy().astype(np.float16)
            feats[idx_keep] = z
            done[idx_keep] = 1
            processed += len(idx_keep)

            if processed % 2000 < len(idx_keep):
                np.save(done_path, done)
                feats.flush()

    # Final save
    np.save(done_path, done)
    feats.flush()

    print(f"\nDone: {int(done.sum())}/{N} embeddings extracted")
    print(f"  features : {feats_path}")
    print(f"  done     : {done_path}")

    return feats_path, done_path, embed_dim


def _close_slides(slides):
    for s in slides.values():
        s.close()


# ──────────────────────────────────────────────
# 4. Load embeddings from disk
# ──────────────────────────────────────────────

def load_embeddings(model_name=None):
    """Loads previously extracted embeddings as memmap.

    Returns:
        feats     : np.memmap (N, embed_dim)
        embed_dim : int
    """
    model_name = model_name or config.EMBEDDING_MODEL
    out_dir = os.path.join(config.EMBEDDINGS_DIR, model_name)

    index_df = pd.read_csv(os.path.join(config.WALKS_DIR, "index.csv"))
    N = len(index_df)

    _, embed_dim = config.EMBEDDING_REGISTRY[model_name]
    feats_path = os.path.join(out_dir, "features.npy")
    done_path = os.path.join(out_dir, "done.npy")

    if not os.path.exists(feats_path):
        raise FileNotFoundError(
            f"Embeddings not found at {feats_path}. "
            f"Run embedding extraction first."
        )

    done = np.load(done_path)
    # Verify actual embed_dim from file size
    actual_bytes = os.path.getsize(feats_path)
    actual_dim = actual_bytes // (N * 2)  # float16 = 2 bytes
    if actual_dim != embed_dim:
        print(f"Adjusting embed_dim: registry={embed_dim}, file={actual_dim}")
        embed_dim = actual_dim

    feats = np.memmap(feats_path, dtype=np.float16, mode="r",
                      shape=(N, embed_dim))

    print(f"Loaded embeddings: {feats.shape}, done ratio: {done.mean():.3f}")
    return feats, embed_dim


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run(model_name=None):
    """Run embedding extraction."""
    return extract_embeddings(model_name=model_name)
