"""
Patch Extractor Module
──────────────────────
Reads patch images from WSI slides using OpenSlide and saves them
into a single memory-mapped numpy array for fast downstream access.

This decouples the slow OpenSlide I/O from the GPU-intensive embedding
extraction, enabling parallel data loading and better GPU utilization.

Output:
    patches/patches.npy  — memmap uint8 (N, PATCH_SIZE, PATCH_SIZE, 3)
    patches/done.npy     — boolean checkpoint mask
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import openslide
from PIL import Image
from tqdm import tqdm

import config


# ──────────────────────────────────────────────
# 1. Open slides
# ──────────────────────────────────────────────

def open_slides(index_df):
    """Opens one OpenSlide handle per unique WSI in the index."""
    pairs = config.discover_wsi_pairs()
    svs_map = {p["wsi_id"]: p["svs"] for p in pairs}

    slides = {}
    for wsi_id in index_df["wsi_id"].unique():
        if wsi_id not in svs_map:
            raise FileNotFoundError(
                f"No .svs file found for wsi_id='{wsi_id}'"
            )
        slides[wsi_id] = openslide.OpenSlide(svs_map[wsi_id])

    print(f"Opened {len(slides)} WSI slides")
    return slides


def _close_slides(slides):
    for s in slides.values():
        s.close()


# ──────────────────────────────────────────────
# 2. Extract & save patches to memmap
# ──────────────────────────────────────────────

def _read_one_patch(slide, px, py, read_size, out_size):
    """Read a single patch at level 0 (read_size px) and resize to out_size.

    read_size > out_size when the slide is scanned above TARGET_MPP (e.g. 40×):
    we read a larger window at full resolution and downscale it, so the model
    sees ~20× content (the magnification it was trained on).
    """
    region = slide.read_region((px, py), 0, (read_size, read_size)).convert("RGB")
    if read_size != out_size:
        region = region.resize((out_size, out_size), Image.LANCZOS)
    return np.array(region, dtype=np.uint8)


def extract_patches(patch_size=None, checkpoint_every=10000, num_threads=8):
    """Extracts all patches from WSI slides and saves to a single memmap.

    Supports checkpoint/resume: if interrupted, re-run to continue
    from where it stopped.

    Uses threaded I/O per WSI — OpenSlide is thread-safe so multiple
    read_region calls run in parallel, significantly reducing wall time.

    Args:
        patch_size        : int (default from config.PATCH_SIZE)
        checkpoint_every  : flush memmap to disk every N patches
        num_threads       : number of parallel OpenSlide read threads per WSI

    Returns:
        patches_path : str  path to patches.npy memmap
        done_path    : str  path to done.npy mask
    """
    patch_size = patch_size or config.PATCH_SIZE

    out_dir = config.PATCHES_DIR
    os.makedirs(out_dir, exist_ok=True)
    patches_path = os.path.join(out_dir, "patches.npy")
    done_path = os.path.join(out_dir, "done.npy")
    meta_path = os.path.join(out_dir, "meta.json")

    # Load node index (created by walk_generator)
    index_path = os.path.join(config.WALKS_DIR, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run walk generation first."
        )
    index_df = pd.read_csv(index_path)
    N = len(index_df)

    # ── Anti-staleness : invalider le checkpoint si le graphe a changé ──
    if os.path.exists(done_path) and not config.checkpoint_is_valid(meta_path, index_df):
        print("Checkpoint patches PÉRIMÉ (index.csv différent) → ré-extraction complète.")
        for p in (patches_path, done_path):
            if os.path.exists(p):
                os.remove(p)

    # Pre-extract columns as numpy arrays (avoids slow iloc in loop)
    wsi_ids = index_df["wsi_id"].values
    px_arr = index_df["px"].values.astype(int)
    py_arr = index_df["py"].values.astype(int)
    # read_size = level-0 window read per node (≥ patch_size if slide > TARGET_MPP).
    # Fallback to patch_size for legacy indexes without the column.
    if "read_size" in index_df.columns:
        read_size_arr = index_df["read_size"].values.astype(int)
    else:
        read_size_arr = np.full(N, patch_size, dtype=int)

    print(f"Patches to extract: {N}")
    print(f"  Output size : {patch_size}x{patch_size}")
    print(f"  Read sizes  : {sorted(np.unique(read_size_arr).tolist())} px @ L0 → resize {patch_size}")
    print(f"  Threads     : {num_threads}")
    print(f"  Memmap size : {N * patch_size * patch_size * 3 / 1e9:.1f} GB")

    # Open slides
    slides = open_slides(index_df)

    # Init or resume memmap + done mask
    shape = (N, patch_size, patch_size, 3)
    if os.path.exists(done_path) and os.path.exists(patches_path):
        done = np.load(done_path)
        patches = np.memmap(patches_path, dtype=np.uint8, mode="r+",
                            shape=shape)
        already = int(done.sum())
        print(f"Resuming: {already}/{N} already extracted")
    else:
        done = np.zeros(N, dtype=np.uint8)
        patches = np.memmap(patches_path, dtype=np.uint8, mode="w+",
                            shape=shape)
        np.save(done_path, done)
        config.write_checkpoint_meta(meta_path, index_df, patch_size=patch_size)
        print(f"Fresh start: {N} patches")

    if done.all():
        print("All patches already extracted.")
        _close_slides(slides)
        return patches_path, done_path

    # Group indices by WSI for batch processing
    wsi_groups = {}
    for i in range(N):
        if done[i]:
            continue
        wid = wsi_ids[i]
        if wid not in wsi_groups:
            wsi_groups[wid] = []
        wsi_groups[wid].append(i)

    total_todo = sum(len(v) for v in wsi_groups.values())
    print(f"  Remaining   : {total_todo} patches across {len(wsi_groups)} WSIs")

    # Process WSI by WSI with thread pool
    processed = 0
    pbar = tqdm(total=total_todo, desc="Extracting patches")

    for wsi_id, indices in wsi_groups.items():
        slide = slides[wsi_id]

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {
                pool.submit(
                    _read_one_patch, slide,
                    int(px_arr[i]), int(py_arr[i]),
                    int(read_size_arr[i]), patch_size
                ): i
                for i in indices
            }

            for future in as_completed(futures):
                i = futures[future]
                patches[i] = future.result()
                done[i] = 1
                processed += 1
                pbar.update(1)

                if processed % checkpoint_every == 0:
                    np.save(done_path, done)
                    patches.flush()

        # Checkpoint after each WSI
        np.save(done_path, done)
        patches.flush()

    pbar.close()

    _close_slides(slides)
    print(f"\nDone: {int(done.sum())}/{N} patches extracted")
    print(f"  patches : {patches_path}")
    print(f"  done    : {done_path}")

    return patches_path, done_path


# ──────────────────────────────────────────────
# 3. Load patches memmap (read-only)
# ──────────────────────────────────────────────

def load_patches():
    """Loads the pre-extracted patches memmap in read-only mode.

    Returns:
        patches   : np.memmap (N, PATCH_SIZE, PATCH_SIZE, 3) uint8
        done      : np.ndarray  boolean mask
    """
    out_dir = config.PATCHES_DIR
    patches_path = os.path.join(out_dir, "patches.npy")
    done_path = os.path.join(out_dir, "done.npy")

    if not os.path.exists(patches_path):
        raise FileNotFoundError(
            f"Patches not found at {patches_path}. "
            f"Run patch extraction first."
        )

    index_df = pd.read_csv(os.path.join(config.WALKS_DIR, "index.csv"))
    N = len(index_df)
    ps = config.PATCH_SIZE

    patches = np.memmap(patches_path, dtype=np.uint8, mode="r",
                        shape=(N, ps, ps, 3))
    done = np.load(done_path)

    print(f"Loaded patches: {patches.shape}, done ratio: {done.mean():.3f}")
    return patches, done


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run():
    """Run patch extraction."""
    return extract_patches()