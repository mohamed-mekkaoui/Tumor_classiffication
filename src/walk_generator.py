"""
Walk Generator Module
─────────────────────
Builds hex graphs for all WSIs, generates random walks, creates a global
node index, assigns train/val/test splits, and exports everything to disk.
"""

import os
import random
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from wsi_graph import WSIHexGraph
import config


# ──────────────────────────────────────────────
# 1. Build graphs & tag annotations
# ──────────────────────────────────────────────

def build_all_graphs(pairs):
    """Builds and annotates a WSIHexGraph for each WSI.

    Returns:
        dict  {wsi_id: WSIHexGraph}
    """
    graphs = {}
    for p in pairs:
        wsi_id = p["wsi_id"]
        print(f"\n{'='*60}")
        print(f"Building graph for {wsi_id}")
        print(f"{'='*60}")

        g = WSIHexGraph(
            p["svs"],
            patch_size=config.PATCH_SIZE,
            white_threshold=config.WHITE_THRESHOLD,
            white_ratio=config.WHITE_RATIO,
        )
        g.build_graph()
        g.load_annotations(p["geojson"])
        g.tag_nodes_with_annotations()

        graphs[wsi_id] = g
        print(f"{wsi_id}: {g.graph.number_of_nodes()} nodes, "
              f"{g.graph.number_of_edges()} edges")

    return graphs


# ──────────────────────────────────────────────
# 1b. Filter out "no_Tissu" nodes
# ──────────────────────────────────────────────

def filter_no_tissu(graphs, label="no_Tissu"):
    """Supprime tous les nœuds annotés ``label`` (par défaut 'no_Tissu')
    de chaque graphe, ainsi que leurs arêtes.

    Met également à jour ``grid_nodes`` pour rester cohérent.

    Args:
        graphs : dict {wsi_id: WSIHexGraph}
        label  : str  annotation à exclure

    Returns:
        graphs (modifié en place)
    """
    for wsi_id, g in graphs.items():
        to_remove = [
            nid for nid, data in g.graph.nodes(data=True)
            if data.get("label") == label
        ]
        if not to_remove:
            continue

        before = g.graph.number_of_nodes()
        g.graph.remove_nodes_from(to_remove)

        # Mettre à jour grid_nodes (inverse lookup (c,r) → nid)
        removed_set = set(to_remove)
        g.grid_nodes = {
            k: v for k, v in g.grid_nodes.items()
            if v not in removed_set
        }

        after = g.graph.number_of_nodes()
        print(f"{wsi_id}: supprimé {before - after} nœuds '{label}' "
              f"({before} → {after})")

    return graphs


# ──────────────────────────────────────────────
# 2. Global node index
# ──────────────────────────────────────────────

def create_node_index(graphs):
    """Creates a global index over all nodes of all WSIs.

    Returns:
        index_df : pd.DataFrame  (global_idx, wsi_id, node_id, c, r, px, py, cx, cy, label, label_id)
        node_to_global : dict    {(wsi_id, node_id) -> global_idx}
    """
    rows = []
    node_to_global = {}
    global_idx = 0

    for wsi_id, g in graphs.items():
        # Taille de lecture au niveau 0 pour cette lame (= tuile 20×, constante par WSI)
        read_size = getattr(g, "tile_l0", config.PATCH_SIZE)
        for nid, data in g.graph.nodes(data=True):
            label_str = data.get("label", "background")
            label_id = config.LABEL_MAP.get(label_str, 0)
            rows.append({
                "global_idx": global_idx,
                "wsi_id": wsi_id,
                "node_id": nid,
                "c": data["c"],
                "r": data["r"],
                "px": data["px"],
                "py": data["py"],
                "cx": data["cx"],
                "cy": data["cy"],
                "read_size": read_size,
                "label": label_str,
                "label_id": label_id,
            })
            node_to_global[(wsi_id, nid)] = global_idx
            global_idx += 1

    index_df = pd.DataFrame(rows)
    print(f"\nGlobal node index: {len(index_df)} nodes across "
          f"{index_df['wsi_id'].nunique()} WSIs")
    print("Label distribution:")
    print(index_df["label"].value_counts().to_string())
    return index_df, node_to_global


# ──────────────────────────────────────────────
# 3. Split WSIs into train / val / test
# ──────────────────────────────────────────────

def assign_splits(wsi_ids, seed=None):
    """Assigns each WSI to train / val / test.
    With 6 WSIs → 4 train, 1 val, 1 test (shuffled).

    Returns:
        dict {wsi_id: "train" | "val" | "test"}
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    ids = list(wsi_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, n // 6)
    n_val = max(1, n // 6)

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]

    splits = {}
    for w in train_ids:
        splits[w] = "train"
    for w in val_ids:
        splits[w] = "val"
    for w in test_ids:
        splits[w] = "test"

    print(f"\nSplit: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    for w, s in sorted(splits.items(), key=lambda x: x[1]):
        print(f"  {s:5s} : {w}")

    return splits


def assign_stratified_walk_splits(rw_meta, seed=None,
                                  train_ratio=None, val_ratio=None):
    """Stratified walk-level split: mixes ALL walks then splits by label_id.

    Ensures every class appears in train, val, and test sets.
    Ignores the WSI-level split column and overwrites it.

    Args:
        rw_meta    : pd.DataFrame with at least 'label_id' column
        seed       : random seed
        train_ratio: fraction for train (default config.TRAIN_RATIO)
        val_ratio  : fraction for val   (default config.VAL_RATIO)

    Returns:
        rw_meta with 'split' column updated in-place
    """
    from sklearn.model_selection import train_test_split

    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO
    seed = seed if seed is not None else config.SPLIT_SEED

    # First split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    idx_trainval, idx_test = train_test_split(
        rw_meta.index, test_size=test_ratio,
        stratify=rw_meta["label_id"], random_state=seed,
    )

    # Second split: train vs val (from trainval)
    val_fraction_of_trainval = val_ratio / (train_ratio + val_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_fraction_of_trainval,
        stratify=rw_meta.loc[idx_trainval, "label_id"],
        random_state=seed,
    )

    rw_meta.loc[idx_train, "split"] = "train"
    rw_meta.loc[idx_val, "split"] = "val"
    rw_meta.loc[idx_test, "split"] = "test"

    print(f"\nStratified walk-level split (seed={seed}):")
    print(f"  train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    # Show classes per split
    for s in ["train", "val", "test"]:
        classes = sorted(rw_meta[rw_meta["split"] == s]["label_id"].unique())
        print(f"  {s:5s} classes: {classes}")

    return rw_meta


# ──────────────────────────────────────────────
# 4. Walk-level label via majority vote
# ──────────────────────────────────────────────

def walk_label(walk_global_indices, index_df):
    """Returns the most common label_id among the walk's nodes."""
    labels = index_df.loc[walk_global_indices, "label_id"].tolist()
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


# ──────────────────────────────────────────────
# 5. Generate all walks
# ──────────────────────────────────────────────

def generate_all_walks(graphs, node_to_global, index_df, splits,
                       walks_per_wsi=None, min_length=None, max_length=None,
                       sharp_turn_weight=None):
    """Generates random walks for every WSI and converts them to global indices.

    Returns:
        all_paths : list of lists  (each list = sequence of global indices)
        rw_meta   : pd.DataFrame   (path_id, wsi_id, split, path_len, label_id)
    """
    walks_per_wsi = walks_per_wsi or config.WALKS_PER_WSI
    min_length = min_length or config.WALK_MIN_LENGTH
    max_length = max_length or config.WALK_MAX_LENGTH
    sharp_turn_weight = sharp_turn_weight if sharp_turn_weight is not None else config.SHARP_TURN_WEIGHT

    all_paths = []
    meta_rows = []
    seen = set()

    for wsi_id, g in graphs.items():
        split = splits[wsi_id]
        nodes = list(g.graph.nodes())
        if len(nodes) < min_length:
            print(f"Skipping {wsi_id}: only {len(nodes)} nodes (< min_length)")
            continue

        accepted = 0
        attempts = 0
        max_attempts = walks_per_wsi * 10

        while accepted < walks_per_wsi and attempts < max_attempts:
            attempts += 1
            start = random.choice(nodes)
            walk = g.generate_random_walk(
                start,
                min_length=min_length,
                max_length=max_length,
                sharp_turn_weight=sharp_turn_weight,
            )
            if len(walk) < min_length:
                continue

            # Convert to global indices
            global_walk = [node_to_global[(wsi_id, nid)] for nid in walk]

            # Deduplicate: reject identical walks and reversed duplicates
            canonical = min(tuple(global_walk), tuple(reversed(global_walk)))
            if canonical in seen:
                continue
            seen.add(canonical)

            label_id = walk_label(global_walk, index_df)

            path_id = len(all_paths)
            all_paths.append(global_walk)
            meta_rows.append({
                "path_id": path_id,
                "wsi_id": wsi_id,
                "split": split,
                "path_len": len(global_walk),
                "label_id": label_id,
            })
            accepted += 1

        print(f"{wsi_id} ({split}): {accepted} walks generated "
              f"({attempts} attempts)")

    rw_meta = pd.DataFrame(meta_rows)

    print(f"\nTotal walks: {len(all_paths)}")
    print(rw_meta.groupby("split")["path_id"].count().to_string())
    print(f"\nLabel distribution across walks:")
    print(rw_meta["label_id"].value_counts().sort_index().to_string())

    return all_paths, rw_meta


# ──────────────────────────────────────────────
# 5b. Generate walks constrained by region
# ──────────────────────────────────────────────

def generate_all_walks_by_region(graphs, node_to_global, index_df, splits,
                                 walks_per_region=None, min_length=None,
                                 max_length=None, sharp_turn_weight=None,
                                 excluded_labels=None, min_region_size=None,
                                 bounce=None):
    """Generates random walks constrained within each connected component
    of each annotation class.

    For each WSI:
      - Groups nodes by annotation label
      - Excludes labels in ``excluded_labels``
      - Finds connected components per label (via NetworkX subgraph)
      - Generates ``walks_per_region`` walks per component large enough

    All patches in a walk share the same class label.

    Args:
        bounce: If True, uses bounce logic (sharp-turn + backtrack at
                boundaries).  If False, walk stops at boundaries.
                Defaults to ``config.WALK_BOUNCE``.

    Returns:
        all_paths : list of lists  (each list = sequence of global indices)
        rw_meta   : pd.DataFrame   (path_id, wsi_id, split, path_len, label_id, label)
    """
    walks_per_region = walks_per_region or config.WALKS_PER_REGION
    min_length = min_length or config.WALK_MIN_LENGTH
    max_length = max_length or config.WALK_MAX_LENGTH
    sharp_turn_weight = (sharp_turn_weight if sharp_turn_weight is not None
                         else config.SHARP_TURN_WEIGHT)
    excluded_labels = excluded_labels or config.EXCLUDED_LABELS
    min_region_size = min_region_size or config.MIN_REGION_SIZE
    if bounce is None:
        bounce = config.WALK_BOUNCE

    walk_method_name = "bounce" if bounce else "stop"
    print(f"\nWalk mode: {walk_method_name} at region boundaries")

    all_paths = []
    meta_rows = []
    seen = set()

    for wsi_id, g in graphs.items():
        split = splits[wsi_id]

        # Group nodes by label
        nodes_by_label = defaultdict(list)
        for nid, data in g.graph.nodes(data=True):
            label = data.get("label", "background")
            nodes_by_label[label].append(nid)

        for label, label_nodes in nodes_by_label.items():
            # Skip excluded labels
            if label in excluded_labels:
                continue

            label_id = config.LABEL_MAP.get(label, 0)

            # Find connected components within this label's subgraph
            subgraph = g.graph.subgraph(label_nodes)
            components = list(nx.connected_components(subgraph))

            for comp_idx, comp_nodes in enumerate(components):
                comp_nodes = list(comp_nodes)
                if len(comp_nodes) < min_region_size:
                    continue

                accepted = 0
                attempts = 0
                max_attempts = walks_per_region * 10

                while accepted < walks_per_region and attempts < max_attempts:
                    attempts += 1
                    start = random.choice(comp_nodes)

                    if bounce:
                        walk = g.generate_random_walk_bounce(
                            start,
                            min_length=min_length,
                            max_length=max_length,
                            constraint_label=label,
                            sharp_turn_weight=sharp_turn_weight,
                        )
                    else:
                        walk = g.generate_random_walk(
                            start,
                            min_length=min_length,
                            max_length=max_length,
                            constraint_label=label,
                            sharp_turn_weight=sharp_turn_weight,
                        )
                    if len(walk) < min_length:
                        continue

                    global_walk = [node_to_global[(wsi_id, nid)]
                                   for nid in walk]

                    # Deduplicate: reject identical walks and reversed duplicates
                    canonical = min(tuple(global_walk), tuple(reversed(global_walk)))
                    if canonical in seen:
                        continue
                    seen.add(canonical)

                    path_id = len(all_paths)
                    all_paths.append(global_walk)
                    meta_rows.append({
                        "path_id": path_id,
                        "wsi_id": wsi_id,
                        "split": split,
                        "path_len": len(global_walk),
                        "label_id": label_id,
                        "label": label,
                    })
                    accepted += 1

                print(f"  {wsi_id}/{label} comp#{comp_idx} "
                      f"({len(comp_nodes)} nodes): "
                      f"{accepted} walks ({attempts} attempts)")

    rw_meta = pd.DataFrame(meta_rows)

    print(f"\nTotal walks: {len(all_paths)}")
    if len(rw_meta) > 0:
        print(rw_meta.groupby("split")["path_id"].count().to_string())
        print(f"\nLabel distribution across walks:")
        label_counts = rw_meta.groupby(["label_id", "label"]).size()
        print(label_counts.to_string())

    return all_paths, rw_meta


# ──────────────────────────────────────────────
# 6. Save to disk
# ──────────────────────────────────────────────

def save_walks(all_paths, rw_meta, index_df, output_dir=None):
    """Saves walk data to disk."""
    output_dir = output_dir or config.WALKS_DIR
    os.makedirs(output_dir, exist_ok=True)

    paths_file = os.path.join(output_dir, "rw_paths.npy")
    meta_file = os.path.join(output_dir, "rw_meta.csv")
    index_file = os.path.join(output_dir, "index.csv")

    np.save(paths_file, np.array(all_paths, dtype=object))
    rw_meta.to_csv(meta_file, index=False)
    index_df.to_csv(index_file, index=False)

    print(f"\nSaved:")
    print(f"  {paths_file}  ({len(all_paths)} paths)")
    print(f"  {meta_file}   ({len(rw_meta)} rows)")
    print(f"  {index_file}  ({len(index_df)} nodes)")


# ──────────────────────────────────────────────
# 7. Load from disk
# ──────────────────────────────────────────────

def load_walks(output_dir=None):
    """Loads previously saved walk data."""
    output_dir = output_dir or config.WALKS_DIR

    all_paths = np.load(
        os.path.join(output_dir, "rw_paths.npy"), allow_pickle=True
    )
    rw_meta = pd.read_csv(os.path.join(output_dir, "rw_meta.csv"))
    index_df = pd.read_csv(os.path.join(output_dir, "index.csv"))

    print(f"Loaded: {len(all_paths)} paths, {len(rw_meta)} meta rows, "
          f"{len(index_df)} nodes")
    return all_paths, rw_meta, index_df


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run(walks_per_region=None):
    """Full walk generation pipeline (region-constrained walks)."""
    pairs = config.discover_wsi_pairs()
    if not pairs:
        raise FileNotFoundError(f"No WSI pairs found in {config.DATA_DIR}")

    graphs = build_all_graphs(pairs)
    graphs = filter_no_tissu(graphs)
    index_df, node_to_global = create_node_index(graphs)
    splits = assign_splits(list(graphs.keys()), seed=config.SPLIT_SEED)
    all_paths, rw_meta = generate_all_walks_by_region(
        graphs, node_to_global, index_df, splits,
        walks_per_region=walks_per_region,
    )

    # Apply stratified walk-level split if enabled
    if getattr(config, "STRATIFIED_SPLIT", False):
        rw_meta = assign_stratified_walk_splits(rw_meta, seed=config.SPLIT_SEED)

    save_walks(all_paths, rw_meta, index_df)
    return all_paths, rw_meta, index_df
