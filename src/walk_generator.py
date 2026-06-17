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


def assign_region_holdout_split(rw_meta, regions_per_class=1, seed=42,
                                train_ratio=None, val_ratio=None):
    """Hold out N whole regions per class as test set; stratified train/val on the rest.

    regions_per_class : int (global) OR dict {"default": 1, "ACINAIRE": 5, ...}
        Classes absent from the dict use the "default" value.
        Selection criterion: the N largest regions (most walks) per class.

    Requires rw_meta to have 'label' (str) and 'region_key' columns, which are
    produced by generate_all_walks_by_region() and generate_all_walks_balanced().
    """
    from sklearn.model_selection import train_test_split as _tts

    train_ratio = train_ratio if train_ratio is not None else config.TRAIN_RATIO
    val_ratio   = val_ratio   if val_ratio   is not None else config.VAL_RATIO

    if "region_key" not in rw_meta.columns:
        raise ValueError(
            "rw_meta has no 'region_key' column. "
            "Re-generate walks with the updated walk_generator."
        )

    def _n(label_name):
        if isinstance(regions_per_class, int):
            return regions_per_class
        return regions_per_class.get(label_name,
               regions_per_class.get("default", 1))

    rw_meta = rw_meta.copy()
    rw_meta["split"] = "train"

    # 1. Select test regions: N largest per class
    test_keys = set()
    print("\nRegion-holdout — test region selection:")
    for label_name in sorted(rw_meta["label"].unique()):
        sub = rw_meta[rw_meta["label"] == label_name]
        sizes = sub.groupby("region_key").size().sort_values(ascending=False)
        n = _n(label_name)
        selected = sizes.head(n).index.tolist()
        test_keys.update(selected)
        print(f"  {label_name:20s}: {n} region(s), "
              f"{sizes[selected].sum()} test walks  {selected}")

    # 2. Mark test walks
    test_mask = rw_meta["region_key"].isin(test_keys)
    rw_meta.loc[test_mask, "split"] = "test"

    # 3. Stratified train/val on the remaining walks
    remaining = rw_meta[~test_mask]
    adj_val = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = _tts(
        remaining.index,
        test_size=adj_val,
        stratify=remaining["label_id"],
        random_state=seed,
    )
    rw_meta.loc[train_idx, "split"] = "train"
    rw_meta.loc[val_idx,   "split"] = "val"

    print(f"\nSplit summary (region holdout, seed={seed}):")
    for s in ["train", "val", "test"]:
        sub = rw_meta[rw_meta["split"] == s]
        classes = sorted(sub["label_id"].unique())
        print(f"  {s:5s}: {len(sub):6d} walks  classes={classes}")

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
                        "component_id": comp_idx,
                        "region_key": f"{wsi_id}__{label}__{comp_idx}",
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
# 5c. Generate class-balanced walks (size-aware, redundancy-capped)
# ──────────────────────────────────────────────

def _walks_for_component(g, wsi_id, comp_nodes, label, node_to_global, seen,
                         n_target, min_length, max_length, sharp_turn_weight, bounce):
    """Generate up to ``n_target`` DISTINCT global-index walks inside one component.

    Uses the shared ``seen`` set for global dedup (canonical, reverse-invariant).
    Returns a list of global walks (each a list of global indices).
    """
    walks = []
    attempts = 0
    max_attempts = max(n_target * 10, 10)
    walk_fn = g.generate_random_walk_bounce if bounce else g.generate_random_walk

    while len(walks) < n_target and attempts < max_attempts:
        attempts += 1
        start = random.choice(comp_nodes)
        walk = walk_fn(
            start,
            min_length=min_length,
            max_length=max_length,
            constraint_label=label,
            sharp_turn_weight=sharp_turn_weight,
        )
        if len(walk) < min_length:
            continue

        global_walk = [node_to_global[(wsi_id, nid)] for nid in walk]
        canonical = min(tuple(global_walk), tuple(reversed(global_walk)))
        if canonical in seen:
            continue
        seen.add(canonical)
        walks.append(global_walk)

    return walks


def generate_all_walks_balanced(graphs, node_to_global, index_df, splits,
                                walks_per_class=None, max_redundancy=None,
                                min_length=None, max_length=None,
                                sharp_turn_weight=None, excluded_labels=None,
                                min_region_size=None, bounce=None):
    """Generate walks with a per-CLASS budget instead of a per-region count.

    For each non-excluded class:
      1. gather all its connected components (>= ``min_region_size``) across all WSIs ;
      2. cap the budget by available tissue (size-aware) so redundancy stays bounded::

             total_nodes    = Σ component sizes
             class_capacity = max_redundancy * total_nodes / L   (L = avg walk length)
             target_eff     = min(walks_per_class, class_capacity)

      3. distribute ``target_eff`` across components proportionally to their size ;
      4. report per class: #regions, nodes, target, produced, avg redundancy.

    Classes poor in tissue plateau below the target (flagged) — honest by design.
    Returns ``(all_paths, rw_meta)`` in the same format as ``generate_all_walks_by_region``.
    """
    walks_per_class   = walks_per_class   or config.WALKS_PER_CLASS
    max_redundancy    = max_redundancy    if max_redundancy is not None else config.MAX_WALK_REDUNDANCY
    min_length        = min_length        or config.WALK_MIN_LENGTH
    max_length        = max_length        or config.WALK_MAX_LENGTH
    sharp_turn_weight = (sharp_turn_weight if sharp_turn_weight is not None
                         else config.SHARP_TURN_WEIGHT)
    excluded_labels   = excluded_labels   or config.EXCLUDED_LABELS
    min_region_size   = min_region_size   or config.MIN_REGION_SIZE
    if bounce is None:
        bounce = config.WALK_BOUNCE

    L = (min_length + max_length) / 2.0  # average walk length

    # 1. Collect components per class across all WSIs
    class_components = defaultdict(list)  # label -> [(wsi_id, g, comp_nodes), ...]
    for wsi_id, g in graphs.items():
        nodes_by_label = defaultdict(list)
        for nid, data in g.graph.nodes(data=True):
            nodes_by_label[data.get("label", "background")].append(nid)

        for label, label_nodes in nodes_by_label.items():
            if label in excluded_labels:
                continue
            subgraph = g.graph.subgraph(label_nodes)
            for comp in nx.connected_components(subgraph):
                comp = list(comp)
                if len(comp) >= min_region_size:
                    class_components[label].append((wsi_id, g, comp))

    walk_method_name = "bounce" if bounce else "stop"
    print(f"\nBalanced walk generation — target={walks_per_class}/class, "
          f"max_redundancy={max_redundancy}x, mode={walk_method_name}")

    all_paths = []
    meta_rows = []
    seen = set()
    report = []

    for label in sorted(class_components.keys()):
        comps = class_components[label]
        label_id = config.LABEL_MAP.get(label, 0)
        total_nodes = sum(len(c) for _, _, c in comps)

        # Size-aware, redundancy-capped class budget
        class_capacity = int(max_redundancy * total_nodes / L)
        target_eff = min(walks_per_class, class_capacity)

        # Distribute proportionally to component size (largest first; shortfalls roll over)
        comps_sorted = sorted(comps, key=lambda t: len(t[2]), reverse=True)
        produced = 0
        for i, (wsi_id, g, comp) in enumerate(comps_sorted):
            remaining_target = target_eff - produced
            remaining_nodes = sum(len(c) for _, _, c in comps_sorted[i:])
            if remaining_target <= 0 or remaining_nodes <= 0:
                break
            alloc = max(1, int(round(remaining_target * len(comp) / remaining_nodes)))

            walks = _walks_for_component(
                g, wsi_id, comp, label, node_to_global, seen,
                alloc, min_length, max_length, sharp_turn_weight, bounce,
            )
            for gw in walks:
                path_id = len(all_paths)
                all_paths.append(gw)
                meta_rows.append({
                    "path_id": path_id,
                    "wsi_id": wsi_id,
                    "split": splits[wsi_id],
                    "path_len": len(gw),
                    "label_id": label_id,
                    "label": label,
                    "component_id": i,
                    "region_key": f"{wsi_id}__{label}__{i}",
                })
            produced += len(walks)

        redund = (produced * L / total_nodes) if total_nodes else 0.0
        capped = produced < walks_per_class
        report.append((label, len(comps), total_nodes, walks_per_class, produced, redund, capped))

    rw_meta = pd.DataFrame(meta_rows)

    # ── Reporting (jury artifact) ──
    print(f"\n{'classe':20s} {'régions':>8s} {'nœuds':>8s} {'cible':>7s} "
          f"{'obtenu':>7s} {'redond':>8s}")
    print("-" * 64)
    for label, nreg, nnodes, target, prod, redund, capped in report:
        flag = "  ← plafonné (tissu)" if capped else ""
        print(f"{label:20s} {nreg:8d} {nnodes:8d} {target:7d} "
              f"{prod:7d} {redund:7.1f}x{flag}")

    print(f"\nTotal walks: {len(all_paths)}")
    if len(rw_meta) > 0:
        print(rw_meta.groupby("split")["path_id"].count().to_string())

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

    if getattr(config, "BALANCE_WALKS", False):
        all_paths, rw_meta = generate_all_walks_balanced(
            graphs, node_to_global, index_df, splits,
        )
    else:
        all_paths, rw_meta = generate_all_walks_by_region(
            graphs, node_to_global, index_df, splits,
            walks_per_region=walks_per_region,
        )

    # Apply split strategy (region holdout > stratified walk-level > WSI-level)
    if getattr(config, "REGION_HOLDOUT_SPLIT", False):
        rw_meta = assign_region_holdout_split(
            rw_meta,
            regions_per_class=getattr(config, "TEST_REGIONS_PER_CLASS", 1),
            seed=getattr(config, "TEST_REGION_SEED", config.SPLIT_SEED),
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
        )
    elif getattr(config, "STRATIFIED_SPLIT", False):
        rw_meta = assign_stratified_walk_splits(rw_meta, seed=config.SPLIT_SEED)

    save_walks(all_paths, rw_meta, index_df)
    return all_paths, rw_meta, index_df
