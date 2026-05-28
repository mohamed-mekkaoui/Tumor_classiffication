"""
Diagnostic Module (jetable)
───────────────────────────
Localise la cause du collapse "prédit une seule classe" en bisectant :
le problème est-il dans les EMBEDDINGS ou dans le MODÈLE de séquence ?

Étape A — complétude & alignement des embeddings :
    Vérifie que features.npy correspond bien au index.csv courant
    (taille fichier, done.npy complet, lignes nulles, normes).

Étape B — test de séparabilité (décisif) :
    Entraîne une régression logistique directement sur les embeddings
    de nœuds → label. Si elle échoue, les embeddings sont le problème
    (zéros / désalignement). Si elle réussit, le souci est dans le
    transformer / le chemin séquence.

Usage:
    python diagnose.py                 # modèle = config.EMBEDDING_MODEL
    python diagnose.py uni             # modèle explicite
"""

import os
import sys

import numpy as np
import pandas as pd

import config


def _features_path(model_name):
    return os.path.join(config.EMBEDDINGS_DIR, model_name, "features.npy")


def _done_path(model_name):
    return os.path.join(config.EMBEDDINGS_DIR, model_name, "done.npy")


# ──────────────────────────────────────────────
# Étape A — complétude & alignement
# ──────────────────────────────────────────────

def step_a(model_name):
    print("=" * 70)
    print(f"ÉTAPE A — Complétude & alignement des embeddings ('{model_name}')")
    print("=" * 70)

    index_path = os.path.join(config.WALKS_DIR, "index.csv")
    if not os.path.exists(index_path):
        print(f"  [ERREUR] index.csv introuvable: {index_path}")
        return None, None

    index_df = pd.read_csv(index_path)
    N = len(index_df)
    print(f"  index.csv courant       : N = {N} nœuds")

    feats_path = _features_path(model_name)
    if not os.path.exists(feats_path):
        print(f"  [ERREUR] features.npy introuvable: {feats_path}")
        return index_df, None

    # Dimension attendue depuis le registre
    _, reg_dim = config.EMBEDDING_REGISTRY[model_name]
    nbytes = os.path.getsize(feats_path)

    # taille = N * dim * 2 (float16)  →  dim déduite
    if nbytes % (N * 2) == 0:
        file_dim = nbytes // (N * 2)
    else:
        file_dim = nbytes / (N * 2)  # non entier → désalignement franc

    print(f"  features.npy            : {nbytes} octets")
    print(f"  embed_dim (registre)    : {reg_dim}")
    print(f"  embed_dim (déduit fichier): {file_dim}")

    aligned = (nbytes == N * reg_dim * 2)
    if aligned:
        print("  [OK] taille fichier == N * embed_dim * 2  →  ALIGNÉ")
    else:
        print("  [!!! ALERTE] taille fichier != N * embed_dim * 2")
        print("       → features.npy ne correspond PAS au index.csv courant.")
        print("       → Cause probable : graphe modifié sans re-extraction des embeddings.")
        print("       → CORRECTIF : supprimer output/embeddings/ et output/patches/, puis re-extraire.")

    # done.npy
    done_path = _done_path(model_name)
    if os.path.exists(done_path):
        done = np.load(done_path)
        ratio = float(done.mean())
        print(f"  done.npy                : taille={len(done)}, ratio complété={ratio:.4f}")
        if len(done) != N:
            print(f"  [!!! ALERTE] done.npy a {len(done)} entrées mais index.csv en a {N}")
        if ratio < 1.0:
            print(f"  [!!! ALERTE] extraction INCOMPLÈTE : {(done == 0).sum()} embeddings manquants (lignes à zéro)")
    else:
        print(f"  [AVERTISSEMENT] done.npy introuvable: {done_path}")

    # Lecture des embeddings (seulement si la taille est cohérente avec une dim entière)
    if not float(file_dim).is_integer():
        print("  [STOP] dim non entière → impossible de mapper proprement, arrêt étape A.")
        return index_df, None

    file_dim = int(file_dim)
    feats = np.memmap(feats_path, dtype=np.float16, mode="r", shape=(N, file_dim))

    # Lignes entièrement nulles
    sample = min(N, 20000)
    idx_sample = np.linspace(0, N - 1, sample, dtype=np.int64)
    block = np.asarray(feats[idx_sample], dtype=np.float32)
    zero_rows = int((block == 0).all(axis=1).sum())
    print(f"  Échantillon {sample} lignes : {zero_rows} entièrement nulles "
          f"({100*zero_rows/sample:.2f}%)")
    if zero_rows > 0:
        print("  [!!! ALERTE] embeddings nuls présents → marches non discriminantes.")

    norms = np.linalg.norm(block, axis=1)
    print(f"  Norme L2 (échantillon)  : moy={norms.mean():.3f}  "
          f"min={norms.min():.3f}  max={norms.max():.3f}  std={norms.std():.3f}")

    return index_df, feats


# ──────────────────────────────────────────────
# Étape B — test de séparabilité (régression logistique)
# ──────────────────────────────────────────────

def step_b(index_df, feats, max_per_class=2000):
    print()
    print("=" * 70)
    print("ÉTAPE B — Séparabilité des embeddings (régression logistique)")
    print("=" * 70)

    if feats is None or index_df is None:
        print("  [STOP] embeddings non chargeables (voir étape A).")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
    )
    from sklearn.preprocessing import StandardScaler

    # Retirer les classes exclues
    excluded_ids = {
        config.LABEL_MAP[l] for l in config.EXCLUDED_LABELS
        if l in config.LABEL_MAP
    }
    df = index_df[~index_df["label_id"].isin(excluded_ids)].copy()
    print(f"  Nœuds après exclusion   : {len(df)} (classes exclues: {config.EXCLUDED_LABELS})")

    # Sous-échantillon stratifié borné par classe
    parts = []
    for lid, grp in df.groupby("label_id"):
        n = min(len(grp), max_per_class)
        parts.append(grp.sample(n=n, random_state=42))
    df = pd.concat(parts).reset_index(drop=True)

    counts = df["label_id"].value_counts().sort_index()
    print(f"  Distribution échantillon : {counts.to_dict()}")

    if df["label_id"].nunique() < 2:
        print("  [STOP] moins de 2 classes disponibles, test impossible.")
        return

    X = np.asarray(feats[df["global_idx"].values], dtype=np.float32)
    y = df["label_id"].values

    # Drop lignes nulles (ne pas fausser le test)
    keep = ~(X == 0).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  [info] {dropped} lignes nulles retirées du test")
    X, y = X[keep], y[keep]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # class_weight='balanced' : neutralise le déséquilibre du probe pour
    # mesurer la séparabilité RÉELLE des embeddings (sinon le probe collapse
    # de lui-même vers les classes majoritaires et donne une lecture faussée).
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1,
                             class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    bacc = balanced_accuracy_score(yte, pred)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    inv = {v: k for k, v in config.LABEL_MAP.items()}
    names = [inv.get(c, str(c)) for c in sorted(np.unique(y))]
    print()
    print(classification_report(yte, pred, target_names=names, zero_division=0))
    print(f"  Accuracy          = {acc:.3f}   (chance ≈ {chance:.3f}, {n_classes} classes)")
    print(f"  Balanced accuracy = {bacc:.3f}   (métrique clé, robuste au déséquilibre)")
    print()
    print("  VERDICT (basé sur balanced accuracy) :")
    if bacc < chance * 1.5:
        print("  >>> ÉCHEC : embeddings NON séparables même avec poids équilibrés.")
        print("  >>> Cause = LABELS BRUITÉS (annotations mal assignées aux nœuds)")
        print("  >>>         ou tâche intrinsèquement trop difficile.")
        print("  >>> Vérifier tag_nodes_with_annotations (chevauchement de polygones).")
    elif bacc > 0.45:
        print("  >>> SUCCÈS : embeddings SÉPARABLES une fois le déséquilibre neutralisé.")
        print("  >>> Le collapse vient de la GESTION DU DÉSÉQUILIBRE à l'entraînement.")
        print("  >>> Les poids sqrt actuels sont trop faibles → passer à 'balanced',")
        print("  >>>     focal loss, ou oversampling des classes minoritaires.")
    else:
        print("  >>> INTERMÉDIAIRE : signal faible. Embeddings partiellement utiles.")
        print("  >>> Vérifier normalisation + qualité des annotations.")


# ──────────────────────────────────────────────
# Étape C — labeling : chevauchement & visu de contrôle
# ──────────────────────────────────────────────

def _build_tagged_graph(wsi_id=None):
    """Construit + tagge un WSIHexGraph pour une lame (la 1ère si wsi_id=None)."""
    from wsi_graph import WSIHexGraph

    pairs = config.discover_wsi_pairs()
    if not pairs:
        raise FileNotFoundError(f"Aucune paire WSI/GeoJSON dans {config.DATA_DIR}")

    pair = pairs[0] if wsi_id is None else next(
        (p for p in pairs if p["wsi_id"] == wsi_id), None
    )
    if pair is None:
        raise ValueError(f"wsi_id '{wsi_id}' introuvable. Dispo: "
                         f"{[p['wsi_id'] for p in pairs]}")

    print(f"Construction du graphe pour : {pair['wsi_id']}")
    g = WSIHexGraph(
        pair["svs"],
        patch_size=config.PATCH_SIZE,
        white_threshold=config.WHITE_THRESHOLD,
        white_ratio=config.WHITE_RATIO,
    )
    g.build_graph()
    g.load_annotations(pair["geojson"])
    g.tag_nodes_with_annotations()
    return g


def verify_labels(wsi_id=None, downsample=16, out_path=None):
    """Génère l'image de contrôle des labels (smallest-polygon-wins appliqué).

    Superpose les centres des nœuds colorés par label + les contours des
    annotations sur la miniature de la lame.
    """
    from visualize import plot_node_labels_on_wsi

    g = _build_tagged_graph(wsi_id)
    path = plot_node_labels_on_wsi(g, out_path=out_path, downsample=downsample)
    print(f"\nVisu de contrôle générée : {path}")
    return path


def step_c_overlap(wsi_id=None):
    """Quantifie le chevauchement et compare ancien (last-write-wins)
    vs nouveau (smallest-polygon-wins) labeling sur une lame.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    from collections import Counter
    from wsi_graph import _extract_label

    g = _build_tagged_graph(wsi_id)

    nodes_data = [
        {"node_id": nid, "geometry": Point(d["cx"], d["cy"])}
        for nid, d in g.graph.nodes(data=True)
    ]
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=g.gdf.crs)
    poly_area = g.gdf.geometry.area
    joined = gpd.sjoin(nodes_gdf, g.gdf, how="inner", predicate="within").copy()
    joined["poly_area"] = poly_area.loc[joined["index_right"]].values
    joined["cls"] = joined["classification"].apply(_extract_label)

    print()
    print("=" * 70)
    print(f"ÉTAPE C — Chevauchement & comparaison labeling")
    print("=" * 70)

    sizes = joined.groupby("node_id").size()
    n_total = g.graph.number_of_nodes()
    n_multi = int((sizes > 1).sum())
    print(f"  Nœuds total           : {n_total}")
    print(f"  Nœuds matchés         : {len(sizes)}")
    print(f"  Nœuds multi-polygones : {n_multi} "
          f"({100*n_multi/max(len(sizes),1):.1f}% des matchés)")

    # ANCIEN : last-write-wins (ordre de la jointure)
    old = {}
    for _, row in joined.iterrows():
        old[row["node_id"]] = row["cls"]

    # NOUVEAU : sous-type prioritaire sur grossier, puis plus petite aire
    coarse = set(getattr(config, "EXCLUDED_LABELS",
                         ["background", "no_Tissu", "no_Tumor", "Tumor"]))
    joined["is_coarse"] = joined["cls"].isin(coarse)
    new = (joined.sort_values(["is_coarse", "poly_area"], ascending=[True, True])
                 .groupby("node_id", sort=False).first()["cls"].to_dict())

    print(f"\n  Distribution ANCIEN (last-write) : {dict(Counter(old.values()))}")
    print(f"  Distribution NOUVEAU (smallest)  : {dict(Counter(new.values()))}")

    changed = [(old[n], new[n]) for n in old if old[n] != new[n]]
    print(f"\n  Labels qui CHANGENT : {len(changed)} / {len(old)} "
          f"({100*len(changed)/max(len(old),1):.1f}%)")
    if changed:
        top = Counter(changed).most_common(12)
        print("  Transitions (ancien → nouveau) les plus fréquentes :")
        for (o, nw), c in top:
            print(f"    {o:20s} → {nw:20s} : {c}")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else config.EMBEDDING_MODEL
    index_df, feats = step_a(model_name)
    step_b(index_df, feats)


if __name__ == "__main__":
    main()
