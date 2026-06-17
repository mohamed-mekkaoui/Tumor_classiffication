"""
Embedding Visualization
───────────────────────
Projette en 2D (PCA ou t-SNE) les embeddings de patchs pour quelques classes,
afin d'inspecter VISUELLEMENT leur séparabilité. C'est le complément visuel de
``diagnose.py`` (qui, lui, mesure la séparabilité par régression logistique).

Chaque point = un patch (nœud du graphe), coloré par sa classe d'annotation.
Si les modèles de fondation (UNI…) séparent bien les sous-types, les nuages de
points des différentes classes doivent former des groupes distincts.

Usage (Colab / notebook) :
    import visualize_embeddings as ve

    # 3 classes au choix, projection t-SNE
    ve.plot_embeddings_2d("uni",
                          classes=["NÉCROSE", "SOLIDE", "LÉPIDIQUE"],
                          method="tsne", per_class=500)

    # autre trio, projection PCA (plus rapide)
    ve.plot_embeddings_2d("uni",
                          classes=["MICROPAPILLAIRE", "STROMA_FIBREUX", "CRIBRIFORME"],
                          method="pca")

    # sans préciser les classes → 3 classes choisies automatiquement
    ve.plot_embeddings_2d("uni")
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config


def _resolve_classes(classes, index_df):
    """Valide la liste de classes (noms). Si None, choisit 3 classes
    non-exclues bien peuplées."""
    counts = index_df["label"].value_counts()
    excluded = set(getattr(config, "EXCLUDED_LABELS", []))

    if classes is None:
        classes = [c for c in counts.index if c not in excluded][:3]
        print(f"Classes (auto) : {classes}")

    valid = set(index_df["label"].unique())
    bad = [c for c in classes if c not in valid]
    if bad:
        raise ValueError(
            f"Classes introuvables dans index.csv : {bad}\n"
            f"Disponibles : {list(counts.index)}"
        )
    return list(classes)


def plot_embeddings_2d(model_name=None, classes=None, method="tsne",
                       per_class=500, out_path=None, seed=42, show=True):
    """Projette en 2D les embeddings de quelques classes et trace un scatter.

    Args:
        model_name : modèle d'embedding (défaut config.EMBEDDING_MODEL)
        classes    : liste de noms de classes, ex. ["NÉCROSE","SOLIDE","LÉPIDIQUE"].
                     None → 3 classes choisies automatiquement.
        method     : "tsne" (meilleure séparation, défaut) ou "pca" (rapide)
        per_class  : nb de points échantillonnés par classe (borné par le dispo)
        out_path   : chemin PNG (défaut MODELS_DIR/embeddings_2d_<model>_<method>.png)
        seed       : graine reproductible
        show       : appelle plt.show() (affichage inline sur Colab)

    Returns:
        out_path : chemin du PNG sauvegardé
    """
    from embedding_extractor import load_embeddings

    model_name = model_name or config.EMBEDDING_MODEL
    rng = np.random.default_rng(seed)

    index_df = pd.read_csv(os.path.join(config.WALKS_DIR, "index.csv"))
    classes = _resolve_classes(classes, index_df)

    feats, embed_dim = load_embeddings(model_name)

    # 1. Échantillonner les nœuds par classe
    print(f"Échantillonnage ({per_class}/classe max) :")
    gidx_parts, labels = [], []
    for cls in classes:
        pool = index_df.index[index_df["label"] == cls].to_numpy()
        n = min(per_class, len(pool))
        chosen = rng.choice(pool, size=n, replace=False)
        gidx_parts.append(index_df.loc[chosen, "global_idx"].to_numpy())
        labels += [cls] * n
        print(f"  {cls:20s}: {n} points")

    gidx_all = np.concatenate(gidx_parts)
    labels = np.array(labels)

    # 2. Charger les embeddings + retirer d'éventuelles lignes nulles
    X = np.asarray(feats[gidx_all], dtype=np.float32)
    keep = ~(X == 0).all(axis=1)
    if (~keep).any():
        print(f"  ({int((~keep).sum())} lignes nulles retirées)")
    X, labels = X[keep], labels[keep]

    # 3. Standardiser puis réduire en 2D
    X = StandardScaler().fit_transform(X)

    if method == "pca":
        emb2d = PCA(n_components=2, random_state=seed).fit_transform(X)
        title_method = "PCA"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        # PCA -> 50 dims d'abord (pratique standard : accélère et débruite t-SNE)
        n_pca = min(50, X.shape[0], X.shape[1])
        if X.shape[1] > n_pca:
            X = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
        perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
        emb2d = TSNE(n_components=2, init="pca", random_state=seed,
                     perplexity=perplexity).fit_transform(X)
        title_method = "t-SNE"
    else:
        raise ValueError("method doit être 'pca' ou 'tsne'")

    # 4. Scatter coloré par classe
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = plt.cm.tab10.colors
    for i, cls in enumerate(classes):
        m = labels == cls
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=10, alpha=0.6,
                   color=palette[i % 10], label=f"{cls} ({int(m.sum())})")

    ax.set_title(f"Embeddings '{model_name}' — projection {title_method} "
                 f"({len(classes)} classes)")
    ax.set_xlabel(f"{title_method} 1")
    ax.set_ylabel(f"{title_method} 2")
    ax.legend(markerscale=2, loc="best", framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if out_path is None:
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        out_path = os.path.join(
            config.MODELS_DIR, f"embeddings_2d_{model_name}_{method}.png"
        )
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path
