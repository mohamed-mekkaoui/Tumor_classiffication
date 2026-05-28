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
    from sklearn.metrics import accuracy_score, classification_report
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

    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    inv = {v: k for k, v in config.LABEL_MAP.items()}
    names = [inv.get(c, str(c)) for c in sorted(np.unique(y))]
    print()
    print(classification_report(yte, pred, target_names=names, zero_division=0))
    print(f"  Accuracy = {acc:.3f}   (chance ≈ {chance:.3f}, {n_classes} classes)")
    print()
    print("  VERDICT :")
    if acc < chance * 1.5:
        print("  >>> ÉCHEC : les embeddings ne sont PAS séparables linéairement.")
        print("  >>> Le problème est dans les EMBEDDINGS (zéros / désalignement / corruption).")
        print("  >>> Ne pas toucher au modèle. Re-extraire patches + embeddings proprement.")
    elif acc > 0.6:
        print("  >>> SUCCÈS : les embeddings SONT discriminants.")
        print("  >>> Le problème est dans le CHEMIN SÉQUENCE/TRANSFORMER")
        print("  >>> (normalisation d'entrée, masking, ou redondance/fuite des walks).")
    else:
        print("  >>> INTERMÉDIAIRE : signal faible. Embeddings partiellement utiles.")
        print("  >>> Vérifier normalisation + qualité des annotations.")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else config.EMBEDDING_MODEL
    index_df, feats = step_a(model_name)
    step_b(index_df, feats)


if __name__ == "__main__":
    main()
