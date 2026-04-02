# Notebooks — Pipeline expérimental (Sujet 7 — Méthode 1)

Ce dossier contient l'ensemble des notebooks Jupyter implémentant le pipeline expérimental du projet **Classification de Whole Slide Images sous supervision faible**.

Les notebooks sont conçus pour être **lus et exécutés dans l'ordre**. Ils ont été développés pour **Kaggle** (accès GPU, datasets Camelyon).

---

## 📋 Liste des notebooks (ordre d'exécution)

| # | Fichier | Description |
|---|---------|-------------|
| 0 | `00-preproc-camelyon-patch-extraction (1).ipynb` | Extraction et filtrage des patchs WSI (tiling 256×256, Laplacien + HSV pour exclure fond) |
| 1 | `01-wsi-level-stratified-split (1).ipynb` | Split stratifié au niveau WSI (train/val/test) — pas de fuite patient |
| 2 | `02-patch-embedding-extraction-vit-checkpoint (5).ipynb` | Extraction des embeddings patch-level via DINOv2 (ViT) |
| 3 | `03-str-randomwalk-sequence-generation-dinoembedv4 (2).ipynb` | Génération des chemins spatiaux SPPR (random walks) |
| 4 | `04-str-dinov2-training-v4-1 (4).ipynb` | Entraînement du Transformer + agrégation MIL |
| 5 | `05-dinov2-embeddings-extraction (1).ipynb` | Évaluation, extraction d'embeddings et analyses complémentaires |

---

## 🧪 Étapes du pipeline (résumé)

1. **00** — Chargement WSIs Camelyon → découpage en patchs → filtrage tissu vs fond → sortie `.png` + métadonnées CSV
2. **01** — Lecture du CSV → split stratifié WSI-level → nouveau CSV avec colonnes `split` (train/val/test)
3. **02** — Chargement patchs → forward DINOv2 → sauvegarde embeddings (`.npy` ou équivalent)
4. **03** — Construction de séquences spatiales (SPPR) à partir des positions des patchs
5. **04** — Dataset séquentiel → Transformer → agrégation MIL → entraînement
6. **05** — Évaluation sur test, extraction d'embeddings finaux, visualisations

---

## ⚙️ Dépendances principales

- `openslide`, `PIL`, `opencv-python`, `numpy`, `pandas`, `matplotlib`
- `torch`, `torchvision` (DINOv2)
- `tqdm`

---

## ⚠️ Notes importantes

- **Aucun label patch-level** n'est utilisé — supervision WSI-level uniquement
- **Pas de fuite de données** — split strict au niveau WSI (et patient si possible)
- Les chemins dans les notebooks pointent vers des datasets Kaggle (`/kaggle/input/...`)
- Pour exécuter localement, adapter les chemins et monter les données Camelyon

---

## 👤 Auteur

**Mohammed Wassim Chikhi** — Master 2 VMI — Université Paris Cité
