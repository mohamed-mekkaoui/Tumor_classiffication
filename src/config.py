import hashlib
import json
import os

import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "DATA")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Sub-directories (created automatically)
WALKS_DIR = os.path.join(OUTPUT_DIR, "walks")
PATCHES_DIR = os.path.join(OUTPUT_DIR, "patches")
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "embeddings")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Graph parameters
# ──────────────────────────────────────────────
PATCH_SIZE = 224          # taille de sortie / entrée modèle (memmap + transform)
WHITE_THRESHOLD = 220
WHITE_RATIO = 0.5

# ── Grossissement / résolution ────────────────
# Les lames sont scannées en 40× (≈ 0,25 µm/px) mais les modèles de fondation
# (UNI, UNI2-h, H-optimus) sont entraînés en 20× (≈ 0,5 µm/px). On lit donc une
# tuile 2× plus grande au niveau 0 (tile_l0 = PATCH_SIZE * scale) puis on la
# redimensionne à PATCH_SIZE, avec scale = round(TARGET_MPP / mpp_natif).
TARGET_MPP = 0.5          # µm/px visé (20×, attendu par UNI/UNI2-h/H-optimus)
DEFAULT_SLIDE_MPP = 0.25  # fallback si la lame n'expose pas openslide.mpp-x

# ──────────────────────────────────────────────
# Random walk parameters
# ──────────────────────────────────────────────
WALKS_PER_WSI = 50
WALK_MIN_LENGTH = 30
WALK_MAX_LENGTH = 40
SHARP_TURN_WEIGHT = 0.1

# Region-constrained walks
WALKS_PER_REGION = 30          # walks per connected component of a class
EXCLUDED_LABELS = ["background", "no_Tissu", "no_Tumor", "Tumor"]  # labels to skip during walk generation
MIN_REGION_SIZE = 30           # min nodes in a connected component to generate walks
WALK_BOUNCE = True             # True = bounce at boundaries, False = stop at boundaries

# ── Génération équilibrée par classe (balance plafonnée, size-aware) ──
# True  → vise WALKS_PER_CLASS walks PAR CLASSE, réparti sur ses régions selon leur taille,
#         avec un plafond de redondance (MAX_WALK_REDUNDANCY). Les classes pauvres en tissu
#         plafonnent honnêtement sous la cible.
# False → ancien comportement (WALKS_PER_REGION walks par composante).
BALANCE_WALKS       = True
WALKS_PER_CLASS     = 2000     # budget de walks visé par classe non-exclue
MAX_WALK_REDUNDANCY = 50       # nb de fois qu'un patch est réutilisé EN MOYENNE (plafond)

# ──────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────
# With 6 WSIs: 4 train / 1 val / 1 test
SPLIT_SEED = 42

# Stratified walk-level split (mélange tous les walks puis sépare)
STRATIFIED_SPLIT = True     # True = walk-level stratified, False = WSI-level
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

# Region-level holdout split (remplace STRATIFIED_SPLIT si True)
# Hold out N régions entières par classe comme test set → élimine le data leakage.
REGION_HOLDOUT_SPLIT = False
TEST_REGION_SEED     = 42
# int global OU dict par classe ; clé "default" = fallback pour les classes non listées.
# ACINAIRE/MICROPAPILLAIRE/CRIBRIFORME ont de petites régions → en prendre plusieurs.
TEST_REGIONS_PER_CLASS = {
    "default":           1,
    "ACINAIRE":          5,
    "MICROPAPILLAIRE":   2,
    "CRIBRIFORME":       2,
    "COMPLEX_GLANDULAR": 2,
}

# ──────────────────────────────────────────────
# Embedding model
# ──────────────────────────────────────────────
# Choices: "dinov2", "vit", "uni"
EMBEDDING_MODEL = "dinov2"
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_NUM_WORKERS = 4  # safe now that patches are pre-extracted (no OpenSlide)

# Model name → (timm model name, embed_dim)
EMBEDDING_REGISTRY = {
    "dinov2":      ("vit_base_patch14_dinov2.lvd142m",    768),
    "vit":         ("vit_base_patch16_224",                768),
    "uni":         ("hf-hub:MahmoodLab/uni",              1024),
    "uni2-h":      ("hf-hub:MahmoodLab/UNI2-h",          1536),
    "h-optimus":   ("hf-hub:bioptimus/H-optimus-1",       1536),
    "h-optimus-0": ("hf-hub:bioptimus/H-optimus-0",       1536),
}

# ──────────────────────────────────────────────
# Transformer / Training
# ──────────────────────────────────────────────
# Label map: class name → integer index
LABEL_MAP = {
    "background":         0,
    "no_Tissu":           1,
    "no_Tumor":           2,
    "Tumor":              3,
    "ACINAIRE":           4,
    "LÉPIDIQUE":          5,
    "MICROPAPILLAIRE":    6,
    "COMPLEX_GLANDULAR":  7,
    "STROMA_FIBREUX":     8,
    "STROMA_INFLAM":      9,
    "NÉCROSE":           10,
    "SOLIDE":            11,
    "CRIBRIFORME":       12,
    "PAPILLAIRE":        13,
}
NUM_CLASSES = len(LABEL_MAP)

D_MODEL = 256
# Si True : pas de projection, le Transformer travaille à la dim native de l'embedder
# Si False : projection Linear(in_dim, D_MODEL) — compression vers D_MODEL
USE_NATIVE_DIM = False
# NHEAD = D_MODEL / 64  (règle du papier Vaswani 2017 : dim/tête ≈ 64)
# D_MODEL=256 → 4 | D_MODEL=512 → 8 | native 1024 → 16 | native 1536 → 24
NHEAD = 4
NUM_LAYERS = 4
DROPOUT = 0.1

# Backbone architecture
# "transformer" : custom nn.TransformerEncoder + sinusoidal PE + CLS token + mean pooling
# "bert"        : HuggingFace BertModel (from scratch, no pre-trained weights) + CLS token + mean pooling
BACKBONE = "bert"

# Agrégation avant la tête de classification
# "cls"       → CLS token uniquement          → Linear(d_model, num_classes)
# "mean"      → mean pooling uniforme          → Linear(d_model, num_classes)
# "concat"    → [CLS ; mean]                  → Linear(2*d_model, num_classes)
# "attention" → gated attention pooling (ABMIL) → Linear(d_model, num_classes)
AGGREGATION = "concat"

# Dimension cachée du MLP d'attention (utilisé si AGGREGATION="attention")
ATTENTION_HIDDEN_DIM = 128

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 32

USE_CLASS_WEIGHTS = True  # weight CrossEntropyLoss by inverse class frequency
WEIGHT_MODE = "sqrt"      # "sqrt" (atténué) | "balanced" (plein inverse de fréquence)
GRAD_CLIP = 1.0           # max_norm for gradient clipping (0 = disabled)

# ── Entraînement en deux phases ────────────────
# Phase 1 : geler l'encoder, entraîner proj + input_norm + cls_head (K premiers epochs)
# Phase 2 : dégeler l'encoder, fine-tuner tout avec LR * PHASE2_LR_FACTOR
# 0 = désactivé (entraînement classique en une seule phase)
FREEZE_ENCODER_EPOCHS = 0
PHASE2_LR_FACTOR      = 0.1

# ── LR Scheduler ──────────────────────────────
# Options: "cosine" | "plateau" | "cosine_restart" | None
SCHEDULER = "cosine"

# CosineAnnealingLR
SCHEDULER_T_MAX  = None   # None = use EPOCHS
SCHEDULER_ETA_MIN = 1e-6

# ReduceLROnPlateau
SCHEDULER_PLATEAU_PATIENCE = 3
SCHEDULER_PLATEAU_FACTOR   = 0.5

# CosineAnnealingWarmRestarts
SCHEDULER_T_0    = 10
SCHEDULER_T_MULT = 1

# ──────────────────────────────────────────────
# Overfit sanity check
# ──────────────────────────────────────────────
# Set OVERFIT_TEST=True to train on N fixed samples (same set used for val).
# A correct model should reach ~100% train accuracy within a few epochs.
# If it doesn't, there is a bug in data loading, model, or loss computation.
OVERFIT_TEST = False
OVERFIT_N_SAMPLES = 32

# ──────────────────────────────────────────────
# Auto-discover WSI / GeoJSON pairs
# ──────────────────────────────────────────────

def discover_wsi_pairs():
    """Finds matching (.svs, .geojson) pairs in DATA_DIR.
    Ignores *_pred.geojson files (model predictions).
    Returns list of dicts: [{"wsi_id": ..., "svs": ..., "geojson": ...}, ...]
    """
    svs_files = {}
    geojson_files = {}

    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".svs"):
            key = os.path.splitext(f)[0]
            svs_files[key] = os.path.join(DATA_DIR, f)
        elif f.lower().endswith(".geojson") and "pred" not in f.lower():
            key = os.path.splitext(f)[0]
            geojson_files[key] = os.path.join(DATA_DIR, f)

    pairs = []
    for key in sorted(svs_files.keys()):
        if key in geojson_files:
            pairs.append({
                "wsi_id": key,
                "svs": svs_files[key],
                "geojson": geojson_files[key],
            })
        else:
            print(f"Warning: no matching GeoJSON for {key}.svs")

    return pairs


# ──────────────────────────────────────────────
# Checkpoint fingerprint (anti-staleness)
# ──────────────────────────────────────────────
# Empêche la réutilisation de patches/embeddings périmés quand le graphe
# change : si le fingerprint du index.csv courant ne correspond pas à celui
# enregistré à l'extraction, le checkpoint est invalidé (ré-extraction).

def index_fingerprint(index_df):
    """Empreinte déterministe du index.csv basée sur N et les positions
    de chaque nœud (wsi_id, px, py). Insensible à l'ordre des colonnes,
    sensible à tout changement de graphe.
    """
    cols = ["wsi_id", "px", "py"]
    key = "|".join(
        index_df[c].astype(str).str.cat(sep=",") for c in cols
    )
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return {"n": int(len(index_df)), "sha256": h}


def write_checkpoint_meta(meta_path, index_df, **extra):
    """Écrit le fingerprint (+ infos extra) à côté d'un done.npy."""
    meta = index_fingerprint(index_df)
    meta.update(extra)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def checkpoint_is_valid(meta_path, index_df, **extra):
    """True si le meta enregistré correspond au index.csv courant
    (et aux champs extra fournis, ex: embed_dim). False sinon ou si absent.
    """
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    expected = index_fingerprint(index_df)
    expected.update(extra)
    return all(saved.get(k) == v for k, v in expected.items())
