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
PATCH_SIZE = 224
WHITE_THRESHOLD = 220
WHITE_RATIO = 0.5

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
# "cls"    → CLS token uniquement  → Linear(d_model, num_classes)
# "mean"   → mean pooling uniquement → Linear(d_model, num_classes)
# "concat" → [CLS ; mean]          → Linear(2*d_model, num_classes)
AGGREGATION = "concat"

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 32
TOPK_AGG = 4  # top-k mean for WSI-level aggregation

USE_CLASS_WEIGHTS = True  # weight CrossEntropyLoss by inverse class frequency

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
