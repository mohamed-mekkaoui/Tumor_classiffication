"""
Analyse de la distribution des classes d'annotations dans les fichiers GeoJSON.
Calcule le nombre estimé de patches (224x224) par classe et détecte les déséquilibres.
Génère des plots Plotly interactifs (HTML + PNG) pour présentation.
"""
import os

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

DATA_DIR = os.path.join(os.path.dirname(__file__), "../DATA")
PATCH_SIZE = 224
PATCH_AREA = PATCH_SIZE * PATCH_SIZE  # 50176 px²


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _save(fig, path_no_ext):
    """Saves a Plotly figure as HTML (always) and PNG (if kaleido installed)."""
    fig.write_html(f"{path_no_ext}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(f"{path_no_ext}.png", scale=3)
        print(f"    → {path_no_ext}.png")
    except (ValueError, ImportError):
        pass
    print(f"    → {path_no_ext}.html")


# ──────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────

def _collect_stats():
    """Parse all GeoJSON files and returns two DataFrames.

    Returns:
        global_df : one row per class (name, polygones, patches, pct)
        per_slide_df : one row per (slide, class)
    """
    geojson_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".geojson") and "pred" not in f.lower()
    )
    if not geojson_files:
        print(f"Aucun fichier GeoJSON trouvé dans {DATA_DIR}")
        return None, None

    rows = []
    for filename in geojson_files:
        path = os.path.join(DATA_DIR, filename)
        gdf = gpd.read_file(path)
        slide_name = os.path.splitext(filename)[0]

        for _, row in gdf.iterrows():
            classification = row["classification"]
            if isinstance(classification, dict):
                name = classification.get("name", "unknown")
            else:
                name = str(classification)

            area = row.geometry.area
            patches_approx = area / PATCH_AREA
            rows.append({
                "slide": slide_name,
                "class": name,
                "polygones": 1,
                "patches": patches_approx,
            })

    df = pd.DataFrame(rows)

    # Per-slide aggregation
    per_slide_df = (
        df.groupby(["slide", "class"], as_index=False)
        .agg(polygones=("polygones", "sum"), patches=("patches", "sum"))
    )

    # Global aggregation
    global_df = (
        df.groupby("class", as_index=False)
        .agg(polygones=("polygones", "sum"), patches=("patches", "sum"))
    )
    total = global_df["patches"].sum()
    global_df["pct"] = global_df["patches"] / total * 100
    global_df = global_df.sort_values("patches", ascending=False).reset_index(drop=True)

    return global_df, per_slide_df


# ──────────────────────────────────────────────
# Plot 1 — Distribution globale
# ──────────────────────────────────────────────

def plot_global_distribution(global_df, output_dir):
    """Bar chart — estimated patches per class (all slides combined)."""
    df = global_df.sort_values("patches", ascending=True)  # horizontal → ascending

    fig = go.Figure(go.Bar(
        x=df["patches"],
        y=df["class"],
        orientation="h",
        text=[f"{v:,.0f}  ({p:.1f}%)" for v, p in zip(df["patches"], df["pct"])],
        textposition="outside",
        marker_color=px.colors.qualitative.Plotly[: len(df)],
    ))

    fig.update_layout(
        title="Distribution globale des patches par classe",
        xaxis_title="Nombre estimé de patches",
        yaxis_title="",
        template="plotly_white",
        height=max(400, 45 * len(df) + 150),
        width=900,
        font=dict(size=14),
        margin=dict(l=180),
    )

    _save(fig, os.path.join(output_dir, "class_distribution_global"))
    return fig


# ──────────────────────────────────────────────
# Plot 2 — Distribution par lame (stacked)
# ──────────────────────────────────────────────

def plot_per_slide_distribution(per_slide_df, output_dir):
    """Grouped bar chart — patches per class, grouped by slide."""
    fig = px.bar(
        per_slide_df.sort_values("patches", ascending=False),
        x="class",
        y="patches",
        color="slide",
        barmode="group",
        text=per_slide_df["patches"].apply(lambda v: f"{v:,.0f}"),
        title="Distribution des patches par classe et par lame",
        labels={"patches": "Nb patches estimé", "class": "Classe", "slide": "Lame"},
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
        width=max(800, 70 * per_slide_df["class"].nunique() + 200),
        font=dict(size=13),
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0.5,
                    xanchor="center"),
    )
    fig.update_traces(textposition="outside", textfont_size=10)

    _save(fig, os.path.join(output_dir, "class_distribution_per_slide"))
    return fig


# ──────────────────────────────────────────────
# Plot 3 — Pie chart (proportions)
# ──────────────────────────────────────────────

def plot_pie(global_df, output_dir):
    """Pie chart — proportion of each class."""
    fig = go.Figure(go.Pie(
        labels=global_df["class"],
        values=global_df["patches"],
        textinfo="label+percent",
        textposition="outside",
        hole=0.35,
        marker=dict(colors=px.colors.qualitative.Plotly[: len(global_df)]),
    ))

    fig.update_layout(
        title="Proportion des classes (% de patches)",
        template="plotly_white",
        height=550,
        width=700,
        font=dict(size=13),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5,
                    xanchor="center"),
    )

    _save(fig, os.path.join(output_dir, "class_proportion_pie"))
    return fig


# ──────────────────────────────────────────────
# Plot 4 — Déséquilibre (log scale)
# ──────────────────────────────────────────────

def plot_imbalance(global_df, output_dir):
    """Bar chart on log scale to highlight class imbalance."""
    df = global_df.sort_values("patches", ascending=False)
    max_val = df["patches"].max()
    min_val = df["patches"].min()
    ratio = max_val / min_val if min_val > 0 else float("inf")

    fig = go.Figure(go.Bar(
        x=df["class"],
        y=df["patches"],
        text=[f"{v:,.0f}" for v in df["patches"]],
        textposition="outside",
        marker_color=px.colors.sequential.Reds_r[: len(df)],
    ))

    fig.update_layout(
        title=f"Déséquilibre des classes (rapport max/min = ×{ratio:,.0f})",
        xaxis_title="Classe",
        yaxis_title="Nb patches (échelle log)",
        yaxis_type="log",
        template="plotly_white",
        height=500,
        width=max(700, 65 * len(df) + 150),
        font=dict(size=13),
        xaxis_tickangle=-30,
    )

    _save(fig, os.path.join(output_dir, "class_imbalance_log"))
    return fig


# ──────────────────────────────────────────────
# Plot 5 — Heatmap slides × classes
# ──────────────────────────────────────────────

def plot_slide_class_heatmap(per_slide_df, output_dir):
    """Heatmap of patch counts per (slide, class)."""
    pivot = per_slide_df.pivot_table(
        index="slide", columns="class", values="patches", fill_value=0
    )

    text = [[f"{v:,.0f}" for v in row] for row in pivot.values]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale="YlOrRd",
        colorbar=dict(title="Patches"),
    ))

    fig.update_layout(
        title="Patches par lame × classe",
        template="plotly_white",
        height=max(350, 60 * len(pivot) + 200),
        width=max(700, 55 * len(pivot.columns) + 200),
        font=dict(size=13),
        yaxis=dict(autorange="reversed"),
        xaxis_tickangle=-30,
    )

    _save(fig, os.path.join(output_dir, "slide_class_heatmap"))
    return fig


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def analyze_all_slides(output_dir=None, show=True):
    """Analyse complète + génération de tous les plots.

    Args:
        output_dir: dossier de sauvegarde (défaut: config.MODELS_DIR)
        show:       si True, affiche les figures inline (Colab/Jupyter)

    Returns:
        global_df, per_slide_df
    """
    output_dir = output_dir or getattr(config, "MODELS_DIR",
                                       os.path.join("output", "models"))
    os.makedirs(output_dir, exist_ok=True)

    global_df, per_slide_df = _collect_stats()
    if global_df is None:
        return None, None

    # ── Résumé texte ──────────────────────────
    total = global_df["patches"].sum()
    print(f"{'='*60}")
    print(f"  RÉSUMÉ — {per_slide_df['slide'].nunique()} lames, "
          f"{len(global_df)} classes, ~{total:,.0f} patches estimés")
    print(f"{'='*60}")
    print(global_df.to_string(index=False, float_format=lambda v: f"{v:,.0f}"
                              if v > 10 else f"{v:.1f}"))

    max_c = global_df.iloc[0]
    min_c = global_df.iloc[-1]
    ratio = max_c["patches"] / min_c["patches"] if min_c["patches"] > 0 else float("inf")
    print(f"\n  Rapport max/min : {max_c['class']} / {min_c['class']} = ×{ratio:,.0f}")

    # ── Sauvegarde CSV ────────────────────────
    csv_global = os.path.join(output_dir, "class_balance_global.csv")
    csv_slide = os.path.join(output_dir, "class_balance_per_slide.csv")
    global_df.to_csv(csv_global, index=False)
    per_slide_df.to_csv(csv_slide, index=False)
    print(f"\n  CSV sauvegardés :")
    print(f"    → {csv_global}")
    print(f"    → {csv_slide}")

    # ── Plots ─────────────────────────────────
    print(f"\n  Génération des plots...")
    figs = []
    figs.append(plot_global_distribution(global_df, output_dir))
    figs.append(plot_per_slide_distribution(per_slide_df, output_dir))
    figs.append(plot_pie(global_df, output_dir))
    figs.append(plot_imbalance(global_df, output_dir))
    figs.append(plot_slide_class_heatmap(per_slide_df, output_dir))

    if show:
        for fig in figs:
            fig.show()

    print(f"\n  Terminé — {len(figs)} plots générés dans {output_dir}")
    return global_df, per_slide_df


if __name__ == "__main__":
    analyze_all_slides(show=False)
