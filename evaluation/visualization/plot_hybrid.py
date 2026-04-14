import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")

ACCUMULATED_DATA = "evaluation/accumulated_data/offline_metrics.csv"
HYBRID_RESULTS_LEGACY = "evaluation/hybrid/dataset_hybrid_results.csv"
OUTPUT_DIR = "evaluation/visualization/plots/"


def load_hybrid_data():
    """
    Carga datos de MACE desde accumulated_data o CSV legacy
    """
    accumulated_path = Path(ACCUMULATED_DATA)

    if accumulated_path.exists():
        print(f"Cargando MACE desde: {ACCUMULATED_DATA}")
        df = pd.read_csv(ACCUMULATED_DATA)

        # Filtrar solo offline y con MACE data
        df = df[(df["source"] == "offline") & (df["mace_score"].notna())].copy()

        # Convertir a numérico por si acaso
        numeric_cols = [
            "mace_score",
            "mace_layer1",
            "mace_layer2",
            "mace_layer3",
            "mace_layer3_used",
            "mace_time",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        print(f"Cargados {len(df)} casos desde accumulated_data")
        return df, "accumulated"

    else:
        print(f"No se encontró {ACCUMULATED_DATA}, buscando CSV legacy...")

        legacy_path = Path(HYBRID_RESULTS_LEGACY)
        if legacy_path.exists():
            df = pd.read_csv(HYBRID_RESULTS_LEGACY)
            print(f"Cargados {len(df)} casos desde CSV legacy")
            return df, "legacy"
        else:
            print(f"No se encontró {HYBRID_RESULTS_LEGACY}")
            return pd.DataFrame(), "none"


def plot_hybrid_layers_distribution():
    """Distribución de scores por capa (Layer 1, 2, 3)"""
    df, _ = load_hybrid_data()

    if df.empty:
        print("No hay datos MACE, omitiendo gráfica.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MACE - Distribución de Scores por Capa", fontsize=16, fontweight="bold"
    )

    # Layer 1
    axes[0, 0].hist(
        df["mace_layer1"].dropna(),
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color="#3498db",
    )
    axes[0, 0].axvline(
        df["mace_layer1"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Media: {df["mace_layer1"].mean():.3f}',
    )
    axes[0, 0].set_title("Layer 1: Guardrails (Deterministic)")
    axes[0, 0].set_xlabel("Score (0-1)")
    axes[0, 0].set_ylabel("Frecuencia")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Layer 2
    axes[0, 1].hist(
        df["mace_layer2"].dropna(),
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color="#e74c3c",
    )
    axes[0, 1].axvline(
        df["mace_layer2"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Media: {df["mace_layer2"].mean():.3f}',
    )
    axes[0, 1].set_title("Layer 2: Semantic (ML)")
    axes[0, 1].set_xlabel("Score (0-1)")
    axes[0, 1].set_ylabel("Frecuencia")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Layer 3 (solo casos donde se usó)
    df_layer3 = df[df["mace_layer3"].notna()]
    if len(df_layer3) > 0:
        axes[1, 0].hist(
            df_layer3["mace_layer3"],
            bins=15,
            edgecolor="black",
            alpha=0.7,
            color="#f39c12",
        )
        axes[1, 0].axvline(
            df_layer3["mace_layer3"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Media: {df_layer3["mace_layer3"].mean():.3f}',
        )
        axes[1, 0].set_title(
            f"Layer 3: LLM-Judge Selectivo (usado en {len(df_layer3)} casos)"
        )
        axes[1, 0].set_xlabel("Score (0-1)")
        axes[1, 0].set_ylabel("Frecuencia")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Layer 3 no usado en ningún caso",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1, 0].axis("off")

    # Score Final
    axes[1, 1].hist(
        df["mace_score"].dropna(),
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color="#9b59b6",
    )
    axes[1, 1].axvline(
        df["mace_score"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Media: {df["mace_score"].mean():.3f}',
    )
    axes[1, 1].set_title("Score Final (Weighted Fusion)")
    axes[1, 1].set_xlabel("Score (0-1)")
    axes[1, 1].set_ylabel("Frecuencia")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}hybrid_layers_distribution.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: hybrid_layers_distribution.png")
    plt.close()


def plot_layer3_usage():
    """Análisis del uso de Layer 3"""
    df, _ = load_hybrid_data()

    if df.empty:
        print("No hay datos MACE, omitiendo gráfica.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart de uso de Layer 3
    layer3_counts = df["mace_layer3_used"].value_counts()
    labels = ["No Escalado (Layers 1-2)", "Escalado a Layer 3"]
    colors = ["#2ecc71", "#e74c3c"]

    axes[0].pie(
        layer3_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05),
    )
    axes[0].set_title("Distribución de Uso de Layer 3")

    # Comparación de tiempos
    df_no_layer3 = df[df["mace_layer3_used"] == 0]
    df_with_layer3 = df[df["mace_layer3_used"] == 1]

    if len(df_no_layer3) > 0 and len(df_with_layer3) > 0:
        data_to_plot = [
            df_no_layer3["mace_time"].dropna(),
            df_with_layer3["mace_time"].dropna(),
        ]
        labels_box = ["Sin Layer 3\n(rápido)", "Con Layer 3\n(profundo)"]

        bp = axes[1].boxplot(
            data_to_plot, tick_labels=labels_box, patch_artist=True, showmeans=True
        )

        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")

        axes[1].set_title("Comparación de Tiempos de Evaluación")
        axes[1].set_ylabel("Tiempo (segundos)")
        axes[1].grid(True, axis="y", alpha=0.3)

        # Añadir promedios
        avg_no_l3 = df_no_layer3["mace_time"].mean()
        avg_with_l3 = df_with_layer3["mace_time"].mean()
        axes[1].text(
            0.5,
            0.95,
            f"Sin L3: {avg_no_l3:.3f}s | Con L3: {avg_with_l3:.3f}s",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Datos insuficientes para comparar tiempos",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hybrid_layer3_usage.png", dpi=300, bbox_inches="tight")
    print(f"Guardado: hybrid_layer3_usage.png")
    plt.close()


def plot_hybrid_by_difficulty():
    """Scores MACE por nivel de dificultad"""
    df, _ = load_hybrid_data()

    if df.empty or "difficulty" not in df.columns:
        print("No hay datos de difficulty, omitiendo gráfica.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    difficulty_order = ["Easy", "Medium", "Hard", "Very Hard"]

    # Agrupar por dificultad
    df_grouped = df.groupby("difficulty").agg(
        {"mace_score": ["mean", "std", "count"], "mace_layer3_used": "sum"}
    )

    df_grouped = df_grouped.reindex(difficulty_order)

    # Bar plot de scores
    x = np.arange(len(difficulty_order))
    bars = ax.bar(
        x,
        df_grouped[("mace_score", "mean")],
        yerr=df_grouped[("mace_score", "std")],
        capsize=5,
        alpha=0.7,
        color="#9b59b6",
        edgecolor="black",
    )

    # Añadir n y % Layer 3
    for i, (bar, diff) in enumerate(zip(bars, difficulty_order)):
        if diff in df_grouped.index:
            count = int(df_grouped.loc[diff, ("mace_score", "count")])
            layer3_count = int(df_grouped.loc[diff, ("mace_layer3_used", "sum")])
            layer3_pct = (layer3_count / count * 100) if count > 0 else 0

            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"n={count}\nL3: {layer3_pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(
        "MACE Score por Dificultad (con % de uso de Layer 3)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Score (0-1)")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_order)
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hybrid_by_difficulty.png", dpi=300, bbox_inches="tight")
    print(f"Guardado: hybrid_by_difficulty.png")
    plt.close()


def plot_quality_confidence_distribution():
    """Distribución de quality labels y confidence levels"""
    df, _ = load_hybrid_data()

    if df.empty:
        print("No hay datos MACE, omitiendo gráfica.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Quality labels
    if "mace_quality" in df.columns:
        quality_counts = df["mace_quality"].value_counts()
        colors_quality = {
            "Excelente": "#2ecc71",
            "Bueno": "#3498db",
            "Mejorable": "#f39c12",
            "Crítico": "#e74c3c",
        }

        colors = [colors_quality.get(q, "#95a5a6") for q in quality_counts.index]

        axes[0].barh(quality_counts.index, quality_counts.values, color=colors)
        axes[0].set_title("Distribución de Quality Labels")
        axes[0].set_xlabel("Frecuencia")
        axes[0].grid(True, axis="x", alpha=0.3)

        # Añadir porcentajes
        for i, (label, count) in enumerate(quality_counts.items()):
            pct = (count / len(df)) * 100
            axes[0].text(count + 0.5, i, f"{pct:.1f}%", va="center")
    else:
        axes[0].text(
            0.5, 0.5, "Columna mace_quality no disponible", ha="center", va="center"
        )
        axes[0].axis("off")

    # Confidence levels
    if "mace_confidence" in df.columns:
        confidence_counts = df["mace_confidence"].value_counts()
        colors_conf = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}

        colors = [colors_conf.get(c, "#95a5a6") for c in confidence_counts.index]

        axes[1].bar(
            confidence_counts.index,
            confidence_counts.values,
            color=colors,
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_title("Distribución de Confidence Levels")
        axes[1].set_ylabel("Frecuencia")
        axes[1].grid(True, axis="y", alpha=0.3)

        # Añadir porcentajes
        for i, (_, count) in enumerate(confidence_counts.items()):
            pct = (count / len(df)) * 100
            axes[1].text(i, count + 0.5, f"{pct:.1f}%", ha="center", va="bottom")
    else:
        axes[1].text(
            0.5, 0.5, "Columna mace_confidence no disponible", ha="center", va="center"
        )
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}hybrid_quality_confidence.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: hybrid_quality_confidence.png")
    plt.close()


def plot_layer_correlation():
    """Correlación entre las 3 capas"""
    df, _ = load_hybrid_data()

    if df.empty:
        print("No hay datos MACE, omitiendo gráfica.")
        return

    # Solo casos con Layer 3
    df_with_l3 = df[df["mace_layer3"].notna()].copy()

    if len(df_with_l3) < 5:
        print("Muy pocos casos con Layer 3, omitiendo gráfica de correlación.")
        return

    layers_data = df_with_l3[["mace_layer1", "mace_layer2", "mace_layer3"]].dropna()

    if layers_data.empty:
        print("No hay datos completos de las 3 capas.")
        return

    corr_matrix = layers_data.corr()
    corr_matrix.columns = [
        "Layer 1\n(Guardrails)",
        "Layer 2\n(Semantic)",
        "Layer 3\n(LLM)",
    ]
    corr_matrix.index = [
        "Layer 1\n(Guardrails)",
        "Layer 2\n(Semantic)",
        "Layer 3\n(LLM)",
    ]

    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=2,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title(
        f"Correlación entre Capas (n={len(df_with_l3)} casos con Layer 3)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}hybrid_layer_correlation.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: hybrid_layer_correlation.png")
    plt.close()


def generate_all_hybrid_plots():
    """Generar todas las visualizaciones de MACE"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERANDO GRÁFICAS MACE (HYBRID EVALUATION)")
    print("=" * 70 + "\n")

    plot_hybrid_layers_distribution()
    plot_layer3_usage()
    plot_hybrid_by_difficulty()
    plot_quality_confidence_distribution()
    plot_layer_correlation()

    print("\n" + "=" * 70)
    print("TODAS LAS GRÁFICAS MACE GENERADAS")
    print(f"Ubicación: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_all_hybrid_plots()
