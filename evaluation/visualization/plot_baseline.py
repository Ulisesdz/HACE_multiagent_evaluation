import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

BASELINE_RESULTS_LEGACY = "evaluation/baseline/dataset_baseline_results.csv"
ACCUMULATED_DATA = "evaluation/accumulated_data/offline_metrics.csv"
OUTPUT_DIR = "evaluation/visualization/plots/"


def load_baseline_data():
    """
    Carga datos de Baseline desde accumulated_data o CSV legacy.
    """
    accumulated_path = Path(ACCUMULATED_DATA)

    if accumulated_path.exists():
        print(f"Cargando Baseline desde: {ACCUMULATED_DATA}")
        df = pd.read_csv(ACCUMULATED_DATA)

        # Filtrar solo offline y con baseline data
        df = df[(df["source"] == "offline") & (df["baseline_score"].notna())].copy()

        print(f"Cargados {len(df)} casos desde accumulated_data")
        return df

    else:
        print(f"No se encontró {ACCUMULATED_DATA}, usando CSV legacy...")
        df = pd.read_csv(BASELINE_RESULTS_LEGACY)
        print(f"Cargados {len(df)} casos desde CSV legacy")
        return df


def plot_baseline_metrics_distribution():
    """Distribución de las 4 métricas baseline"""
    df = load_baseline_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribución de Métricas Baseline", fontsize=16, fontweight="bold")

    metrics = [
        ("baseline_routing_f1", "Routing F1-Score", axes[0, 0]),
        ("baseline_numeric_f1", "Numeric F1-Score", axes[0, 1]),
        ("baseline_task_coverage", "Task Coverage", axes[1, 0]),
        ("baseline_sql_correctness", "SQL Correctness", axes[1, 1]),
    ]

    for col, title, ax in metrics:
        if col in df.columns:
            data = df[col].dropna()
            ax.hist(data, bins=20, edgecolor="black", alpha=0.7)
            ax.axvline(
                data.mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Media: {data.mean():.3f}",
            )
            ax.set_title(title)
            ax.set_xlabel("Score (0-1)")
            ax.set_ylabel("Frecuencia")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                f"Columna {col} no disponible",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}baseline_metrics_distribution.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: baseline_metrics_distribution.png")
    plt.close()


def plot_baseline_by_difficulty():
    """Scores por nivel de dificultad"""
    df = load_baseline_data()

    if "difficulty" not in df.columns:
        print("Columna 'difficulty' no disponible, omitiendo gráfica.")
        return

    _, ax = plt.subplots(figsize=(12, 6))

    difficulty_order = ["Easy", "Medium", "Hard", "Very Hard"]

    # Renombrar columnas si vienen de accumulated_data
    columns_to_plot = {
        "baseline_score": "Baseline Score",
        "baseline_routing_f1": "Routing F1",
        "baseline_numeric_f1": "Numeric F1",
        "baseline_task_coverage": "Task Coverage",
    }

    # Seleccionar columnas existentes
    available_cols = {k: v for k, v in columns_to_plot.items() if k in df.columns}

    df_plot = df.groupby("difficulty")[list(available_cols.keys())].mean()
    df_plot = df_plot.reindex(difficulty_order)
    df_plot.columns = [available_cols[col] for col in df_plot.columns]

    df_plot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Desempeño Baseline por Dificultad", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dificultad")
    ax.set_ylabel("Score Promedio")
    ax.set_xticklabels(df_plot.index, rotation=45)
    ax.legend(title="Métricas", loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}baseline_by_difficulty.png", dpi=300, bbox_inches="tight")
    print(f"Guardado: baseline_by_difficulty.png")
    plt.close()


def plot_hallucination_analysis():
    """Análisis detallado de alucinaciones"""
    df = load_baseline_data()

    halluc_col = (
        "baseline_hallucination_rate"
        if "baseline_hallucination_rate" in df.columns
        else "hallucination_rate"
    )

    if halluc_col not in df.columns:
        print("Columna de hallucination_rate no disponible, omitiendo gráfica.")
        return

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma de hallucination rate
    axes[0].hist(df[halluc_col], bins=20, edgecolor="black", alpha=0.7, color="coral")
    axes[0].axvline(
        df[halluc_col].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Media: {df[halluc_col].mean():.1%}",
    )
    axes[0].set_title("Distribución de Alucinaciones Numéricas")
    axes[0].set_xlabel("Hallucination Rate")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Casos con alucinaciones por categoría
    if "category" in df.columns:
        df_halluc = (
            df[df[halluc_col] > 0.1]
            .groupby("category")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        if len(df_halluc) > 0:
            axes[1].barh(df_halluc.index, df_halluc.values, color="coral")
            axes[1].set_title("Categorías con Mayor Tasa de Alucinaciones (>10%)")
            axes[1].set_xlabel("Número de Casos")
            axes[1].grid(True, axis="x", alpha=0.3)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No hay casos con >10% alucinaciones",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[1].axis("off")
    else:
        axes[1].text(
            0.5,
            0.5,
            'Columna "category" no disponible',
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}baseline_hallucination_analysis.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: baseline_hallucination_analysis.png")
    plt.close()


def plot_sql_correctness_breakdown():
    """Desglose de SQL correctness"""
    df = load_baseline_data()

    sql_correct_col = (
        "baseline_sql_correctness"
        if "baseline_sql_correctness" in df.columns
        else "sql_correctness"
    )

    # Filtrar solo casos con SQL queries
    if "sql_total_queries" in df.columns:
        df_sql = df[df["sql_total_queries"] > 0].copy()
    elif "baseline_sql_total_queries" in df.columns:
        df_sql = df[df["baseline_sql_total_queries"] > 0].copy()
    else:
        print("No se encontraron columnas de SQL, omitiendo gráfica.")
        return

    if len(df_sql) == 0:
        print("No hay casos con SQL queries en el dataset")
        return

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribución de correctness
    if sql_correct_col in df_sql.columns:
        axes[0].hist(
            df_sql[sql_correct_col],
            bins=10,
            edgecolor="black",
            alpha=0.7,
            color="lightblue",
        )
        axes[0].axvline(
            df_sql[sql_correct_col].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Media: {df_sql[sql_correct_col].mean():.1%}",
        )
        axes[0].set_title("Distribución de SQL Correctness")
        axes[0].set_xlabel("Correctness (0-1)")
        axes[0].set_ylabel("Frecuencia")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Columna {sql_correct_col} no disponible",
            ha="center",
            va="center",
        )
        axes[0].axis("off")

    # Top violaciones
    if "sql_violations" in df_sql.columns:
        violations_all = []
        for violations_str in df_sql["sql_violations"].dropna():
            if violations_str:
                violations_all.extend(violations_str.split("; "))

        if violations_all:
            from collections import Counter

            top_violations = Counter(violations_all).most_common(5)
            names, counts = zip(*top_violations)

            axes[1].barh(names, counts, color="lightcoral")
            axes[1].set_title("Top 5 Violaciones SQL")
            axes[1].set_xlabel("Frecuencia")
            axes[1].grid(True, axis="x", alpha=0.3)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No hay violaciones SQL",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[1].axis("off")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Datos de violaciones no disponibles\n(solo en CSV legacy)",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}baseline_sql_correctness.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: baseline_sql_correctness.png")
    plt.close()


def plot_baseline_heatmap():
    """Heatmap de correlación entre métricas"""
    df = load_baseline_data()

    # Mapear columnas según fuente
    metrics_map = {
        "baseline_score": "Baseline Score",
        "baseline_routing_f1": "Routing F1",
        "baseline_numeric_f1": "Numeric F1",
        "baseline_task_coverage": "Task Coverage",
        "baseline_sql_correctness": "SQL Correctness",
        "baseline_hallucination_rate": "Hallucination Rate",
    }

    # Legacy fallback
    if "routing_f1" in df.columns:
        metrics_map = {
            "baseline_score": "Baseline Score",
            "routing_f1": "Routing F1",
            "numeric_f1": "Numeric F1",
            "task_coverage": "Task Coverage",
            "sql_correctness": "SQL Correctness",
            "hallucination_rate": "Hallucination Rate",
        }

    # Seleccionar columnas existentes
    available_cols = [col for col in metrics_map.keys() if col in df.columns]

    if len(available_cols) < 2:
        print("No hay suficientes métricas para generar heatmap, omitiendo gráfica.")
        return

    corr_matrix = df[available_cols].corr()
    corr_matrix.columns = [metrics_map[col] for col in corr_matrix.columns]
    corr_matrix.index = [metrics_map[col] for col in corr_matrix.index]

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlación entre Métricas Baseline", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}baseline_correlation_heatmap.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: baseline_correlation_heatmap.png")
    plt.close()


def generate_all_baseline_plots():
    """Generar todas las visualizaciones de Baseline"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERANDO GRÁFICAS BASELINE")
    print("=" * 60 + "\n")

    plot_baseline_metrics_distribution()
    plot_baseline_by_difficulty()
    plot_hallucination_analysis()
    plot_sql_correctness_breakdown()
    plot_baseline_heatmap()

    print("\n" + "=" * 60)
    print("TODAS LAS GRÁFICAS BASELINE GENERADAS")
    print(f"Ubicación: {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    generate_all_baseline_plots()
