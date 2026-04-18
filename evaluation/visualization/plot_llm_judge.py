import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")

LLM_JUDGE_RESULTS_LEGACY = "evaluation/llm_j/dataset_llmj_results.csv"
ACCUMULATED_DATA = "evaluation/accumulated_data/offline_metrics.csv"
OUTPUT_DIR = "evaluation/visualization/plots/"


def load_llm_judge_data():
    """
    Carga datos de LLM-Judge desde accumulated_data o CSV legacy.
    Escala 1-4
    """
    accumulated_path = Path(ACCUMULATED_DATA)

    if accumulated_path.exists():
        print(f"Cargando LLM-Judge desde: {ACCUMULATED_DATA}")
        df = pd.read_csv(ACCUMULATED_DATA)

        # Filtrar solo offline y con LLM-Judge data
        df = df[(df["source"] == "offline") & (df["llm_judge_overall"].notna())].copy()

        # NO normalizar: escala 1-4 nativa
        print(f"Cargados {len(df)} casos desde accumulated_data (escala 1-4)")
        return df, 4  # Devuelve escala máxima

    else:
        print(f"No se encontró {ACCUMULATED_DATA}, usando CSV legacy...")
        df = pd.read_csv(LLM_JUDGE_RESULTS_LEGACY)

        # Legacy: escala 0-10
        print(f"Cargados {len(df)} casos desde CSV legacy (escala 0-10)")
        return df, 10  # Devuelve escala máxima


def plot_llm_judge_modules():
    """Scores por módulo (Planner, Supervisor, Agents, Final)"""
    df, max_scale = load_llm_judge_data()

    _, ax = plt.subplots(figsize=(12, 6))

    if max_scale == 4:
        # Accumulated data (escala 1-4)
        modules = [
            "llm_judge_planner",
            "llm_judge_supervisor",
            "llm_judge_agents",
            "llm_judge_final",
        ]
        module_names = ["Planner", "Supervisor", "Agentes", "Output Final"]
        ylabel = "Score (1-4)"
        ylim = [0, 4.5]
    else:
        # Legacy (escala 0-10)
        modules = [
            "planner_score",
            "supervisor_score",
            "agents_avg_score",
            "final_output_score",
        ]
        module_names = ["Planner", "Supervisor", "Agentes", "Output Final"]
        ylabel = "Score (0-10)"
        ylim = [0, 10.5]

    # Filtrar columnas existentes
    available_modules = [col for col in modules if col in df.columns]
    available_names = [module_names[modules.index(col)] for col in available_modules]

    if len(available_modules) == 0:
        print("No se encontraron columnas de módulos, omitiendo gráfica.")
        return

    data = [df[col].dropna() for col in available_modules]

    bp = ax.boxplot(
        data, tick_labels=available_names, patch_artist=True, showmeans=True
    )

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    for patch, color in zip(bp["boxes"], colors[: len(available_modules)]):
        patch.set_facecolor(color)

    ax.set_title(
        f"Distribución de Scores LLM-Judge por Módulo (Escala {max_scale})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}llm_judge_modules_boxplot.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: llm_judge_modules_boxplot.png")
    plt.close()


def plot_error_categories():
    """Distribución de categorías de error"""
    df, _ = load_llm_judge_data()

    error_col = (
        "llm_judge_error_category"
        if "llm_judge_error_category" in df.columns
        else "error_category"
    )

    if error_col not in df.columns:
        print("Columna de error_category no disponible, omitiendo gráfica.")
        return

    error_counts = df[error_col].value_counts()

    _, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(error_counts)))
    bars = ax.barh(error_counts.index, error_counts.values, color=colors)

    ax.set_title(
        "Distribución de Categorías de Error (LLM-Judge)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Frecuencia")
    ax.grid(True, axis="x", alpha=0.3)

    # Añadir porcentajes
    for _, (bar, count) in enumerate(zip(bars, error_counts.values)):
        percentage = (count / len(df)) * 100
        ax.text(
            count + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}%",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}llm_judge_error_categories.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: llm_judge_error_categories.png")
    plt.close()


def plot_llm_judge_by_difficulty():
    """Overall score por dificultad"""
    df, max_scale = load_llm_judge_data()

    if "difficulty" not in df.columns:
        print("Columna 'difficulty' no disponible, omitiendo gráfica.")
        return

    overall_col = (
        "llm_judge_overall" if "llm_judge_overall" in df.columns else "overall_score"
    )

    if overall_col not in df.columns:
        print("Columna de overall_score no disponible, omitiendo gráfica.")
        return

    _, ax = plt.subplots(figsize=(10, 6))

    difficulty_order = ["Easy", "Medium", "Hard", "Very Hard"]
    df_plot = df.groupby("difficulty")[overall_col].agg(["mean", "std", "count"])
    df_plot = df_plot.reindex(difficulty_order)

    bars = ax.bar(
        df_plot.index,
        df_plot["mean"],
        yerr=df_plot["std"],
        capsize=5,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )

    # Añadir n
    for bar, count in zip(bars, df_plot["count"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"n={int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ylabel = f"Score (1-4)" if max_scale == 4 else "Score (0-10)"
    ylim = [0, max_scale + 1] if max_scale == 4 else [0, 11]

    ax.set_title(
        f"Overall Score LLM-Judge por Dificultad (Escala {max_scale})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}llm_judge_by_difficulty.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: llm_judge_by_difficulty.png")
    plt.close()


def generate_all_llm_judge_plots():
    """Generar todas las gráficas de LLM-Judge"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERANDO GRÁFICAS LLM-JUDGE")
    print("=" * 60 + "\n")

    plot_llm_judge_modules()
    plot_error_categories()
    plot_llm_judge_by_difficulty()

    print("\n" + "=" * 60)
    print("TODAS LAS GRÁFICAS LLM-JUDGE GENERADAS")
    print(f"Ubicación: {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    generate_all_llm_judge_plots()
