import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

plt.style.use("seaborn-v0_8-darkgrid")

# Paths
BASELINE_RESULTS_LEGACY = "evaluation/baseline/dataset_baseline_results.csv"
LLM_JUDGE_RESULTS_LEGACY = "evaluation/llm_j/dataset_llmj_results.csv"
HYBRID_RESULTS_LEGACY = "evaluation/hybrid/dataset_hybrid_results.csv"
ACCUMULATED_DATA = "evaluation/accumulated_data/offline_metrics.csv"
ONLINE_DATA = "evaluation/accumulated_data/online_metrics.csv"
OUTPUT_DIR = "evaluation/visualization/plots/"


def load_comparison_data_all():
    """
    Carga datos de todos los métodos desde accumulated_data o CSVs legacy.
    """
    accumulated_path = Path(ACCUMULATED_DATA)

    if accumulated_path.exists():
        print(f"Cargando datos comparativos desde: {ACCUMULATED_DATA}")
        df_raw = pd.read_csv(ACCUMULATED_DATA)

        # Filtrar solo offline
        df_raw = df_raw[df_raw["source"] == "offline"].copy()

        print(f"Total de registros offline: {len(df_raw)}")

        # Convertir columnas numéricas
        numeric_cols = [
            "baseline_score",
            "baseline_time",
            "llm_judge_overall",
            "llm_judge_time",
            "HACE_score",
            "HACE_time",
            "HACE_layer3_used",
        ]

        for col in numeric_cols:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

        # Colapsa las múltiples filas del mismo TC-001 en una sola,
        # cogiendo los datos válidos (no nulos) de cada evaluador.
        # df_raw = df_raw.groupby(["query_id", "difficulty"], as_index=False).last()

        print(f"Casos únicos tras fusionar (deberían ser 45): {len(df_raw)}")

        # Crear DataFrame unificado
        df_merged = df_raw[["query_id", "difficulty"]].copy()

        # 1. Baseline
        has_baseline = df_raw["baseline_score"].notna()
        if has_baseline.any():
            df_merged.loc[has_baseline, "baseline_score"] = df_raw.loc[
                has_baseline, "baseline_score"
            ]
            df_merged.loc[has_baseline, "baseline_eval_time"] = df_raw.loc[
                has_baseline, "baseline_time"
            ]
            print(f"   Baseline: {has_baseline.sum()} casos")
        else:
            print(f"   Baseline: 0 casos")

        # 2. LLM-Judge
        has_llm = df_raw["llm_judge_overall"].notna()
        if has_llm.any():
            df_merged.loc[has_llm, "llm_judge_overall"] = df_raw.loc[
                has_llm, "llm_judge_overall"
            ]
            df_merged.loc[has_llm, "llm_judge_eval_time"] = df_raw.loc[
                has_llm, "llm_judge_time"
            ]
            # Normalizar 1-4 → 0-1
            df_merged["llm_judge_normalized"] = df_merged["llm_judge_overall"] / 4
            print(f"   LLM-Judge: {has_llm.sum()} casos (escala 1-4)")
        else:
            print(f"   LLM-Judge: 0 casos")

        # 3. HACE
        has_HACE = df_raw["HACE_score"].notna()
        if has_HACE.any():
            df_merged.loc[has_HACE, "HACE_score"] = df_raw.loc[has_HACE, "HACE_score"]
            df_merged.loc[has_HACE, "HACE_eval_time"] = df_raw.loc[
                has_HACE, "HACE_time"
            ]
            df_merged.loc[has_HACE, "HACE_layer3_used"] = df_raw.loc[
                has_HACE, "HACE_layer3_used"
            ]
            print(f"   HACE: {has_HACE.sum()} casos")
        else:
            print(f"   HACE: 0 casos")

        # Contar casos coincidentes
        all_three = has_baseline & has_llm & has_HACE
        print(f"\nCasos con los 3 métodos: {all_three.sum()}")

        baseline_and_llm = has_baseline & has_llm
        baseline_and_HACE = has_baseline & has_HACE
        llm_and_HACE = has_llm & has_HACE

        print(f"   • Baseline + LLM-Judge: {baseline_and_llm.sum()}")
        print(f"   • Baseline + HACE: {baseline_and_HACE.sum()}")
        print(f"   • LLM-Judge + HACE: {llm_and_HACE.sum()}")

        return df_merged, "accumulated", 4

    else:
        print(f"No se encontró {ACCUMULATED_DATA}, usando CSVs legacy...")

        # Intentar cargar desde CSVs individuales
        baseline_path = Path(BASELINE_RESULTS_LEGACY)
        llm_path = Path(LLM_JUDGE_RESULTS_LEGACY)
        HACE_path = Path(HYBRID_RESULTS_LEGACY)

        # Baseline
        if baseline_path.exists():
            df_baseline = pd.read_csv(baseline_path)
            df_baseline = df_baseline.rename(
                columns={"id": "query_id", "evaluation_time": "baseline_eval_time"}
            )
            print(f"   Baseline legacy: {len(df_baseline)} casos")
        else:
            df_baseline = pd.DataFrame()
            print(f"   Baseline legacy: No encontrado")

        # LLM-Judge
        if llm_path.exists():
            df_llm = pd.read_csv(llm_path)
            df_llm = df_llm.rename(columns={"id": "query_id"})
            # Normalizar si es escala 0-10
            if "overall_score" in df_llm.columns:
                max_score = df_llm["overall_score"].max()
                if max_score > 4:
                    df_llm["llm_judge_normalized"] = df_llm["overall_score"] / 10
                    print(f"   LLM-Judge legacy: {len(df_llm)} casos (escala 0-10)")
                else:
                    df_llm["llm_judge_normalized"] = df_llm["overall_score"] / 4
                    print(f"   LLM-Judge legacy: {len(df_llm)} casos (escala 1-4)")

                df_llm["llm_judge_overall"] = df_llm["overall_score"]
        else:
            df_llm = pd.DataFrame()
            print(f"   LLM-Judge legacy: No encontrado")

        # HACE
        if HACE_path.exists():
            df_HACE = pd.read_csv(HACE_path)
            df_HACE = df_HACE.rename(
                columns={
                    "id": "query_id",
                    "hybrid_score": "HACE_score",
                    "evaluation_time": "HACE_eval_time",
                }
            )
            print(f"   HACE legacy: {len(df_HACE)} casos")
        else:
            df_HACE = pd.DataFrame()
            print(f"   HACE legacy: No encontrado")

        # Merge incremental
        if not df_baseline.empty:
            df_merged = df_baseline[
                ["query_id", "baseline_score", "baseline_eval_time", "difficulty"]
            ].copy()
        elif not df_llm.empty:
            df_merged = df_llm[["query_id", "difficulty"]].copy()
        elif not df_HACE.empty:
            df_merged = df_HACE[["query_id", "difficulty"]].copy()
        else:
            print("No se encontraron datos en ningún CSV")
            return pd.DataFrame(), "none", 4

        # Merge LLM-Judge
        if not df_llm.empty and not df_merged.empty:
            df_merged = pd.merge(
                df_merged,
                df_llm[["query_id", "llm_judge_overall", "llm_judge_normalized"]],
                on="query_id",
                how="outer",
            )

        # Merge HACE
        if not df_HACE.empty and not df_merged.empty:
            df_merged = pd.merge(
                df_merged,
                df_HACE[["query_id", "HACE_score", "HACE_eval_time", "layer3_used"]],
                on="query_id",
                how="outer",
            )
            df_merged = df_merged.rename(columns={"layer3_used": "HACE_layer3_used"})

        print(f"\nTotal merged: {len(df_merged)} casos")

        return df_merged, "legacy", 4


def plot_score_comparison_triple():
    """Comparación directa de scores de los 3 métodos"""
    df, _, _ = load_comparison_data_all()

    if len(df) == 0:
        print("No hay datos para comparar.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogramas
    plotted_any = False

    if "baseline_score" in df.columns and df["baseline_score"].notna().any():
        axes[0].hist(
            df["baseline_score"].dropna(),
            bins=20,
            alpha=0.5,
            label=f'Baseline (n={df["baseline_score"].notna().sum()})',
            color="steelblue",
            edgecolor="black",
        )
        plotted_any = True

    if (
        "llm_judge_normalized" in df.columns
        and df["llm_judge_normalized"].notna().any()
    ):
        axes[0].hist(
            df["llm_judge_normalized"].dropna(),
            bins=20,
            alpha=0.5,
            label=f'LLM-Judge (n={df["llm_judge_normalized"].notna().sum()})',
            color="coral",
            edgecolor="black",
        )
        plotted_any = True

    if "HACE_score" in df.columns and df["HACE_score"].notna().any():
        axes[0].hist(
            df["HACE_score"].dropna(),
            bins=20,
            alpha=0.5,
            label=f'HACE (n={df["HACE_score"].notna().sum()})',
            color="mediumpurple",
            edgecolor="black",
        )
        plotted_any = True

    if plotted_any:
        axes[0].set_title(
            "Distribución de Scores Globales (Normalizados a 0-1)", fontweight="bold"
        )
        axes[0].set_xlabel("Score (0-1)")
        axes[0].set_ylabel("Frecuencia")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No hay datos de scores disponibles",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].axis("off")

    # Boxplot comparativo
    data_to_plot = []
    labels_plot = []

    if "baseline_score" in df.columns and df["baseline_score"].notna().any():
        data_to_plot.append(df["baseline_score"].dropna())
        labels_plot.append("Baseline")

    if (
        "llm_judge_normalized" in df.columns
        and df["llm_judge_normalized"].notna().any()
    ):
        data_to_plot.append(df["llm_judge_normalized"].dropna())
        labels_plot.append("LLM-Judge")

    if "HACE_score" in df.columns and df["HACE_score"].notna().any():
        data_to_plot.append(df["HACE_score"].dropna())
        labels_plot.append("HACE")

    if len(data_to_plot) >= 2:
        bp = axes[1].boxplot(
            data_to_plot, tick_labels=labels_plot, patch_artist=True, showmeans=True
        )

        colors = ["steelblue", "coral", "mediumpurple"]
        for patch, color in zip(bp["boxes"], colors[: len(data_to_plot)]):
            patch.set_facecolor(color)

        axes[1].set_title("Comparación de Scores (Boxplot)", fontweight="bold")
        axes[1].set_ylabel("Score Normalizado (0-1)")
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Se necesitan al menos 2 métodos para comparar",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}comparison_scores_triple.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: comparison_scores_triple.png")
    plt.close()


def plot_time_comparison_triple():
    """Comparación de tiempos de los 3 métodos"""
    df, _, _ = load_comparison_data_all()

    if len(df) == 0:
        print("No hay datos para comparar.")
        return

    # Calcular tiempos promedio
    times = {}

    if "baseline_eval_time" in df.columns and df["baseline_eval_time"].notna().any():
        times["Baseline"] = df["baseline_eval_time"].mean()
        print(f"   Baseline time: {times['Baseline']:.3f}s")
    else:
        times["Baseline"] = 0.023
        print(f"   Baseline time: ~0.023s (estimado)")

    if "llm_judge_eval_time" in df.columns and df["llm_judge_eval_time"].notna().any():
        times["LLM-Judge"] = df["llm_judge_eval_time"].mean()
        print(f"   LLM-Judge time: {times['LLM-Judge']:.3f}s")
    else:
        times["LLM-Judge"] = 180.5
        print(f"   LLM-Judge time: ~180.5s (estimado)")

    if "HACE_eval_time" in df.columns and df["HACE_eval_time"].notna().any():
        times["HACE"] = df["HACE_eval_time"].mean()
        print(f"   HACE time: {times['HACE']:.3f}s")
    else:
        times["HACE"] = 1.9
        print(f"   HACE time: ~1.9s (estimado)")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Labels SIN saltos de línea
    methods = ["Baseline\n(Automático)", "LLM-Judge\n(Cualitativo)", "HACE\n(Híbrido)"]
    values = [times["Baseline"], times["LLM-Judge"], times["HACE"]]
    colors = ["steelblue", "coral", "mediumpurple"]

    bars = ax.bar(methods, values, color=colors, edgecolor="black", alpha=0.7)

    # Añadir valores
    for bar, time in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f"{time:.3f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title(
        "Comparación de Tiempos de Evaluación (3 Métodos)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_ylim([0, max(values) * 1.2])
    ax.grid(True, axis="y", alpha=0.3)

    # Añadir speedup comparativo
    if times["HACE"] and times["LLM-Judge"]:
        HACE_time = times["HACE"]
        llm_time = times["LLM-Judge"]
        speedup = ((llm_time - HACE_time) / llm_time) * 100

        ax.text(
            0.5,
            0.95,
            f"HACE es {speedup:.0f}% más rápido que LLM-Judge",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}comparison_time_triple.png", dpi=300, bbox_inches="tight")
    print(f"Guardado: comparison_time_triple.png")
    plt.close()


def plot_difficulty_comparison_triple():
    """Comparación por dificultad de los 3 métodos"""
    df, source, llm_scale = load_comparison_data_all()

    if len(df) == 0 or "difficulty" not in df.columns:
        print("No hay datos de difficulty.")
        return

    difficulty_order = ["Easy", "Medium", "Hard", "Very Hard"]

    # Agrupar por dificultad
    grouped_data = {}

    if "baseline_score" in df.columns and df["baseline_score"].notna().any():
        grouped_data["Baseline"] = (
            df.groupby("difficulty")["baseline_score"].mean().reindex(difficulty_order)
        )
        print(
            f"   Baseline por difficulty: {grouped_data['Baseline'].notna().sum()} niveles"
        )

    if (
        "llm_judge_normalized" in df.columns
        and df["llm_judge_normalized"].notna().any()
    ):
        grouped_data["LLM-Judge"] = (
            df.groupby("difficulty")["llm_judge_normalized"]
            .mean()
            .reindex(difficulty_order)
        )
        print(
            f"   LLM-Judge por difficulty: {grouped_data['LLM-Judge'].notna().sum()} niveles"
        )

    if "HACE_score" in df.columns and df["HACE_score"].notna().any():
        grouped_data["HACE"] = (
            df.groupby("difficulty")["HACE_score"].mean().reindex(difficulty_order)
        )
        print(f"   HACE por difficulty: {grouped_data['HACE'].notna().sum()} niveles")

    if not grouped_data:
        print("No hay datos válidos para gráfica de dificultad.")
        return

    _, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(difficulty_order))
    width = 0.25

    colors = {"Baseline": "steelblue", "LLM-Judge": "coral", "HACE": "mediumpurple"}

    for i, (method, data) in enumerate(grouped_data.items()):
        offset = (i - len(grouped_data) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            data,
            width,
            label=method,
            color=colors.get(method, "gray"),
            edgecolor="black",
            alpha=0.7,
        )

        # Añadir valores
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(
        "Desempeño por Dificultad: Comparación de 3 Métodos (Normalizados a 0-1)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Score Normalizado (0-1)")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_order)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}comparison_by_difficulty_triple.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: comparison_by_difficulty_triple.png")
    plt.close()


def plot_summary_table_triple():
    """Tabla resumen comparativa de los 3 métodos"""
    df, _, llm_scale = load_comparison_data_all()

    if len(df) == 0:
        print("No hay datos para tabla.")
        return

    # Calcular stats
    summary_data = {
        "Método": [],
        "Casos": [],
        "Score Promedio": [],
        "Desv. Estándar": [],
        "Tiempo Promedio": [],
        "Características": [],
    }

    # Baseline
    if "baseline_score" in df.columns and df["baseline_score"].notna().any():
        baseline_data = df["baseline_score"].dropna()
        summary_data["Método"].append("Baseline")
        summary_data["Casos"].append(f"{len(baseline_data)}")
        summary_data["Score Promedio"].append(f"{baseline_data.mean():.3f}")
        summary_data["Desv. Estándar"].append(f"{baseline_data.std():.3f}")

        if (
            "baseline_eval_time" in df.columns
            and df["baseline_eval_time"].notna().any()
        ):
            summary_data["Tiempo Promedio"].append(
                f"{df['baseline_eval_time'].mean():.3f}s"
            )
        else:
            summary_data["Tiempo Promedio"].append("~0.023s")
        summary_data["Características"].append("Determinista, numérico/SQL")

    # LLM-Judge
    if (
        "llm_judge_normalized" in df.columns
        and df["llm_judge_normalized"].notna().any()
    ):
        llm_data = df["llm_judge_normalized"].dropna()
        llm_mean = llm_data.mean() * llm_scale
        llm_std = llm_data.std() * llm_scale

        summary_data["Método"].append("LLM-Judge")
        summary_data["Casos"].append(f"{len(llm_data)}")
        summary_data["Score Promedio"].append(f"{llm_mean:.2f}/{llm_scale}")
        summary_data["Desv. Estándar"].append(f"{llm_std:.2f}")

        if (
            "llm_judge_eval_time" in df.columns
            and df["llm_judge_eval_time"].notna().any()
        ):
            summary_data["Tiempo Promedio"].append(
                f"{df['llm_judge_eval_time'].mean():.2f}s"
            )
        else:
            summary_data["Tiempo Promedio"].append("~180.5s")
        summary_data["Características"].append("Cualitativo, semántico")

    # HACE
    if "HACE_score" in df.columns and df["HACE_score"].notna().any():
        HACE_data = df["HACE_score"].dropna()
        summary_data["Método"].append("HACE")
        summary_data["Casos"].append(f"{len(HACE_data)}")
        summary_data["Score Promedio"].append(f"{HACE_data.mean():.3f}")
        summary_data["Desv. Estándar"].append(f"{HACE_data.std():.3f}")

        if "HACE_eval_time" in df.columns and df["HACE_eval_time"].notna().any():
            summary_data["Tiempo Promedio"].append(
                f"{df['HACE_eval_time'].mean():.3f}s"
            )
        else:
            summary_data["Tiempo Promedio"].append("~1.9s")

        # Calcular % Layer 3
        if "HACE_layer3_used" in df.columns:
            layer3_count = df["HACE_layer3_used"].notna().sum()
            if layer3_count > 0:
                layer3_pct = (df["HACE_layer3_used"].sum() / layer3_count) * 100
                summary_data["Características"].append(
                    f"Híbrido, Layer 3: {layer3_pct:.0f}%"
                )
            else:
                summary_data["Características"].append("Híbrido (3 capas)")
        else:
            summary_data["Características"].append("Híbrido (3 capas)")

    if not summary_data["Método"]:
        print("No hay datos de ningún método.")
        return

    df_summary = pd.DataFrame(summary_data)

    _, ax = plt.subplots(figsize=(15, 3))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.08, 0.12, 0.12, 0.14, 0.14, 0.28],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Colorear header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor("#4ECDC4")
        table[(0, i)].set_text_props(weight="bold")

    # Colorear filas
    row_colors = ["lightsteelblue", "lightcoral", "plum"]
    for i in range(1, len(df_summary) + 1):
        color = row_colors[(i - 1) % len(row_colors)]
        for j in range(len(df_summary.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title(
        f"Resumen Comparativo: Baseline vs LLM-Judge vs HACE",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.savefig(
        f"{OUTPUT_DIR}comparison_summary_table_triple.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: comparison_summary_table_triple.png")
    plt.close()


def plot_methods_correlation():
    """Calcula y grafica la correlación de Pearson entre los 3 métodos evaluadores."""
    df, _, _ = load_comparison_data_all()

    if len(df) == 0:
        print("No hay datos para correlación.")
        return

    # Mapeo de nombres limpios a las columnas
    cols_map = {
        "Baseline": "baseline_score",
        "LLM-Judge": "llm_judge_normalized",
        "HACE": "HACE_score",
    }

    # Filtrar solo las columnas que realmente existen y tienen datos
    available_cols = {
        name: col
        for name, col in cols_map.items()
        if col in df.columns and df[col].notna().any()
    }

    if len(available_cols) < 2:
        print("No hay suficientes métodos para calcular la correlación.")
        return

    # Crear un sub-dataframe limpio sin NaNs
    df_corr = df[list(available_cols.values())].dropna().copy()

    if len(df_corr) < 5:
        print("Insuficientes datos superpuestos (mínimo 5) para calcular correlación.")
        return

    # Renombrar columnas para que se vean bien en la gráfica
    df_corr.columns = list(available_cols.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Mapa de Calor (Heatmap) de correlación
    corr_matrix = df_corr.corr(method="pearson")
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=2,
        cbar_kws={"shrink": 0.8},
        ax=axes[0],
        vmin=-1,
        vmax=1,
    )
    axes[0].set_title(
        f"Matriz de Correlación (n={len(df_corr)})", fontweight="bold", fontsize=13
    )

    # 2. Scatter plot (HACE vs LLM-Judge) si ambos existen
    if "LLM-Judge" in df_corr.columns and "HACE" in df_corr.columns:
        sns.regplot(
            x="LLM-Judge",
            y="HACE",
            data=df_corr,
            ax=axes[1],
            scatter_kws={
                "alpha": 0.7,
                "color": "mediumpurple",
                "s": 60,
                "edgecolor": "white",
            },
            line_kws={"color": "coral", "linestyle": "--", "linewidth": 2.5},
        )

        # Calcular valor exacto de p y r
        r, p_val = pearsonr(df_corr["LLM-Judge"], df_corr["HACE"])
        p_text = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"

        axes[1].set_title(
            "Regresión: LLM-Judge vs HACE", fontweight="bold", fontsize=13
        )
        axes[1].set_xlabel("Score LLM-Judge (Normalizado 0-1)")
        axes[1].set_ylabel("Score HACE (0-1)")

        # Cajita de texto con los resultados
        axes[1].text(
            0.05,
            0.95,
            f"Pearson $r$ = {r:.3f}\n$p$-value {p_text}",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"
            ),
        )
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Faltan datos de LLM-Judge o HACE\npara dibujar el scatter plot.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}comparison_correlation_triple.png", dpi=300, bbox_inches="tight"
    )
    print(f"Guardado: comparison_correlation_triple.png")
    plt.close()


def generate_all_comparison_plots():
    """Generar todas las comparativas (3 métodos)"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERANDO GRÁFICAS COMPARATIVAS (3 MÉTODOS)")
    print("=" * 70 + "\n")

    plot_score_comparison_triple()
    plot_time_comparison_triple()
    plot_difficulty_comparison_triple()
    plot_summary_table_triple()
    plot_methods_correlation()

    print("\n" + "=" * 70)
    print("TODAS LAS GRÁFICAS COMPARATIVAS GENERADAS")
    print(f"Ubicación: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


def generate_online_comparison_plots():
    """Genera comparativas exclusivas para los datos capturados en tiempo real (Streamlit)"""
    online_path = Path(ONLINE_DATA)

    if not online_path.exists():
        print(f"No se encontró el archivo de métricas online: {ONLINE_DATA}")
        return

    print("\n" + "=" * 70)
    print("GENERANDO GRÁFICAS ONLINE (PRODUCCIÓN)")
    print("=" * 70 + "\n")

    df = pd.read_csv(ONLINE_DATA)
    df = df[df["source"] == "online"].copy()

    if len(df) == 0:
        print("No hay registros online guardados aún.")
        return

    print(f"Total de registros online encontrados: {len(df)}")

    # Asegurar tipos numéricos
    cols_to_numeric = [
        "baseline_score",
        "baseline_time",
        "llm_judge_overall",
        "llm_judge_time",
        "HACE_score",
        "HACE_time",
    ]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalizar LLM-Judge (viene de 1-4, lo paso a 0-1 para comparar)
    if "llm_judge_overall" in df.columns:
        df["llm_judge_normalized"] = df["llm_judge_overall"] / 4

    # 1. GRÁFICA DE CAJAS (BOXPLOT) - SCORES ONLINE
    data_to_plot = []
    labels_plot = []

    if "baseline_score" in df.columns and df["baseline_score"].notna().any():
        data_to_plot.append(df["baseline_score"].dropna())
        labels_plot.append("Baseline")

    if (
        "llm_judge_normalized" in df.columns
        and df["llm_judge_normalized"].notna().any()
    ):
        data_to_plot.append(df["llm_judge_normalized"].dropna())
        labels_plot.append("LLM-Judge")

    if "HACE_score" in df.columns and df["HACE_score"].notna().any():
        data_to_plot.append(df["HACE_score"].dropna())
        labels_plot.append("HACE")

    if len(data_to_plot) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(
            data_to_plot, tick_labels=labels_plot, patch_artist=True, showmeans=True
        )
        colors = ["steelblue", "coral", "mediumpurple"]
        for patch, color in zip(bp["boxes"], colors[: len(data_to_plot)]):
            patch.set_facecolor(color)

        ax.set_title(
            "Rendimiento en Producción (Interacciones Online)", fontweight="bold"
        )
        ax.set_ylabel("Score Normalizado (0-1)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}online_comparison_scores.png", dpi=300)
        print("Guardado: online_comparison_scores.png")
        plt.close()

    # 2. GRÁFICA DE TIEMPOS ONLINE
    times = {}
    if "baseline_time" in df.columns:
        times["Baseline"] = df["baseline_time"].mean()
    if "llm_judge_time" in df.columns:
        times["LLM-Judge"] = df["llm_judge_time"].mean()
    if "HACE_time" in df.columns:
        times["HACE"] = df["HACE_time"].mean()

    # Filtrar solo los que no son NaN
    times = {k: v for k, v in times.items() if not np.isnan(v)}

    if len(times) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        methods = list(times.keys())
        values = list(times.values())

        # Mapear colores a los métodos correctos
        color_map = {
            "Baseline": "steelblue",
            "LLM-Judge": "coral",
            "HACE": "mediumpurple",
        }
        bar_colors = [color_map.get(m, "gray") for m in methods]

        bars = ax.bar(methods, values, color=bar_colors, edgecolor="black", alpha=0.7)

        for bar, time_val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max(values) * 0.02),
                f"{time_val:.2f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(
            "Tiempos de Respuesta en Producción", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("Tiempo promedio (segundos)")
        ax.set_ylim([0, max(values) * 1.15])
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}online_comparison_times.png", dpi=300)
        print("Guardado: online_comparison_times.png")
        plt.close()


if __name__ == "__main__":
    # Genera las gráficas del dataset (45 casos)
    generate_all_comparison_plots()

    # Genera las gráficas de uso real en la interfaz web
    generate_online_comparison_plots()
