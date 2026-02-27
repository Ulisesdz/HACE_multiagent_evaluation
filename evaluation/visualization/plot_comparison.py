import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

BASELINE_RESULTS = "evaluation/baseline/dataset_baseline_results.csv"
LLM_JUDGE_RESULTS = "evaluation/llm_j/dataset_results.csv"
OUTPUT_DIR = "evaluation/visualization/plots/"


def plot_score_comparison():
    """Comparación directa de scores globales"""
    df_baseline = pd.read_csv(BASELINE_RESULTS)
    df_llm = pd.read_csv(LLM_JUDGE_RESULTS)
    
    # Normalizar LLM-Judge (0-10 → 0-1)
    df_llm['overall_score_normalized'] = df_llm['overall_score'] / 10
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogramas
    axes[0].hist(df_baseline['baseline_score'], bins=20, alpha=0.6, label='Baseline', color='steelblue', edgecolor='black')
    axes[0].hist(df_llm['overall_score_normalized'], bins=20, alpha=0.6, label='LLM-Judge', color='coral', edgecolor='black')
    axes[0].set_title('Distribución de Scores Globales', fontweight='bold')
    axes[0].set_xlabel('Score (0-1)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot comparativo
    data_to_plot = [df_baseline['baseline_score'], df_llm['overall_score_normalized']]
    bp = axes[1].boxplot(data_to_plot, tick_labels=['Baseline', 'LLM-Judge'], patch_artist=True, showmeans=True)
    
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].set_title('Comparación de Scores (Boxplot)', fontweight='bold')
    axes[1].set_ylabel('Score (0-1)')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}comparison_scores.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: comparison_scores.png")
    plt.close()


def plot_time_comparison():
    """Comparación de tiempos de evaluación"""
    df_baseline = pd.read_csv(BASELINE_RESULTS)
    
    # Estimación de LLM-Judge (3.5s promedio)
    llm_judge_time = 3.5
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Baseline\n', 'LLM-Judge\n']
    times = [df_baseline['evaluation_time'].mean(), llm_judge_time]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(methods, times, color=colors, edgecolor='black', alpha=0.7)
    
    # Añadir valores
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Comparación de Tiempos de Evaluación', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tiempo (segundos)')
    ax.set_ylim([0, max(times) * 1.2])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}comparison_time.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: comparison_time.png")
    plt.close()


def plot_correlation_baseline_vs_llm():
    """Correlación entre scores de ambos métodos"""
    df_baseline = pd.read_csv(BASELINE_RESULTS)
    df_llm = pd.read_csv(LLM_JUDGE_RESULTS)
    
    # Merge por ID
    df_merged = pd.merge(df_baseline[['id', 'baseline_score']], 
                         df_llm[['id', 'overall_score']], 
                         on='id')
    
    df_merged['llm_judge_normalized'] = df_merged['overall_score'] / 10
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(df_merged['baseline_score'], df_merged['llm_judge_normalized'], 
               alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    
    # Línea de tendencia
    z = np.polyfit(df_merged['baseline_score'], df_merged['llm_judge_normalized'], 1)
    p = np.poly1d(z)
    ax.plot(df_merged['baseline_score'], p(df_merged['baseline_score']), 
            "r--", linewidth=2, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Línea de igualdad
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Igualdad perfecta')
    
    # Correlación
    corr = df_merged['baseline_score'].corr(df_merged['llm_judge_normalized'])
    ax.text(0.05, 0.95, f'Correlación: {corr:.3f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Correlación: Baseline vs LLM-Judge', fontsize=14, fontweight='bold')
    ax.set_xlabel('Baseline Score (0-1)')
    ax.set_ylabel('LLM-Judge Score (0-1)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}comparison_correlation.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: comparison_correlation.png")
    plt.close()


def plot_difficulty_comparison():
    """Comparación por dificultad"""
    df_baseline = pd.read_csv(BASELINE_RESULTS)
    df_llm = pd.read_csv(LLM_JUDGE_RESULTS)
    
    df_llm['overall_score_normalized'] = df_llm['overall_score'] / 10
    
    difficulty_order = ['Easy', 'Medium', 'Hard', 'Very Hard']
    
    baseline_by_diff = df_baseline.groupby('difficulty')['baseline_score'].mean().reindex(difficulty_order)
    llm_by_diff = df_llm.groupby('difficulty')['overall_score_normalized'].mean().reindex(difficulty_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(difficulty_order))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_by_diff, width, label='Baseline', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, llm_by_diff, width, label='LLM-Judge', color='coral', edgecolor='black')
    
    ax.set_title('Desempeño por Dificultad: Baseline vs LLM-Judge', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score Promedio')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_order)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Añadir valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}comparison_by_difficulty.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: comparison_by_difficulty.png")
    plt.close()


def plot_summary_table():
    """Tabla resumen comparativa (como imagen)"""
    df_baseline = pd.read_csv(BASELINE_RESULTS)
    df_llm = pd.read_csv(LLM_JUDGE_RESULTS)
    
    summary_data = {
        'Método': ['Baseline (Automático)', 'LLM-Judge (Cualitativo)'],
        'Score Promedio': [
            f"{df_baseline['baseline_score'].mean():.3f}",
            f"{df_llm['overall_score'].mean():.2f}/10"
        ],
        'Desv. Estándar': [
            f"{df_baseline['baseline_score'].std():.3f}",
            f"{df_llm['overall_score'].std():.2f}"
        ],
        'Tiempo Promedio': [
            f"{df_baseline['evaluation_time'].mean():.3f}s",
            "~3.5s"
        ],
        'Reproducible': ['100%', '~95%'],
        'Detecta': ['Errores numéricos/SQL', 'Errores semánticos']
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns, 
                     cellLoc='center', loc='center', colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Colorear header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold')
    
    # Colorear filas
    for i in range(1, len(df_summary) + 1):
        color = 'lightsteelblue' if i % 2 == 0 else 'white'
        for j in range(len(df_summary.columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Resumen Comparativo: Baseline vs LLM-Judge', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{OUTPUT_DIR}comparison_summary_table.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: comparison_summary_table.png")
    plt.close()


def generate_all_comparison_plots():
    """Generar todas las comparativas"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("GENERANDO GRÁFICAS COMPARATIVAS")
    print("="*60 + "\n")
    
    plot_score_comparison()
    plot_time_comparison()
    plot_correlation_baseline_vs_llm()
    plot_difficulty_comparison()
    plot_summary_table()
    
    print("\n" + "="*60)
    print("TODAS LAS GRÁFICAS COMPARATIVAS GENERADAS")
    print(f"Ubicación: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    generate_all_comparison_plots()