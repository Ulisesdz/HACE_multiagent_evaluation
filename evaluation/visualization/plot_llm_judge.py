import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

LLM_JUDGE_RESULTS = "evaluation/llm_j/dataset_results.csv"
OUTPUT_DIR = "evaluation/visualization/plots/"


def plot_llm_judge_modules():
    """Scores por módulo (Planner, Supervisor, Agents, Final)"""
    df = pd.read_csv(LLM_JUDGE_RESULTS)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modules = ['planner_score', 'supervisor_score', 'agents_avg_score', 'final_output_score']
    module_names = ['Planner', 'Supervisor', 'Agentes', 'Output Final']
    
    data = [df[col].dropna() for col in modules]
    
    bp = ax.boxplot(data, tick_labels=module_names, patch_artist=True, showmeans=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Distribución de Scores LLM-Judge por Módulo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (0-10)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 10.5])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}llm_judge_modules_boxplot.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: llm_judge_modules_boxplot.png")
    plt.close()


def plot_error_categories():
    """Distribución de categorías de error"""
    df = pd.read_csv(LLM_JUDGE_RESULTS)
    
    error_counts = df['error_category'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(error_counts)))
    bars = ax.barh(error_counts.index, error_counts.values, color=colors)
    
    ax.set_title('Distribución de Categorías de Error (LLM-Judge)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frecuencia')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Añadir porcentajes
    for i, (bar, count) in enumerate(zip(bars, error_counts.values)):
        percentage = (count / len(df)) * 100
        ax.text(count + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}llm_judge_error_categories.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: llm_judge_error_categories.png")
    plt.close()


def plot_llm_judge_by_difficulty():
    """Overall score por dificultad"""
    df = pd.read_csv(LLM_JUDGE_RESULTS)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    difficulty_order = ['Easy', 'Medium', 'Hard', 'Very Hard']
    df_plot = df.groupby('difficulty')['overall_score'].agg(['mean', 'std', 'count'])
    df_plot = df_plot.reindex(difficulty_order)
    
    bars = ax.bar(df_plot.index, df_plot['mean'], yerr=df_plot['std'], 
                   capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Añadir n
    for bar, count in zip(bars, df_plot['count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Overall Score LLM-Judge por Dificultad', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (0-10)')
    ax.set_ylim([0, 11])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}llm_judge_by_difficulty.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: llm_judge_by_difficulty.png")
    plt.close()


def generate_all_llm_judge_plots():
    """Generar todas las gráficas de LLM-Judge"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERANDO GRÁFICAS LLM-JUDGE")
    print("="*60 + "\n")
    
    plot_llm_judge_modules()
    plot_error_categories()
    plot_llm_judge_by_difficulty()
    
    print("\n" + "="*60)
    print("TODAS LAS GRÁFICAS LLM-JUDGE GENERADAS")
    print(f"Ubicación: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    generate_all_llm_judge_plots()