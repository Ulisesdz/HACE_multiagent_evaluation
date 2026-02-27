import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

BASELINE_RESULTS = "evaluation/baseline/dataset_baseline_results.csv"
OUTPUT_DIR = "evaluation/visualization/plots/"


def plot_baseline_metrics_distribution():
    """Distribución de las 4 métricas baseline"""
    df = pd.read_csv(BASELINE_RESULTS)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribución de Métricas Baseline', fontsize=16, fontweight='bold')
    
    metrics = [
        ('routing_f1', 'Routing F1-Score', axes[0, 0]),
        ('numeric_f1', 'Numeric F1-Score', axes[0, 1]),
        ('task_coverage', 'Task Coverage', axes[1, 0]),
        ('sql_correctness', 'SQL Correctness', axes[1, 1])
    ]
    
    for col, title, ax in metrics:
        ax.hist(df[col], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df[col].mean():.3f}')
        ax.set_title(title)
        ax.set_xlabel('Score (0-1)')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}baseline_metrics_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: baseline_metrics_distribution.png")
    plt.close()


def plot_baseline_by_difficulty():
    """Scores por nivel de dificultad"""
    df = pd.read_csv(BASELINE_RESULTS)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    difficulty_order = ['Easy', 'Medium', 'Hard', 'Very Hard']
    df_plot = df.groupby('difficulty')[['baseline_score', 'routing_f1', 'numeric_f1', 'task_coverage']].mean()
    df_plot = df_plot.reindex(difficulty_order)
    
    df_plot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Desempeño Baseline por Dificultad', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dificultad')
    ax.set_ylabel('Score Promedio')
    ax.set_xticklabels(df_plot.index, rotation=45)
    ax.legend(title='Métricas', loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}baseline_by_difficulty.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: baseline_by_difficulty.png")
    plt.close()


def plot_hallucination_analysis():
    """Análisis detallado de alucinaciones"""
    df = pd.read_csv(BASELINE_RESULTS)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de hallucination rate
    axes[0].hist(df['hallucination_rate'], bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[0].axvline(df['hallucination_rate'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Media: {df['hallucination_rate'].mean():.1%}')
    axes[0].set_title('Distribución de Alucinaciones Numéricas')
    axes[0].set_xlabel('Hallucination Rate')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Casos con alucinaciones por categoría
    df_halluc = df[df['hallucination_rate'] > 0.1].groupby('category').size().sort_values(ascending=False).head(10)
    axes[1].barh(df_halluc.index, df_halluc.values, color='coral')
    axes[1].set_title('Categorías con Mayor Tasa de Alucinaciones (>10%)')
    axes[1].set_xlabel('Número de Casos')
    axes[1].grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}baseline_hallucination_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: baseline_hallucination_analysis.png")
    plt.close()


def plot_sql_correctness_breakdown():
    """Desglose de SQL correctness"""
    df = pd.read_csv(BASELINE_RESULTS)
    
    # Filtrar solo casos con SQL queries
    df_sql = df[df['sql_total_queries'] > 0].copy()
    
    if len(df_sql) == 0:
        print("(WARNING) No hay casos con SQL queries en el dataset")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribución de correctness
    axes[0].hist(df_sql['sql_correctness'], bins=10, edgecolor='black', alpha=0.7, color='lightblue')
    axes[0].axvline(df_sql['sql_correctness'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Media: {df_sql['sql_correctness'].mean():.1%}')
    axes[0].set_title('Distribución de SQL Correctness')
    axes[0].set_xlabel('Correctness (0-1)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Top violaciones
    violations_all = []
    for violations_str in df_sql['sql_violations'].dropna():
        if violations_str:
            violations_all.extend(violations_str.split('; '))
    
    if violations_all:
        from collections import Counter
        top_violations = Counter(violations_all).most_common(5)
        names, counts = zip(*top_violations)
        
        axes[1].barh(names, counts, color='lightcoral')
        axes[1].set_title('Top 5 Violaciones SQL')
        axes[1].set_xlabel('Frecuencia')
        axes[1].grid(True, axis='x', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No hay violaciones SQL', ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}baseline_sql_correctness.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: baseline_sql_correctness.png")
    plt.close()


def plot_baseline_heatmap():
    """Heatmap de correlación entre métricas"""
    df = pd.read_csv(BASELINE_RESULTS)
    
    metrics_cols = ['baseline_score', 'routing_f1', 'numeric_f1', 'task_coverage', 'sql_correctness', 'hallucination_rate']
    corr_matrix = df[metrics_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlación entre Métricas Baseline', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}baseline_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Guardado: baseline_correlation_heatmap.png")
    plt.close()


def generate_all_baseline_plots():
    """Generar todas las visualizaciones de Baseline"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERANDO GRÁFICAS BASELINE")
    print("="*60 + "\n")
    
    plot_baseline_metrics_distribution()
    plot_baseline_by_difficulty()
    plot_hallucination_analysis()
    plot_sql_correctness_breakdown()
    plot_baseline_heatmap()
    
    print("\n" + "="*60)
    print("TODAS LAS GRÁFICAS BASELINE GENERADAS")
    print(f"Ubicación: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    generate_all_baseline_plots()