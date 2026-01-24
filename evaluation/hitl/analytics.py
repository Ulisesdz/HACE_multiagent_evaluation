import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

def calculate_alignment_metrics():
    csv_path = "evaluation/hitl/golden_dataset.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("No hay golden dataset aún. Usa labeling_app.py primero.")
        return

    y_human = df['human_score']
    y_ai = df['score']

    print(f"\nREPORTE DE ALINEACIÓN HUMANO-IA (n={len(df)})")
    print("=======================================")

    # 1. Precisión Exacta
    acc = accuracy_score(y_human, y_ai)
    print(f"✅ Exact Match (Accuracy): {acc:.2%}")

    # 2. Kappa de Cohen (Acuerdo ajustado por azar)
    # > 0.8: Excelente | 0.6-0.8: Bueno | < 0.4: Pobre
    kappa = cohen_kappa_score(y_human, y_ai, labels=[0, 5, 10])
    print(f"Cohen's Kappa:        {kappa:.3f}")
    
    # 3. Matriz de Confusión
    cm = confusion_matrix(y_human, y_ai, labels=[0, 5, 10])
    print("\nMatriz de Confusión (Filas=Tú, Cols=IA):")
    print("      IA_0  IA_5  IA_10")
    print(f"Tú_0   {cm[0]}")
    print(f"Tú_5   {cm[1]}")
    print(f"Tú_10  {cm[2]}")

    # 4. Análisis de Discrepancias
    disagreements = df[df['human_score'] != df['score']]
    if not disagreements.empty:
        print("\nDiscrepancias Destacadas:")
        for idx, row in disagreements.iterrows():
            print(f"- Caso {row.get('id', idx)}: Tú={row['human_score']} vs IA={row['score']}")
            print(f"  Notas: {row.get('human_notes', 'N/A')}\n")

if __name__ == "__main__":
    calculate_alignment_metrics()