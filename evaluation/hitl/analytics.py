import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

def calculate_alignment_metrics():
    csv_path = "evaluation/hitl/golden_dataset.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("No hay golden dataset aún.")
        return

    y_human = df['human_score']
    y_ai = df['score']

    # 1. Precisión Exacta (Exact Match)
    acc = accuracy_score(y_human, y_ai)
    
    # 2. Matriz de Confusión (Dónde discrepan)
    # Muestra si la IA es más severa o más laxa que tú
    cm = confusion_matrix(y_human, y_ai, labels=[0, 5, 10])

    print(f"REPORTE DE ALINEACIÓN (n={len(df)})")
    print("---------------------------------------")
    print(f"Acuerdo Exacto (Accuracy): {acc:.2%}")
    print(f"\nMatriz de Confusión (Filas=Tú, Cols=IA):\n{cm}")
    print("\nInterpetación Matriz:")
    print("- Diagonal: Acuerdo.")
    print("- Arriba-Derecha: IA sobreestima (Tu pones 0, IA pone 10).")
    print("- Abajo-Izquierda: IA subestima (Tu pones 10, IA pone 0).")

if __name__ == "__main__":
    calculate_alignment_metrics()