import pandas as pd
import sqlite3
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_all_models():
    # 1. Rutas
    folder = "./crypto"
    models_folder = os.path.join(folder, "models")
    plots_folder = os.path.join(folder, "plots")
    db_path = os.path.join(folder, "crypto_data.db")

    if not os.path.exists(db_path):
        print("Error: No se encontró la base de datos.")
        return

    # Creo la carpeta para los modelos y plots si aun no existe
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # 2. Obtener lista de tablas
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        # 3. Cargar datos
        try:
            df_full = pd.read_sql(f'SELECT * FROM "{table}" ORDER BY Date ASC', conn)
        except:
            df_full = pd.read_sql(f'SELECT * FROM "{table}"', conn)

        # Limpieza de columnas
        col_close = next((c for c in df_full.columns if "Close" in str(c)), None)
        col_date = next(
            (c for c in df_full.columns if "Date" in str(c) or "date" in str(c)), None
        )
        df = df_full[[col_close]].copy()
        df.columns = ["Close"]

        dates = None
        if col_date:
            try:
                dates = pd.to_datetime(df_full[col_date])
            except Exception as e:
                print(f"No se pudo convertir la fecha: {e}")
                dates = None

        # 4. Feature Engineering: Lags
        df["d-1"] = df["Close"].shift(1)
        df["d-2"] = df["Close"].shift(2)
        df["d-3"] = df["Close"].shift(3)
        valid_indices = df.dropna().index
        df = df.dropna()

        # Aline fechas con los datos limpios
        if dates is not None:
            dates = dates.loc[valid_indices]

        X = df[["d-1", "d-2", "d-3"]]
        y = df["Close"]

        # 5. Split Cronológico (80% tren, 20% test)
        # No uso shuffle=True porque perdería el orden temporal
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        dates_test = dates.iloc[split:] if dates is not None else range(len(y_test))

        # 6. Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 7. Evaluación (Test) - Random Forest
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Evaluación - BASELINE (Media de los 3 días anteriores)
        # X_test tiene las columnas "d-1", "d-2", "d-3", así que su media por fila es el baseline
        baseline_predictions = X_test.mean(axis=1)
        baseline_mae = mean_absolute_error(y_test, baseline_predictions)

        # Precisión porcentual simple
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            accuracy_perc = 100 - mape if not np.isnan(mape) else 0

        # 8. Guardar modelo
        model_name = f"model_{table}.joblib"
        model_path = os.path.join(models_folder, model_name)
        joblib.dump(model, model_path)

        # 9. Gráficas
        plt.figure(figsize=(12, 6))
        plt.plot(
            dates_test,
            y_test.values,
            label="Precio Real",
            color="blue",
            linewidth=2,
            alpha=0.7,
        )
        plt.plot(
            dates_test,
            predictions,
            label="Predicción",
            color="orange",
            linestyle="--",
            linewidth=2,
        )

        plt.title(
            f"Validación Modelo: {table}\nMAE: ${mae:.2f} | R2: {r2:.4f} | Acc: {accuracy_perc:.1f}%"
        )
        plt.xlabel("Fecha")
        plt.ylabel("Precio (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(plots_folder, f"validation_{table}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"\n--- Resultados para {table} ---")
        print(f"Modelo guardado en: {model_path}")
        print(f"Error Medio (MAE): {mae:.2f} USD")
        print(f"R2 Score: {r2:.4f}")
        print(f"Accuracy aprox (100-MAPE): {accuracy_perc:.2f}%")

        print(f"MAE Baseline (Media 3d): {baseline_mae:.2f} USD")
        print(f"Mejora vs Baseline: {baseline_mae - mae:.2f} USD")

    conn.close()


if __name__ == "__main__":
    train_all_models()
