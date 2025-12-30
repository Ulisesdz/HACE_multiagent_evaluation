import pandas as pd
import sqlite3
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_all_models():
    # 1. Rutas
    folder = "./crypto"
    models_folder = os.path.join(folder, "models")
    db_path = os.path.join(folder, "crypto_data.db")
    
    if not os.path.exists(db_path):
        print("Error: No se encontró la base de datos.")
        return

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        
    conn = sqlite3.connect(db_path)
    
    # 2. Obtener lista de tablas
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        # 3. Cargar datos
        df_full = pd.read_sql(f'SELECT * FROM {table}', conn)
        
        # Limpieza de columnas
        col_close = [c for c in df_full.columns if 'Close' in str(c)][0]
        df = df_full[[col_close]].copy()
        df.columns = ['Close']
        
        # 4. Feature Engineering: Lags
        df['d-1'] = df['Close'].shift(1)
        df['d-2'] = df['Close'].shift(2)
        df['d-3'] = df['Close'].shift(3)
        df = df.dropna()

        X = df[['d-1', 'd-2', 'd-3']]
        y = df['Close']

        # 5. Split Cronológico (80% tren, 20% test)
        # No uso shuffle=True porque perdería el orden temporal
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # 6. Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 7. Evaluación (Test)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Precisión porcentual simple
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        accuracy_perc = 100 - mape

        # 8. Guardar modelo
        model_name = f"model_{table}.joblib"
        model_path = os.path.join(models_folder, model_name)
        joblib.dump(model, model_path)
        
        print(f"\n--- Resultados para {table} ---")
        print(f"Modelo guardado en: {model_path}")
        print(f"Error Medio (MAE): {mae:.2f} USD")
        print(f"R2 Score: {r2:.4f}")
        print(f"Accuracy aprox (100-MAPE): {accuracy_perc:.2f}%")

    conn.close()

if __name__ == "__main__":
    train_all_models()