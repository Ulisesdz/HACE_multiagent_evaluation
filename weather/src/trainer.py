import pandas as pd
import sqlite3
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_all_models():
    # 1. Definir rutas
    folder = "./weather"
    models_folder = os.path.join(folder, "models")
    plots_folder = os.path.join(folder, "plots")
    db_path = os.path.join(folder, "weather_data.db")
    
    # Compruebo si existe la base de datos antes de continuar
    if not os.path.exists(db_path):
        print(f"Error: No he encontrado la base de datos en {db_path}")
        return

    # Creo la carpeta para los modelos y plots si aun no existe
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Establezco conexion con la base de datos sqlite
    conn = sqlite3.connect(db_path)
    
    # Obtengo los nombres de todas las tablas que corresponden a las ciudades
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        # 2. Cargar datos
        try:
            df_full = pd.read_sql(f'SELECT * FROM "{table}" ORDER BY Year, Month, Day', conn)
        except Exception as e:
            print(f"Error leyendo tabla {table}: {e}")
            continue
        
        # 3. Identificar columna de Temperatura
        col_temp = next((c for c in df_full.columns if 'Temp' in c), None)

        # 4. Construcción de fecha para graficar
        dates = None
        try:
            date_cols = df_full[['Year', 'Month', 'Day']].copy()
            date_cols.columns = ['year', 'month', 'day']
            dates = pd.to_datetime(date_cols)
        except Exception as e:
            print(f"No pude construir fechas para gráfico en {table}: {e}")
            dates = None
        df = df_full[[col_temp]].copy()
        df.columns = ['Temp']

        # 5. Feature Engineering: Lags (t-1, t-2)
        df['t-1'] = df['Temp'].shift(1)
        df['t-2'] = df['Temp'].shift(2)
        
        # Elimino las filas con valores nulos generados por el desplazamiento
        valid_indices = df.dropna().index
        df = df.dropna()
        if dates is not None:
            dates = dates.loc[valid_indices]

        # Defino mis variables de entrada y la variable objetivo
        X = df[['t-1', 't-2']]
        y = df['Temp']

        # 6. Split Cronológico (80% / 20%)
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # 7. Entrenar Modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Fechas para test
        dates_test = dates.iloc[split:] if dates is not None else range(len(y_test))
        
        # 8. Evaluación
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculo el porcentaje de precision basado en el error absoluto
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            # Si hay infinitos (temp=0), reemplazamos o ajustamos
            if np.isinf(mape).any(): mape = np.nan
            accuracy_perc = 100 - mape if not np.isnan(mape) else 0

        # Guardo el modelo entrenado en formato joblib
        model_name = f"model_{table}.joblib"
        model_path = os.path.join(models_folder, model_name)
        joblib.dump(model, model_path)

        # 10. Gráfica
        plt.figure(figsize=(12, 6))
        
        # Graficamos Real vs Predicción
        plt.plot(dates_test, y_test.values, label='Temperatura Real', color='blue', linewidth=2, alpha=0.6)
        plt.plot(dates_test, predictions, label='Predicción IA', color='red', linestyle='--', linewidth=2)
        
        plt.title(f"Validación Clima: {table}\nMAE: {mae:.2f}° | R2: {r2:.4f} | Acc: {accuracy_perc:.1f}%")
        plt.xlabel("Fecha")
        plt.ylabel("Temperatura")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        plot_path = os.path.join(plots_folder, f"validation_{table}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Muestro los resultados por consola para cada ciudad
        print(f"\n--- Resultados para {table} ---")
        print(f"Modelo guardado en: {model_path}")
        print(f"Error Medio (MAE): {mae:.2f} grados")
        print(f"R2 Score: {r2:.4f}")
        print(f"Precision aprox: {accuracy_perc:.2f}%")

    # Cierro la conexion a la base de datos
    conn.close()

if __name__ == "__main__":
    # Ejecuto el proceso de entrenamiento global
    train_all_models()