import pandas as pd
import sqlite3
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_all_models():
    # Defino las rutas principales del modulo
    folder = "./weather"
    models_folder = os.path.join(folder, "models")
    db_path = os.path.join(folder, "weather_data.db")
    
    # Compruebo si existe la base de datos antes de continuar
    if not os.path.exists(db_path):
        print(f"Error: No he encontrado la base de datos en {db_path}")
        return

    # Creo la carpeta para los modelos si aun no existe
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print(f"Carpeta creada: {models_folder}")

    # Establezco conexion con la base de datos sqlite
    conn = sqlite3.connect(db_path)
    
    # Obtengo los nombres de todas las tablas que corresponden a las ciudades
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        # Cargo los datos de la ciudad actual
        df_full = pd.read_sql(f'SELECT AvgTemperature FROM {table}', conn)
        
        # Renombro la columna para trabajar con un estandar
        df = df_full.copy()
        df.columns = ['Temp']
        
        # Genero los retrasos (lags) de los 2 dias anteriores para predecir el actual
        df['t-1'] = df['Temp'].shift(1)
        df['t-2'] = df['Temp'].shift(2)
        
        # Elimino las filas con valores nulos generados por el desplazamiento
        df = df.dropna()

        # Defino mis variables de entrada y la variable objetivo
        X = df[['t-1', 't-2']]
        y = df['Temp']

        # Realizo un split cronologico: 80% entrenamiento y 20% test
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Entreno el regresor de bosque aleatorio
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculo metricas de evaluacion con el conjunto de test
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculo el porcentaje de precision basado en el error absoluto
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        accuracy_perc = 100 - mape

        # Guardo el modelo entrenado en formato joblib
        model_name = f"model_{table}.joblib"
        model_path = os.path.join(models_folder, model_name)
        joblib.dump(model, model_path)
        
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