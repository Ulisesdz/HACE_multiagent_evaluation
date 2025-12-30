import pandas as pd
import sqlite3
import os
import kagglehub

def download_and_save(cities=["Madrid", "Paris", "New York City", "Tokyo"]):
    # 1. Carpeta 'weather'
    folder = "./weather"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 2. Descargar datos (vía kagglehub)
    print("Descargando dataset de Kaggle...")
    path = kagglehub.dataset_download("sudalairajkumar/daily-temperature-of-major-cities")
    csv_file = os.path.join(path, "city_temperature.csv")
    
    print("Procesando CSV...")
    # Cargo el dataframe completo
    df_full = pd.read_csv(csv_file, low_memory=False)

    # 3. Conectar
    db_path = os.path.join(folder, "weather_data.db")
    conn = sqlite3.connect(db_path)

    for city in cities:
        # Filtro por ciudad y elimino los valores -99 que son errores
        city_df = df_full[(df_full['City'] == city) & (df_full['AvgTemperature'] > -90)].copy()
        
        # Convierto la temperatura de Fahrenheit a Celsius
        # Aplico la formula (F - 32) * 5/9
        city_df['AvgTemperature'] = (city_df['AvgTemperature'] - 32) * (5/9)
        
        # Limpio el nombre de la ciudad para la tabla
        table_name = city.replace(" ", "_")
        
        # Guardo cada ciudad en su propia tabla con los datos ya convertidos
        city_df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Tabla '{table_name}' creada en Celsius con {len(city_df)} registros.")

    conn.close()
    print(f"\nBase de datos completa guardada en: {db_path}")

if __name__ == "__main__":
    download_and_save()