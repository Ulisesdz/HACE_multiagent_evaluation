import joblib
import os
import numpy as np
import pandas as pd

def predict_weather(city_name, t1, t2):
    """
    city_name: Nombre de la ciudad (ej: 'Madrid')
    t1, t2: Temperaturas de los últimos 2 días (orden: t-1, t-2)
    """
    # Defino las rutas a la carpeta de modelos de clima
    folder = "./weather"
    models_folder = os.path.join(folder, "models")
    
    # Normalizo el nombre para que coincida con la tabla de la base de datos
    table_name = city_name.replace(" ", "_")
    model_name = f"model_{table_name}.joblib"
    model_path = os.path.join(models_folder, model_name)
    
    # Compruebo si el archivo del modelo existe
    if not os.path.exists(model_path):
        return f"Error: No he encontrado el modelo para {city_name} en {model_path}"

    # Cargo el modelo entrenado
    model = joblib.load(model_path)
    
    # Creo el DataFrame con los nombres de columnas usados en el entrenamiento
    # Esto evita el aviso de nombres de caracteristicas (feature names)
    input_df = pd.DataFrame([[t1, t2]], columns=['t-1', 't-2'])
    
    # Realizo la prediccion
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    # Defino la ciudad y las temperaturas previas para la prueba
    ciudad = "Madrid"
    
    # Introduzco valores de ejemplo (suelen estar en Fahrenheit en este dataset)
    temp_ayer = 27.0
    temp_anteayer = 29.5
    
    # Ejecuto la funcion de prediccion para la ciudad seleccionada
    resultado = predict_weather(ciudad, temp_ayer, temp_anteayer)
    
    # Imprimo el resultado final
    if isinstance(resultado, (float, np.float64)):
        print(f"Prediccion de temperatura para {ciudad}: {resultado:.2f}")
    else:
        print(resultado)