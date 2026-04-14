import joblib
import os
import numpy as np
import pandas as pd


def predict_price(symbol, p1, p2, p3):
    """
    symbol: Nombre de la moneda (ej: 'BTC_USD')
    p1, p2, p3: Precios de los últimos 3 días (orden: t-1, t-2, t-3)
    """
    # Defino las rutas a la carpeta de modelos
    folder = "./crypto"
    models_folder = os.path.join(folder, "models")

    # Normalizo el nombre para que coincida con el archivo guardado
    table_name = symbol.replace("-", "_")
    model_name = f"model_{table_name}.joblib"
    model_path = os.path.join(models_folder, model_name)

    # Compruebo si el archivo existe antes de intentar cargarlo
    if not os.path.exists(model_path):
        return f"Error: No he encontrado el modelo en {model_path}"

    # Cargo el modelo
    model = joblib.load(model_path)

    # Preparo los datos en el formato 2D que requiere sklearn
    input_df = pd.DataFrame([[p1, p2, p3]], columns=["d-1", "d-2", "d-3"])

    # Realizo la prediccion y devuelvo el valor escalar
    prediction = model.predict(input_df)
    return prediction[0]


if __name__ == "__main__":
    # Defino la moneda y los datos de prueba
    moneda = "BTC_USD"

    # Asigno valores ficticios para probar el funcionamiento
    precio_ayer = 95000.50
    precio_anteayer = 94200.10
    precio_hace_3_dias = 93800.00

    # Ejecuto la funcion de prediccion
    resultado = predict_price(moneda, precio_ayer, precio_anteayer, precio_hace_3_dias)

    if isinstance(resultado, (float, np.float64)):
        print(f"Prediccion para el proximo cierre de {moneda}: {resultado:.2f}")
    else:
        print(resultado)
