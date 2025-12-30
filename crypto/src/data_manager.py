import yfinance as yf
import sqlite3
import os

def download_and_save(symbols=["BTC-USD", "ETH-USD", "SOL-USD"]):
    # 1. Carpeta 'crypto'
    folder = "./crypto"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 2. Conectar
    db_path = os.path.join(folder, "crypto_data.db")
    conn = sqlite3.connect(db_path)
    
    # 3. Descargar datos
    for symbol in symbols:
        df = yf.download(symbol, period="2y", interval="1d")
        # Convierto ('Close', 'BTC-USD') en 'Close'
        df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        # Cripto por tabla
        table_name = symbol.replace("-", "_") 
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"{symbol} guardado en tabla: {table_name}")
    
    conn.close()

    print(f"Datos de {symbol} guardados en {db_path}")

if __name__ == "__main__":
    download_and_save()