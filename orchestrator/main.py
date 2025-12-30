from orchestrator.graph import build_graph
from langchain_core.messages import HumanMessage

def main():
    app = build_graph()
    print("### SISTEMA MULTI-AGENTE CRYPTO & WEATHER ###")
    
    while True:
        query = input("\nUsuario: ")
        if query.lower() in ["salir", "exit"]:
            break
        
        # Iniciar ejecución
        inputs = {"messages": [HumanMessage(content=query)]}
        for chunk in app.stream(inputs):
            # Imprimimos quién está trabajando
            for key, value in chunk.items():
                print(f"Actualización de nodo: {key}")
                if "messages" in value:
                    print(f"Respuesta: {value['messages'][-1].content}")

if __name__ == "__main__":
    main()