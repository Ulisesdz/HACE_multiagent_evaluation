import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.graph import build_graph
from langchain_core.messages import HumanMessage

# Códigos de color ANSI para la consola
COLORS = {
    "Supervisor": "\033[93m",  # Amarillo
    "Technical_Analyst": "\033[94m",  # Azul
    "Fundamental_Analyst": "\033[92m",  # Verde
    "Risk_Officer": "\033[91m",  # Rojo (Alerta)
    "RESET": "\033[0m",
}


def main():
    app = build_graph()

    print("\n" + "=" * 50)
    print("   🏛️  AI INVESTMENT COMMITTEE (CLI)  🏛️")
    print("=" * 50)
    print("Roles Activos:")
    print(
        f" {COLORS['Technical_Analyst']}• Technical Analyst{COLORS['RESET']} (Quant, ML, Gráficos)"
    )
    print(
        f" {COLORS['Fundamental_Analyst']}• Fundamental Analyst{COLORS['RESET']} (Noticias, RAG)"
    )
    print(
        f" {COLORS['Risk_Officer']}• Risk Officer{COLORS['RESET']} (Volatilidad, Seguridad)"
    )
    print(f" {COLORS['Supervisor']}• Supervisor{COLORS['RESET']} (CIO)")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("Usuario (Inversor): ")
            if query.lower() in ["salir", "exit", "quit"]:
                print("Cerrando sesión de trading...")
                break

            # Iniciar ejecución
            inputs = {"messages": [HumanMessage(content=query)]}

            print("\n--- 🧠 Procesando Solicitud ---")

            for chunk in app.stream(inputs):
                # Agente trabajando
                for key, value in chunk.items():
                    agent_color = COLORS.get(key, COLORS["RESET"])

                    print(f"{agent_color}▶ Actividad en nodo: {key}{COLORS['RESET']}")

                    if "messages" in value:
                        # Último mensaje generado por el agente
                        last_msg = value["messages"][-1]
                        content = last_msg.content

                        # Si es un mensaje de herramienta
                        print(
                            f"{agent_color}  └─ Respuesta:{COLORS['RESET']} {content}\n"
                        )

        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario.")
            break
        except Exception as e:
            print(f"Error en la ejecución: {e}")


if __name__ == "__main__":
    main()
