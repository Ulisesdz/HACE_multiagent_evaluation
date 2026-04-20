import os
import warnings

warnings.filterwarnings("ignore")

import json
import ast
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

DATASET_JSON = "evaluation/metrics_accumulator/dataset.json"
OFFLINE_CSV = "evaluation/accumulated_data/offline_metrics.csv"
OUTPUT_CSV = "evaluation/hybrid/hace_calibration_results.csv"

with open(DATASET_JSON, encoding="utf-8") as f:
    dataset = json.load(f)

df_csv = pd.read_csv(OFFLINE_CSV)

# SEVERITY: cuánto baja el score si el agente SÍ cayó en la trampa
SEVERITY_IF_FAILED = {
    "None": 1.00,  # no había trampa
    "Planning_Error": 0.68,
    "Routing_Error": 0.66,
    "Task_Completion": 0.64,
    "Incompleteness": 0.62,
    "Logic_Error": 0.58,
    "Output_Fidelity": 0.55,
    "Tool_Error": 0.42,
    "Fabrication": 0.38,
    "Hallucination": 0.38,
    "Risk_Negligence": 0.40,
    "Loop_Error": 0.35,
    "Parametric_Leak": 0.52,
    "Chart_Attribution_Error": 0.48,
    "Structure": 0.60,
}

SCORE_IF_DODGED = 0.92  # el agente esquivó correctamente la trampa

DIFFICULTY_DISCOUNT = {
    "Easy": 0.00,
    "Medium": 0.02,
    "Hard": 0.04,
    "Very Hard": 0.06,
}


def parse_failure(s):
    if not s or s == "None":
        return "None"
    return s.split("(")[0].split(" or ")[0].strip()


def agent_actually_failed(row):
    """
    Detecta si el agente realmente cayó en la trampa, leyendo el CSV.
    Usa tres señales independientes del LLM:
      1. critical_failures tiene contenido real (no solo 'numeric anomalies')
      2. HACE_layer1 < 0.85 (guardrails detectaron problema estructural)
      3. La traza contiene palabras clave de error en tool_outputs
    """
    failed_signals = 0

    # Señal 1: critical_failures tiene errores no-numéricos
    try:
        cf = row.get("critical_failures", "[]")
        if isinstance(cf, str):
            cf_list = ast.literal_eval(cf)
        else:
            cf_list = cf if cf else []
        real_failures = [
            x
            for x in cf_list
            if "numeric anomalies" not in str(x).lower() and len(str(x).strip()) > 2
        ]
        if len(real_failures) > 0:
            failed_signals += 2  # señal fuerte
    except Exception:
        pass

    # Señal 2: HACE Layer 1 (guardrails) detectó problema
    l1 = row.get("HACE_layer1", 1.0)
    if pd.notna(l1) and l1 < 0.85:
        failed_signals += 1

    # Señal 3: la traza contiene error de herramienta
    try:
        trace = str(row.get("raw_trace", ""))
        error_keywords = [
            "Error invoking tool",
            "tool_error",
            "ToolException",
            "Input should be a valid string",
            "No data found",
            "not found in database",
            "Error:",
        ]
        if any(kw.lower() in trace.lower() for kw in error_keywords):
            failed_signals += 1
    except Exception:
        pass

    return failed_signals >= 2  # requiere al menos 2 señales para evitar FP


# Construir ground truth adaptativo
id_to_case = {c["id"]: c for c in dataset}


def build_gt(row):
    qid = row["query_id"]
    case = id_to_case.get(qid, {})
    fail = parse_failure(case.get("failure_type_expected", "None"))
    diff = case.get("difficulty", "Medium")
    disc = DIFFICULTY_DISCOUNT.get(diff, 0.03)

    if fail == "None":
        # No había trampa: score perfecto menos descuento por dificultad
        return max(0.0, 1.0 - disc)
    else:
        # Había trampa: ¿el agente cayó?
        if agent_actually_failed(row):
            base = SEVERITY_IF_FAILED.get(fail, 0.58)
            return max(0.0, base - disc)
        else:
            # Esquivó la trampa correctamente → score alto
            return max(0.0, SCORE_IF_DODGED - disc)


df_csv["ref_score"] = df_csv.apply(build_gt, axis=1)

needed = [
    "query_id",
    "HACE_layer1",
    "HACE_layer2",
    "HACE_layer3",
    "HACE_layer3_used",
    "ref_score",
    "difficulty",
]
df = df_csv[needed].dropna(subset=["HACE_layer1", "HACE_layer2", "ref_score"])

print("=" * 65)
print(f"Casos válidos: {len(df)}")
print(f"Distribución dificultad:\n{df['difficulty'].value_counts().to_string()}")
print(f"\nGround truth stats:")
print(f"  Media:  {df['ref_score'].mean():.3f}")
print(f"  Std:    {df['ref_score'].std():.3f}")
print(f"  Min:    {df['ref_score'].min():.3f}")
print(f"  Max:    {df['ref_score'].max():.3f}")

# Cuántos casos el agente "esquivó" la trampa vs cayó
failed_cases = (
    df_csv[df_csv["query_id"].isin(df["query_id"])]
    .apply(
        lambda r: agent_actually_failed(r)
        and parse_failure(
            id_to_case.get(r["query_id"], {}).get("failure_type_expected", "None")
        )
        != "None",
        axis=1,
    )
    .sum()
)
total_trap = sum(
    1
    for c in dataset
    if parse_failure(c.get("failure_type_expected", "None")) != "None"
)
print(f"\n  Casos con trampa en dataset: {total_trap}")
print(f"  Agente cayó en la trampa:    {failed_cases}")
print(f"  Agente esquivó la trampa:    {total_trap - failed_cases}")
print("=" * 65)

# GRID SEARCH
w1_range = np.round(np.arange(0.10, 0.51, 0.05), 2)
thresh_range = np.round(np.arange(0.55, 0.86, 0.05), 2)
alpha_range = np.round(np.arange(0.40, 0.71, 0.10), 2)

results = []

for w1, thresh, alpha in product(w1_range, thresh_range, alpha_range):
    w_sem = 1.0 - w1
    w2_no_l3 = w_sem
    w2_l3 = w_sem * alpha
    w3_l3 = w_sem * (1 - alpha)

    scores = []
    targets = df["ref_score"].values

    for _, row in df.iterrows():
        s1 = row["HACE_layer1"]
        s2 = row["HACE_layer2"]
        s3 = row["HACE_layer3"] if not np.isnan(row["HACE_layer3"]) else s2
        l3_active = (s2 <= thresh) or (abs(s1 - s2) > 0.3)
        if l3_active and not np.isnan(row["HACE_layer3"]):
            score = w1 * s1 + w2_l3 * s2 + w3_l3 * s3
        else:
            score = w1 * s1 + w2_no_l3 * s2
        scores.append(score)

    scores_arr = np.array(scores)
    if np.std(scores_arr) < 1e-6:
        continue

    r, _ = pearsonr(scores_arr, targets)
    rmse = np.sqrt(mean_squared_error(targets, scores_arr))
    mae = np.mean(np.abs(targets - scores_arr))
    results.append(
        {
            "w1": w1,
            "threshold": thresh,
            "alpha_l2": alpha,
            "w2_no_l3": round(w_sem, 3),
            "w2_l3": round(w2_l3, 3),
            "w3_l3": round(w3_l3, 3),
            "pearson_r": round(r, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        }
    )

results_df = pd.DataFrame(results).sort_values("pearson_r", ascending=False)
results_df.to_csv(OUTPUT_CSV, index=False)

# LOO-CV para estimación sin data leakage
print("\nEjecutando LOO-CV...")
n = len(df)
oos_preds = np.zeros(n)

for i in range(n):
    train_idx = [j for j in range(n) if j != i]
    test_row = df.iloc[i]

    best_r_train = -99
    best_p = None

    for w1, thresh, alpha in product(w1_range, thresh_range, alpha_range):
        w_sem = 1.0 - w1
        w2_no_l3 = w_sem
        w2_l3 = w_sem * alpha
        w3_l3 = w_sem * (1 - alpha)

        sc_train = []
        for j in train_idx:
            r_ = df.iloc[j]
            s1, s2 = r_["HACE_layer1"], r_["HACE_layer2"]
            s3 = r_["HACE_layer3"] if not np.isnan(r_["HACE_layer3"]) else s2
            l3 = (s2 <= thresh) or (abs(s1 - s2) > 0.3)
            sc_train.append(
                w1 * s1 + w2_l3 * s2 + w3_l3 * s3
                if (l3 and not np.isnan(r_["HACE_layer3"]))
                else w1 * s1 + w2_no_l3 * s2
            )

        tgt_train = df.iloc[train_idx]["ref_score"].values
        if np.std(sc_train) < 1e-6:
            continue
        r_tr, _ = pearsonr(sc_train, tgt_train)
        if r_tr > best_r_train:
            best_r_train = r_tr
            best_p = (w1, thresh, alpha, w2_no_l3, w2_l3, w3_l3)

    w1, thresh, alpha, w2_no_l3, w2_l3, w3_l3 = best_p
    s1, s2 = test_row["HACE_layer1"], test_row["HACE_layer2"]
    s3 = test_row["HACE_layer3"] if not np.isnan(test_row["HACE_layer3"]) else s2
    l3 = (s2 <= thresh) or (abs(s1 - s2) > 0.3)
    oos_preds[i] = (
        w1 * s1 + w2_l3 * s2 + w3_l3 * s3
        if (l3 and not np.isnan(test_row["HACE_layer3"]))
        else w1 * s1 + w2_no_l3 * s2
    )

final_r, _ = pearsonr(oos_preds, df["ref_score"].values)

# RESULTADOS
best = results_df.iloc[0]
print("\n" + "=" * 65)
print("TOP 5 CONFIGURACIONES:")
print("=" * 65)
print(results_df.head(5).to_string(index=False))

print("\n" + "=" * 65)
print("CONFIGURACIÓN ÓPTIMA:")
print("=" * 65)
print(f"  w1 (Guardrails)    = {best['w1']:.2f}")
print(f"  Umbral L3          = {best['threshold']:.2f}")
print(f"  alpha L2 (vs L3)   = {best['alpha_l2']:.2f}")
print(f"  -> w2 (sin L3)     = {best['w2_no_l3']:.3f}")
print(f"  -> w2 (con L3)     = {best['w2_l3']:.3f}")
print(f"  -> w3 (con L3)     = {best['w3_l3']:.3f}")
print(f"  Pearson r (global) = {best['pearson_r']:.4f}")

print("\n" + "=" * 65)
print("LOO-CV RESULTADO FINAL (sin data leakage):")
print("=" * 65)
print(f"  Pearson r (OOS) = {final_r:.4f}")

# Comparación con heurístico actual
print("\n" + "=" * 65)
print("COMPARACIÓN CON CONFIGURACIÓN HEURÍSTICA ACTUAL:")
print("=" * 65)
current = results_df[
    (results_df["w1"] == 0.30)
    & (results_df["threshold"].between(0.70, 0.80))
    & (results_df["alpha_l2"] == 0.50)
]
if not current.empty:
    c = current.iloc[0]
    print(f"  Heurístico (w1=0.30, thresh=0.75, alpha=0.50): r={c['pearson_r']:.4f}")
    print(
        f"  Óptimo     (w1={best['w1']:.2f}, thresh={best['threshold']:.2f}, alpha={best['alpha_l2']:.2f}): r={best['pearson_r']:.4f}"
    )
    print(f"  Δr = {best['pearson_r'] - c['pearson_r']:+.4f}")

# Análisis de sensibilidad
print("\n" + "=" * 65)
print("SENSIBILIDAD a w1 (thresh=0.75, alpha=0.50):")
print("=" * 65)
s1 = (
    results_df[
        (results_df["threshold"].between(0.70, 0.80)) & (results_df["alpha_l2"] == 0.50)
    ]
    .groupby("w1")["pearson_r"]
    .max()
)
print(s1.to_string())

print("\nSENSIBILIDAD a threshold (w1=0.30, alpha=0.50):")
s2 = (
    results_df[(results_df["w1"] == 0.30) & (results_df["alpha_l2"] == 0.50)]
    .groupby("threshold")["pearson_r"]
    .max()
)
print(s2.to_string())

print(f"\nResultados completos: {OUTPUT_CSV}")
print("=" * 65)

# Gráficas
VISUALIZATION_DIR = "evaluation/visualization/plots/"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Análisis de Sensibilidad de Hiperparámetros en HACE",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

# GRÁFICA 1: Sensibilidad a w1
sens_w1_plot = (
    results_df[
        (results_df["threshold"].between(0.70, 0.80)) & (results_df["alpha_l2"] == 0.50)
    ]
    .groupby("w1")["pearson_r"]
    .max()
    .reset_index()
)

axes[0].plot(
    sens_w1_plot["w1"],
    sens_w1_plot["pearson_r"],
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#2ecc71",
)
axes[0].axvline(
    x=0.30,
    color="red",
    linestyle="--",
    alpha=0.7,
    label="Peso Layer 1 Elegido (w1=0.30)",
)

# Anotar el punto máximo
max_w1_idx = sens_w1_plot["pearson_r"].idxmax()
axes[0].annotate(
    f"r={sens_w1_plot['pearson_r'].max():.4f}",
    (sens_w1_plot["w1"][max_w1_idx], sens_w1_plot["pearson_r"].max()),
    textcoords="offset points",
    xytext=(0, 10),
    ha="center",
    fontweight="bold",
)

axes[0].set_title("Efecto del Peso Estructural ($w_1$)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Peso de la Capa 1 ($w_1$)", fontsize=11)
axes[0].set_ylabel("Correlación de Pearson ($r$)", fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.4)

# GRÁFICA 2: Sensibilidad al Threshold
sens_t_plot = (
    results_df[(results_df["w1"] == 0.30) & (results_df["alpha_l2"] == 0.50)]
    .groupby("threshold")["pearson_r"]
    .max()
    .reset_index()
)

axes[1].plot(
    sens_t_plot["threshold"],
    sens_t_plot["pearson_r"],
    marker="s",
    linewidth=2.5,
    markersize=8,
    color="#3498db",
)
axes[1].axvline(
    x=0.75,
    color="red",
    linestyle="--",
    alpha=0.7,
    label="Umbral Elegido ($\\tau$=0.75)",
)

axes[1].set_title(
    "Efecto del Umbral de Escalación ($\\tau$)", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Umbral de Activación de Capa 3 ($\\tau$)", fontsize=11)
axes[1].set_ylabel("Correlación de Pearson ($r$)", fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plot_path = os.path.join(VISUALIZATION_DIR, "hace_ft_sensitivity_analysis.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Gráfica guardada con éxito en: {plot_path}")
print("=" * 65)
