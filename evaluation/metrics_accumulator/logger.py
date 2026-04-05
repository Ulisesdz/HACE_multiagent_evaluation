import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json


class MetricsLogger:
    """
    Acumulador de métricas de evaluación para análisis posterior

    Guarda automáticamente en CSVs separados:
    - online_metrics.csv: Evaluaciones desde Streamlit
    - offline_metrics.csv: Evaluaciones desde run_eval.py
    """

    # Headers ordenados
    FIELDNAMES = [
        "timestamp",
        "source",
        "query_id",
        "user_query",
        "num_tasks",
        "difficulty",
        "category",
        # Baseline Metrics
        "baseline_score",
        "baseline_routing_f1",
        "baseline_numeric_f1",
        "baseline_hallucination_rate",
        "baseline_task_coverage",
        "baseline_sql_correctness",
        "baseline_time",
        # LLM-Judge Metrics
        "llm_judge_overall",
        "llm_judge_planner",
        "llm_judge_supervisor",
        "llm_judge_agents",
        "llm_judge_final",
        "llm_judge_error_category",
        "llm_judge_time",
        # MACE Metrics
        "mace_score",
        "mace_quality",
        "mace_confidence",
        "mace_layer1",
        "mace_layer2",
        "mace_layer3",
        "mace_layer3_used",
        "mace_time",
        # Metadata
        "critical_failures",
        "raw_trace",
    ]

    def __init__(self, output_dir: str = "evaluation/accumulated_data"):
        """
        Inicializar logger de métricas

        Args:
            output_dir: Directorio donde guardar los CSVs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.online_csv_path = self.output_dir / "online_metrics.csv"
        self.offline_csv_path = self.output_dir / "offline_metrics.csv"

        # Inicializar CSVs con headers
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        """Crear archivos CSV con headers si no existen"""
        for csv_path in [self.online_csv_path, self.offline_csv_path]:
            if not csv_path.exists():
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()

    def _generate_query_id(self) -> str:
        """Generar ID único para query"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _append_to_csv(self, record: Dict[str, Any], csv_path: Path):
        """
        Añadir registro al CSV

        Args:
            record: Dict con todos los campos (en cualquier orden)
            csv_path: Path al archivo CSV
        """
        # Asegurar que el record tiene todos los campos (rellenar con None si falta)
        complete_record = {field: record.get(field, None) for field in self.FIELDNAMES}

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(complete_record)

    def log_online_evaluation(
        self,
        trace_data: Dict[str, Any],
        baseline_eval: Optional[Any] = None,
        llm_judge_data: Optional[Dict[str, Any]] = None,
        hybrid_eval: Optional[Dict[str, Any]] = None,
    ):
        """
        Registrar evaluación online (desde app.py)

        Args:
            trace_data: Dict con trazas del sistema
            baseline_eval: Resultado de evaluate_baseline (opcional)
            llm_judge_data: Dict con evaluación LLM-Judge (opcional)
            hybrid_eval: Dict resultado de HybridEvaluator.evaluate() (opcional)
        """
        query_id = self._generate_query_id()
        timestamp = datetime.now().isoformat()

        record = {
            "query_id": query_id,
            "timestamp": timestamp,
            "source": "online",
            "user_query": trace_data.get("user_question", ""),
            "num_tasks": len(trace_data.get("planner_tasks", [])),
            "difficulty": "",  # No disponible en online
            "category": "",  # No disponible en online
        }

        # 1. Baseline Metrics
        if baseline_eval:
            record.update(
                {
                    "baseline_score": float(baseline_eval.baseline_score),
                    "baseline_routing_f1": float(baseline_eval.routing_metrics.f1),
                    "baseline_numeric_f1": float(baseline_eval.numeric_metrics.f1),
                    "baseline_hallucination_rate": float(
                        baseline_eval.numeric_metrics.hallucination_rate
                    ),
                    "baseline_task_coverage": float(
                        baseline_eval.task_coverage_metrics.coverage
                    ),
                    "baseline_sql_correctness": float(
                        baseline_eval.sql_metrics.correctness
                    ),
                    "baseline_time": float(baseline_eval.evaluation_time),
                }
            )
        else:
            record.update(
                {
                    k: None
                    for k in [
                        "baseline_score",
                        "baseline_routing_f1",
                        "baseline_numeric_f1",
                        "baseline_hallucination_rate",
                        "baseline_task_coverage",
                        "baseline_sql_correctness",
                        "baseline_time",
                    ]
                }
            )

        # 2. LLM-Judge Metrics
        if llm_judge_data:
            comprehensive = llm_judge_data["comprehensive_eval"]
            record.update(
                {
                    "llm_judge_overall": float(comprehensive.overall_score),
                    "llm_judge_planner": float(llm_judge_data["planner_score"]),
                    "llm_judge_supervisor": float(llm_judge_data["supervisor_score"]),
                    "llm_judge_agents": float(llm_judge_data["agents_avg_score"]),
                    "llm_judge_final": float(llm_judge_data["final_output_score"]),
                    "llm_judge_error_category": str(comprehensive.error_category),
                    "llm_judge_time": float(llm_judge_data.get("elapsed_time", 0)),
                }
            )
        else:
            record.update(
                {
                    k: None
                    for k in [
                        "llm_judge_overall",
                        "llm_judge_planner",
                        "llm_judge_supervisor",
                        "llm_judge_agents",
                        "llm_judge_final",
                        "llm_judge_error_category",
                        "llm_judge_time",
                    ]
                }
            )

        # 3. MACE (Híbrido)
        if hybrid_eval:
            record.update(
                {
                    "mace_score": float(hybrid_eval["final_score"]),
                    "mace_quality": str(hybrid_eval["quality_label"]),
                    "mace_confidence": str(hybrid_eval["confidence"]),
                    "mace_layer1": float(hybrid_eval["layer1_score"]),
                    "mace_layer2": float(hybrid_eval["layer2_score"]),
                    "mace_layer3": (
                        float(hybrid_eval["layer3_score"])
                        if hybrid_eval["layer3_score"] is not None
                        else None
                    ),
                    "mace_layer3_used": int(hybrid_eval["layer3_used"]),
                    "mace_time": float(hybrid_eval["evaluation_time"]),
                }
            )
        else:
            record.update(
                {
                    k: None
                    for k in [
                        "mace_score",
                        "mace_quality",
                        "mace_confidence",
                        "mace_layer1",
                        "mace_layer2",
                        "mace_layer3",
                        "mace_layer3_used",
                        "mace_time",
                    ]
                }
            )

        # 4. Metadata
        crit_fails = []
        if llm_judge_data:
            crit_fails.extend(llm_judge_data["comprehensive_eval"].critical_failures)
        if hybrid_eval:
            crit_fails.extend(hybrid_eval.get("critical_failures", []))

        record["critical_failures"] = json.dumps(list(set(crit_fails)))
        record["raw_trace"] = json.dumps(
            {
                "planner_tasks": trace_data.get("planner_tasks", []),
                "routing_trace": trace_data.get("routing_trace", []),
                "agent_executions": trace_data.get("agent_executions", []),
                "sql_queries": trace_data.get("sql_queries", []),
            }
        )

        # Guardar
        self._append_to_csv(record, self.online_csv_path)

    def log_offline_evaluation(
        self,
        test_case: Dict[str, Any],
        trace_data: Dict[str, Any],
        baseline_eval: Optional[Any] = None,
        llm_judge_data: Optional[Dict[str, Any]] = None,
        hybrid_eval: Optional[Dict[str, Any]] = None,
    ):
        """
        Registrar evaluación offline (desde run_eval.py)

        Args:
            test_case: Dict con metadata del caso de prueba
            trace_data: Dict con trazas del sistema
            baseline_eval: Resultado de evaluate_baseline (opcional)
            llm_judge_data: Dict con evaluación LLM-Judge (opcional)
            hybrid_eval: Dict resultado de HybridEvaluator.evaluate() (opcional)
        """
        timestamp = datetime.now().isoformat()

        record = {
            "timestamp": timestamp,
            "source": "offline",
            "query_id": test_case.get("id", ""),
            "user_query": test_case.get("query", ""),
            "num_tasks": len(trace_data.get("planner_tasks", [])),
            "difficulty": test_case.get("difficulty", ""),
            "category": test_case.get("category", ""),
        }

        # 1. Baseline Metrics
        if baseline_eval:
            record.update(
                {
                    "baseline_score": float(baseline_eval.baseline_score),
                    "baseline_routing_f1": float(baseline_eval.routing_metrics.f1),
                    "baseline_numeric_f1": float(baseline_eval.numeric_metrics.f1),
                    "baseline_hallucination_rate": float(
                        baseline_eval.numeric_metrics.hallucination_rate
                    ),
                    "baseline_task_coverage": float(
                        baseline_eval.task_coverage_metrics.coverage
                    ),
                    "baseline_sql_correctness": float(
                        baseline_eval.sql_metrics.correctness
                    ),
                    "baseline_time": float(baseline_eval.evaluation_time),
                }
            )
        else:
            record.update(
                {
                    k: None
                    for k in [
                        "baseline_score",
                        "baseline_routing_f1",
                        "baseline_numeric_f1",
                        "baseline_hallucination_rate",
                        "baseline_task_coverage",
                        "baseline_sql_correctness",
                        "baseline_time",
                    ]
                }
            )

        # 2. LLM-Judge Metrics
        if llm_judge_data:
            comprehensive = llm_judge_data["comprehensive_eval"]
            record.update(
                {
                    "llm_judge_overall": float(comprehensive.overall_score),
                    "llm_judge_planner": float(llm_judge_data["planner_score"]),
                    "llm_judge_supervisor": float(llm_judge_data["supervisor_score"]),
                    "llm_judge_agents": float(llm_judge_data["agents_avg_score"]),
                    "llm_judge_final": float(llm_judge_data["final_output_score"]),
                    "llm_judge_error_category": str(comprehensive.error_category),
                    "llm_judge_time": float(llm_judge_data.get("elapsed_time", 0)),
                }
            )
        else:
            record.update(
                {
                    k: None
                    for k in [
                        "llm_judge_overall",
                        "llm_judge_planner",
                        "llm_judge_supervisor",
                        "llm_judge_agents",
                        "llm_judge_final",
                        "llm_judge_error_category",
                        "llm_judge_time",
                    ]
                }
            )

        # 3. MACE (Híbrido)
        if hybrid_eval:
            record.update(
                {
                    "mace_score": float(hybrid_eval["final_score"]),
                    "mace_quality": str(hybrid_eval["quality_label"]),
                    "mace_confidence": str(hybrid_eval["confidence"]),
                    "mace_layer1": float(hybrid_eval["layer1_score"]),
                    "mace_layer2": float(hybrid_eval["layer2_score"]),
                    "mace_layer3": (
                        float(hybrid_eval["layer3_score"])
                        if hybrid_eval["layer3_score"] is not None
                        else None
                    ),
                    "mace_layer3_used": int(hybrid_eval["layer3_used"]),
                    "mace_time": float(hybrid_eval["evaluation_time"]),
                }
            )
        else:
            record.update(
                {
                    k: None
                    for k in [
                        "mace_score",
                        "mace_quality",
                        "mace_confidence",
                        "mace_layer1",
                        "mace_layer2",
                        "mace_layer3",
                        "mace_layer3_used",
                        "mace_time",
                    ]
                }
            )

        # 4. Metadata
        crit_fails = []
        if llm_judge_data:
            crit_fails.extend(llm_judge_data["comprehensive_eval"].critical_failures)
        if hybrid_eval:
            crit_fails.extend(hybrid_eval.get("critical_failures", []))

        record["critical_failures"] = json.dumps(list(set(crit_fails)))
        record["raw_trace"] = json.dumps(
            {
                "planner_tasks": trace_data.get("planner_tasks", []),
                "routing_trace": trace_data.get("routing_trace", []),
                "agent_executions": trace_data.get("agent_executions", []),
                "sql_queries": trace_data.get("sql_queries", []),
            }
        )

        # Guardar
        self._append_to_csv(record, self.offline_csv_path)

    def get_statistics(self, source: str = "all") -> Dict[str, Any]:
        """
        Calcular estadísticas agregadas de evaluaciones

        Args:
            source: 'online', 'offline', o 'all'

        Returns:
            Dict con estadísticas (promedios, counts, etc.)
        """
        # Cargar datos según source
        if source == "all":
            dfs = []
            if self.online_csv_path.exists():
                dfs.append(pd.read_csv(self.online_csv_path))
            if self.offline_csv_path.exists():
                dfs.append(pd.read_csv(self.offline_csv_path))

            if not dfs:
                return {"total_evaluations": 0}

            df = pd.concat(dfs, ignore_index=True)

        elif source == "online":
            if not self.online_csv_path.exists():
                return {"total_evaluations": 0}
            df = pd.read_csv(self.online_csv_path)

        elif source == "offline":
            if not self.offline_csv_path.exists():
                return {"total_evaluations": 0}
            df = pd.read_csv(self.offline_csv_path)

        else:
            raise ValueError(
                f"Invalid source: {source}. Must be 'online', 'offline', or 'all'"
            )

        if df.empty:
            return {"total_evaluations": 0}

        # Convertir columnas numéricas (por si acaso hay strings)
        numeric_cols = [
            "baseline_score",
            "llm_judge_overall",
            "mace_score",
            "baseline_hallucination_rate",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calcular estadísticas
        stats = {
            "total_evaluations": len(df),
            "baseline_avg_score": (
                float(df["baseline_score"].mean())
                if "baseline_score" in df and df["baseline_score"].notna().any()
                else None
            ),
            "llm_judge_avg_score": (
                float(df["llm_judge_overall"].mean())
                if "llm_judge_overall" in df and df["llm_judge_overall"].notna().any()
                else None
            ),
            "hybrid_avg_score": (
                float(df["mace_score"].mean())
                if "mace_score" in df and df["mace_score"].notna().any()
                else None
            ),
            "avg_hallucination_rate": (
                float(df["baseline_hallucination_rate"].mean())
                if "baseline_hallucination_rate" in df
                and df["baseline_hallucination_rate"].notna().any()
                else None
            ),
            "error_categories": (
                df["llm_judge_error_category"].value_counts().to_dict()
                if "llm_judge_error_category" in df
                else {}
            ),
            "date_range": {
                "first": df["timestamp"].min() if "timestamp" in df else None,
                "last": df["timestamp"].max() if "timestamp" in df else None,
            },
        }

        return stats
