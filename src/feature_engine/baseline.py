"""
Modelos de referencia para la evaluación de la Fase 2.

Según Reformulacion/08_evaluacion_validacion.md:
  - División cronológica: entrenamiento 60 % / validación 20 % / prueba 20 % (por tiempo, sin barajar)
  - Tres modelos de referencia:
      1. Probabilidad incondicional (frecuencia histórica de clases)
      2. Regresión logística (conjunto pequeño de características)
      3. Clasificador LightGBM / Gradient Boosting
  - Métricas: Log Loss, exactitud, informe de clasificación y diagrama de confiabilidad
  - Los escaladores se ajustan solo con los datos de entrenamiento

Objetivo: clase de rendimiento futuro a 7 días (1 semana; 3 clases: UP +1 / LATERAL 0 / DOWN -1).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

# LightGBM es opcional: se usa sklearn GradientBoosting como alternativa si no está disponible
try:
    import lightgbm as lgb

    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier

    _HAS_LGBM = False
    warnings.warn(
        "LightGBM no está instalado; se usará sklearn GradientBoostingClassifier como alternativa.",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

TARGET_COL = "fwd_class_7d"

# Características clave para los modelos de referencia (subconjunto siempre disponible)
# Se eligen porque abarcan varias familias y son robustas.
KEY_FEATURES = [
    "log_return",
    "rsi_14",
    "macd_histogram",
    "atr_relative",
    "volume_relative_20",
    "dist_to_sma_20",
    "bollinger_pctb",
    "roc_10",
    "ctx_volatility_90d",
    "ctx_corr_btc_60d",
]


# ---------------------------------------------------------------------------
# Carga y división de datos
# ---------------------------------------------------------------------------

def load_features(path: str | Path) -> pd.DataFrame:
    """Carga el CSV consolidado de características y lo prepara para el modelado.

    Parámetros
    ----------
    path : str or Path
        Ruta a ``features_consolidated.csv``.

    Devuelve
    -------
    pd.DataFrame
        Ordenado por fecha, con ``date`` en formato datetime.
    """
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    horizon_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide los datos cronológicamente en entrenamiento / validación / prueba.

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de datos completo, ordenado por fecha.
    date_col : str
        Columna con valores de fecha y hora.
    train_frac : float
        Fracción para entrenamiento (predeterminado: 0.60).
    val_frac : float
        Fracción para validación (predeterminado: 0.20).
        El resto corresponde a prueba.
    horizon_days : int
        Si es positivo, elimina muestras cuyo objetivo futuro atraviese un corte.

    Devuelve
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) sin superposición de fechas.
    """
    dates = df[date_col].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * train_frac) - 1]
    val_end = dates[int(n * (train_frac + val_frac)) - 1]

    if horizon_days < 0:
        raise ValueError("horizon_days no puede ser negativo")
    target_end = df[date_col] + pd.Timedelta(days=horizon_days)
    train = df[(df[date_col] <= train_end) & (target_end <= train_end)].copy()
    val = df[
        (df[date_col] > train_end)
        & (df[date_col] <= val_end)
        & (target_end <= val_end)
    ].copy()
    test = df[(df[date_col] > val_end) & (target_end <= dates[-1])].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Funciones auxiliares para seleccionar características
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Devuelve el subconjunto de *feature_cols* que existe en *df*."""
    available = [c for c in feature_cols if c in df.columns]
    return available


def prepare_Xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extrae X e y y descarta las filas donde el objetivo o alguna característica sea NaN."""
    cols = feature_cols + [target_col]
    subset = df[cols].dropna()
    X = subset[feature_cols]
    y = subset[target_col].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Modelo de referencia 1: probabilidad incondicional
# ---------------------------------------------------------------------------

class UnconditionalBaseline:
    """Predice la frecuencia histórica de clases para cada muestra."""

    def __init__(self) -> None:
        self.class_probs_: dict[int, float] = {}
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UnconditionalBaseline":
        counts = y.value_counts(normalize=True).sort_index()
        self.classes_ = counts.index.values
        self.class_probs_ = counts.to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        probs = np.array([self.class_probs_.get(c, 0.0) for c in self.classes_])
        return np.tile(probs, (n, 1))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Entrenamiento de modelos
# ---------------------------------------------------------------------------

def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler: StandardScaler | None = None,
) -> dict[str, Any]:
    """Entrena los tres modelos de referencia.

    Parámetros
    ----------
    X_train : pd.DataFrame
        Características de entrenamiento (ya seleccionadas, sin NaN).
    y_train : pd.Series
        Etiquetas de entrenamiento.
    scaler : StandardScaler or None
        Si se proporciona, se usa para transformar X. De lo contrario, se ajusta uno nuevo.

    Devuelve
    -------
    dict
        ``{"unconditional": model, "logistic": model, "gbm": model,
           "scaler": scaler}``
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        X_scaled = scaler.transform(X_train)

    # 1. Unconditional
    uncond = UnconditionalBaseline().fit(X_train, y_train)

    # 2. Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    lr.fit(X_scaled, y_train)

    # 3. Gradient Boosting
    if _HAS_LGBM:
        gbm = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        gbm.fit(X_scaled, y_train)
    else:
        gbm = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=50,
            subsample=0.8,
            random_state=42,
        )
        gbm.fit(X_scaled, y_train)

    return {
        "unconditional": uncond,
        "logistic": lr,
        "gbm": gbm,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scaler: StandardScaler,
    model_name: str,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Evalúa un modelo en un conjunto de datos.

    Devuelve
    -------
    dict
        Claves: model_name, accuracy, log_loss, classification_report,
                reliability (frecuencia predicha frente a real por clase).
    """
    if model_name == "unconditional":
        X_input = X
    else:
        X_input = scaler.transform(X)

    y_pred = model.predict(X_input) if model_name == "unconditional" else model.predict(X_input)
    y_proba = model.predict_proba(X_input)

    # Asegurar que las columnas de probabilidad coincidan con las clases esperadas
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    else:
        model_classes = classes

    acc = accuracy_score(y, y_pred)
    try:
        ll = log_loss(y, y_proba, labels=classes)
    except ValueError:
        ll = float("nan")

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    # Datos sencillos para el diagrama de confiabilidad: comparar, por clase,
    # la probabilidad predicha con la frecuencia observada en intervalos decílicos.
    reliability = _reliability_data(y, y_proba, model_classes)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "log_loss": ll,
        "classification_report": report,
        "reliability": reliability,
    }


def _reliability_data(
    y_true: pd.Series,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[dict]]:
    """Calcula los datos de confiabilidad / calibración por clase."""
    result = {}
    for i, cls in enumerate(classes):
        probs = y_proba[:, i]
        actual = (y_true.values == cls).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_data = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bin_data.append({
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "mean_predicted": float(probs[mask].mean()),
                    "mean_actual": float(actual[mask].mean()),
                    "count": int(mask.sum()),
                })
        result[str(cls)] = bin_data
    return result


def evaluate_all(
    models: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classes: np.ndarray,
) -> dict[str, list[dict]]:
    """Evalúa todos los modelos de referencia en los conjuntos de validación y prueba.

    Devuelve
    -------
    dict
        ``{"val": [results...], "test": [results...]}``
    """
    scaler = models["scaler"]
    results: dict[str, list[dict]] = {"val": [], "test": []}
    for name in ["unconditional", "logistic", "gbm"]:
        model = models[name]
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            res = evaluate_model(model, X, y, scaler, name, classes)
            res["split"] = split_name
            results[split_name].append(res)
    return results


# ---------------------------------------------------------------------------
# Generación del informe
# ---------------------------------------------------------------------------

def generate_report(
    results: dict[str, list[dict]],
    train_info: dict[str, Any],
    output_path: str | Path,
) -> str:
    """Genera un informe Markdown y lo guarda en disco.

    Parámetros
    ----------
    results : dict
        Resultado de ``evaluate_all``.
    train_info : dict
        ``{"n_train", "n_val", "n_test", "date_range_train",
           "date_range_val", "date_range_test", "features_used",
           "class_distribution_train"}``
    output_path : str or Path
        Ruta donde se guardará el informe.

    Devuelve
    -------
    str
        El informe como cadena.
    """
    lines = [
        "# Resultados de los modelos de referencia — Fase 2",
        "",
        "## División de datos (cronológica)",
        "",
        f"Las ventanas se purgan para que el objetivo a {train_info.get('purge_days', 0)} días no atraviese los cortes entre train, validación y prueba.",
        "",
        f"| Conjunto | Filas | Rango de fechas |",
        f"|-------|------|------------|",
        f"| Entrenamiento | {train_info['n_train']:,} | {train_info['date_range_train']} |",
        f"| Validación    | {train_info['n_val']:,} | {train_info['date_range_val']} |",
        f"| Prueba        | {train_info['n_test']:,} | {train_info['date_range_test']} |",
        "",
        f"**Objetivo:** `{TARGET_COL}` (3 clases: DOWN -1 / LATERAL 0 / UP +1)",
        "",
        "**Distribución de clases del entrenamiento:**",
        "",
    ]
    for cls, frac in train_info["class_distribution_train"].items():
        lines.append(f"- Clase {cls}: {frac:.1%}")
    lines.append("")

    lines.append(f"**Características utilizadas ({len(train_info['features_used'])}):** "
                 + ", ".join(f"`{f}`" for f in train_info["features_used"]))
    lines.append("")

    # Tablas de resultados
    for split in ["val", "test"]:
        lines.append(f"## Resultados del conjunto de {('validación' if split == 'val' else 'prueba')}")
        lines.append("")
        lines.append("| Modelo | Exactitud | Log Loss |")
        lines.append("|-------|----------|----------|")
        for r in results[split]:
            ll_str = f"{r['log_loss']:.4f}" if not np.isnan(r["log_loss"]) else "N/A"
            lines.append(
                f"| {r['model_name'].title()} | {r['accuracy']:.4f} | {ll_str} |"
            )
        lines.append("")

        # Informes de clasificación detallados
        for r in results[split]:
            lines.append(f"### {r['model_name'].title()} — Informe de clasificación ({'validación' if split == 'val' else 'prueba'})")
            lines.append("")
            lines.append("```")
            cr = r["classification_report"]
            lines.append(f"{'':>15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
            for key, vals in cr.items():
                if isinstance(vals, dict):
                    lines.append(
                        f"{key:>15s} {vals.get('precision',0):>10.4f} "
                        f"{vals.get('recall',0):>10.4f} {vals.get('f1-score',0):>10.4f} "
                        f"{vals.get('support',0):>10.0f}"
                    )
                else:
                    lines.append(f"{key:>15s} {vals:>10.4f}")
            lines.append("```")
            lines.append("")

    # Resumen de confiabilidad
    lines.append("## Resumen de confiabilidad / calibración")
    lines.append("")
    lines.append("Para cada modelo y clase, se compara la probabilidad predicha con "
                 "la frecuencia observada en distintos intervalos de probabilidad. "
                 "Un modelo bien calibrado presenta valores predichos ≈ observados en cada intervalo.")
    lines.append("")
    lines.append("*(Los datos detallados de confiabilidad se guardaron junto con los resultados.)*")
    lines.append("")

    report = "\n".join(lines)

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def run_baseline(
    features_path: str | Path,
    report_path: str | Path = "reports/baseline/baseline_results.md",
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Ejecuta todo el flujo de evaluación de los modelos de referencia.

    Parámetros
    ----------
    features_path : str or Path
        Path to ``features_consolidated.csv``.
    report_path : str or Path
        Ruta de salida del informe Markdown.
    feature_cols : list[str] or None
        Reemplaza las columnas de características. Por defecto usa KEY_FEATURES.

    Devuelve
    -------
    dict
        Diccionario completo de resultados con modelos, evaluación e informe.
    """
    if feature_cols is None:
        feature_cols = KEY_FEATURES

    print("=" * 60)
    print("EVALUACIÓN DE MODELOS DE REFERENCIA")
    print("=" * 60)

    # Cargar
    print(f"\n[1/5] Cargando características desde {features_path} ...")
    df = load_features(features_path)
    print(f"  → {len(df):,} filas, {len(df.columns)} columnas")

    # Seleccionar las características disponibles
    available_features = select_features(df, feature_cols)
    print(f"\n[2/5] Se seleccionaron {len(available_features)} características: {available_features}")

    if len(available_features) == 0:
        raise ValueError(
            "No features available for baseline.  Check that feature names match."
        )

    # Dividir
    print("\n[3/5] División cronológica purgada a 7 días (60/20/20) ...")
    train, val, test = chronological_split(df, horizon_days=7)
    X_train, y_train = prepare_Xy(train, available_features)
    X_val, y_val = prepare_Xy(val, available_features)
    X_test, y_test = prepare_Xy(test, available_features)

    classes = np.array(sorted(y_train.unique()))
    print(f"  Entrenamiento: {len(X_train):,} filas  ({train['date'].min().date()} — {train['date'].max().date()})")
    print(f"  Validación: {len(X_val):,} filas  ({val['date'].min().date()} — {val['date'].max().date()})")
    print(f"  Prueba: {len(X_test):,} filas  ({test['date'].min().date()} — {test['date'].max().date()})")
    print(f"  Clases: {classes}")
    print(f"  Distribución del entrenamiento: {y_train.value_counts(normalize=True).sort_index().to_dict()}")

    # Entrenar
    print("\n[4/5] Entrenando modelos de referencia ...")
    models = train_baselines(X_train, y_train)
    print("  ✓ Incondicional (frecuencia histórica)")
    print("  ✓ Regresión logística")
    print(f"  ✓ {'LightGBM' if _HAS_LGBM else 'GradientBoosting'}")

    # Evaluar
    print("\n[5/5] Evaluando ...")
    results = evaluate_all(models, X_val, y_val, X_test, y_test, classes)

    # Imprimir el resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    for split in ["val", "test"]:
        print(f"\n  CONJUNTO DE {'VALIDACIÓN' if split == 'val' else 'PRUEBA'}:")
        for r in results[split]:
            ll_str = f"{r['log_loss']:.4f}" if not np.isnan(r["log_loss"]) else "N/A"
            print(f"    {r['model_name']:20s}  Exactitud={r['accuracy']:.4f}  LogLoss={ll_str}")

    # Informe
    train_info = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "date_range_train": f"{train['date'].min().date()} — {train['date'].max().date()}",
        "date_range_val": f"{val['date'].min().date()} — {val['date'].max().date()}",
        "date_range_test": f"{test['date'].min().date()} — {test['date'].max().date()}",
        "features_used": available_features,
        "class_distribution_train": y_train.value_counts(normalize=True).sort_index().to_dict(),
        "purge_days": 7,
    }
    report = generate_report(results, train_info, report_path)
    print(f"\n  Informe guardado en {report_path}")

    return {
        "models": models,
        "results": results,
        "report_path": str(report_path),
        "train_info": train_info,
    }
