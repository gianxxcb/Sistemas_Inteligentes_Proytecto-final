"""
QuantLab ML — Módulo 05: Specialist Models (Modelos Especialistas de Trading).
Clase base abstracta y contrato de salida estándar para todos los expertos.

Principios metodológicos:
1. Coherencia del playbook: setup, disparador, confirmación, invalidación, objetivo.
2. Versiones comparativas actualmente implementadas:
   - V1: Reglas fijas tradicionales.
   - V2: Meta-labeling con Machine Learning (Gradient Boosting / Regresión de cuantiles).
   V3 requiere una arquitectura temporal y permanece sin implementar.
3. Contrato de salida unificado para la Red de Compuertas (Módulo 06).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


class BaseExpert(ABC):
    """
    Clase base para todos los expertos especialistas de trading.
    """

    def __init__(
        self,
        name: str,
        horizon_days: int = 14,
        version: str = "V2",
        random_state: int = 42,
    ):
        self.name = name
        self.horizon_days = horizon_days
        self.version = version.upper()
        if self.version not in ("V1", "V2"):
            raise NotImplementedError(
                f"{self.version} no está implementada; V3 requiere un modelo temporal."
            )
        self.random_state = random_state

        self.classifier = None
        self.regressor_p10 = None
        self.regressor_p50 = None
        self.regressor_p90 = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_fitted = False
        self.target_unit = "forward_return_fraction"

    @abstractmethod
    def evaluate_playbook(self, row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evalúa las reglas específicas del playbook sobre una observación.

        Returns:
            Dict con:
            - estado: 'inactivo', 'setup', o 'disparado'
            - aplicabilidad: float [0, 1]
            - direccion: 'largo', 'corto', o 'neutral'
            - invalidacion: float (nivel relativo de stop)
            - objetivo: float (nivel relativo de profit)
            - traza_reglas: List[str]
        """
        pass

    @abstractmethod
    def get_confirmation_feature_names(self) -> List[str]:
        """Devuelve la lista de nombres de variables de confirmación propias de la técnica."""
        pass

    def _extract_playbook_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Filtra las observaciones candidatas (setup / disparado) para el meta-labeling.
        Utiliza diccionarios nativos para máxima velocidad de ejecución y robustez numérica.
        """
        target_col = f"fwd_return_{self.horizon_days}d"
        feat_cols = self.get_confirmation_feature_names()

        records_list = df.to_dict("records") if isinstance(df, pd.DataFrame) else df
        records = []
        X_list = []
        y_success_list = []
        y_return_list = []

        for row in records_list:
            pb = self.evaluate_playbook(row)
            if pb["estado"] in ("setup", "disparado"):
                ret = row.get(target_col, None)
                if ret is not None and not np.isnan(ret):
                    threshold = row.get(f"fwd_threshold_{self.horizon_days}d", 0.015)
                    if threshold is None or pd.isna(threshold):
                        threshold = 0.015
                    threshold = float(threshold)
                    if pb["direccion"] == "largo":
                        success = 1 if ret > threshold else 0
                    elif pb["direccion"] == "corto":
                        success = 1 if ret < -threshold else 0
                    else:
                        success = 1 if abs(ret) > threshold else 0

                    x_vals = []
                    for col in feat_cols:
                        v = row.get(col, 0.0)
                        x_vals.append(0.0 if v is None or np.isnan(v) else float(v))

                    X_list.append(x_vals)
                    y_success_list.append(success)
                    y_return_list.append(float(ret))
                    records.append(pb)

        X = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0) if X_list else np.empty((0, len(feat_cols)), dtype=np.float32)
        y_succ = np.array(y_success_list, dtype=np.int32) if y_success_list else np.empty((0,), dtype=np.int32)
        y_ret = np.array(y_return_list, dtype=np.float32) if y_return_list else np.empty((0,), dtype=np.float32)
        return X, y_succ, y_ret, records

    def fit(
        self,
        df_train: pd.DataFrame,
        calibration_df: Optional[pd.DataFrame] = None,
    ) -> "BaseExpert":
        """
        Entrena el modelo secundario de meta-labeling sobre los eventos candidatos del playbook.
        """
        self.target_unit = "forward_return_fraction"
        if self.version == "V1":
            self.is_fitted = True
            return self

        X, y_succ, y_ret, _ = self._extract_playbook_dataset(df_train)

        if len(X) < 30 or len(np.unique(y_succ)) < 2:
            self.is_fitted = True
            return self

        if self.version == "V2":
            self.classifier = GradientBoostingClassifier(
                n_estimators=35,
                max_depth=3,
                learning_rate=0.08,
                random_state=self.random_state,
            ).fit(X, y_succ)

            self.regressor_p10 = GradientBoostingRegressor(
                loss="quantile", alpha=0.10, n_estimators=25, max_depth=3, random_state=self.random_state
            ).fit(X, y_ret)
            self.regressor_p50 = GradientBoostingRegressor(
                loss="quantile", alpha=0.50, n_estimators=25, max_depth=3, random_state=self.random_state
            ).fit(X, y_ret)
            self.regressor_p90 = GradientBoostingRegressor(
                loss="quantile", alpha=0.90, n_estimators=25, max_depth=3, random_state=self.random_state
            ).fit(X, y_ret)

        self.is_fitted = True
        if calibration_df is not None and self.classifier is not None:
            X_cal, y_cal, _, _ = self._extract_playbook_dataset(calibration_df)
            if len(y_cal) >= 50 and len(np.unique(y_cal)) == 2:
                raw_probabilities = self.classifier.predict_proba(X_cal)[:, 1]
                self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                    raw_probabilities, y_cal
                )
        return self

    def _extract_episode_dataset(
        self, episodes: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        required = {"episode_return_R", "outcome_status", "playbook_state"}
        missing = required.difference(episodes.columns)
        if missing:
            raise ValueError(f"Faltan resultados del replay: {sorted(missing)}")
        feature_cols = self.get_confirmation_feature_names()
        missing_features = set(feature_cols).difference(episodes.columns)
        if missing_features:
            raise ValueError(f"Faltan features del experto: {sorted(missing_features)}")

        valid = (
            episodes["outcome_status"].eq("resolved")
            & episodes["playbook_state"].eq("disparado")
            & pd.to_numeric(episodes["episode_return_R"], errors="coerce").notna()
        )
        selected = episodes.loc[valid]
        if selected.empty:
            return (
                np.empty((0, len(feature_cols)), dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        X = selected[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        returns_r = selected["episode_return_R"].to_numpy(dtype=np.float32)
        labels = (returns_r > 0.0).astype(np.int32)
        if "sample_weight" in selected:
            weights = pd.to_numeric(selected["sample_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            weights = np.ones(len(selected), dtype=float)
        if (weights < 0).any() or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("sample_weight debe contener valores finitos no negativos")
        return X, labels, returns_r, weights

    def fit_episodes(
        self,
        train_episodes: pd.DataFrame,
        calibration_episodes: Optional[pd.DataFrame] = None,
    ) -> "BaseExpert":
        """Train V2 only on completed replay outcomes expressed in risk units."""
        if self.version != "V2":
            raise ValueError("fit_episodes está reservado para expertos V2")
        X, y_success, returns_r, weights = self._extract_episode_dataset(train_episodes)
        if len(X) < 30 or len(np.unique(y_success)) < 2:
            raise ValueError(
                f"{self.name}: se requieren al menos 30 episodios resueltos y ambas clases; "
                f"se encontraron {len(X)}"
            )

        self.classifier = GradientBoostingClassifier(
            n_estimators=35,
            max_depth=3,
            learning_rate=0.08,
            random_state=self.random_state,
        ).fit(X, y_success, sample_weight=weights)
        self.regressor_p10 = GradientBoostingRegressor(
            loss="quantile", alpha=0.10, n_estimators=25, max_depth=3,
            random_state=self.random_state,
        ).fit(X, returns_r, sample_weight=weights)
        self.regressor_p50 = GradientBoostingRegressor(
            loss="quantile", alpha=0.50, n_estimators=25, max_depth=3,
            random_state=self.random_state,
        ).fit(X, returns_r, sample_weight=weights)
        self.regressor_p90 = GradientBoostingRegressor(
            loss="quantile", alpha=0.90, n_estimators=25, max_depth=3,
            random_state=self.random_state,
        ).fit(X, returns_r, sample_weight=weights)
        self.calibrator = None
        self.target_unit = "R"

        if calibration_episodes is not None:
            X_cal, y_cal, _, _ = self._extract_episode_dataset(calibration_episodes)
            if len(y_cal) >= 50 and len(np.unique(y_cal)) == 2:
                raw_probabilities = self.classifier.predict_proba(X_cal)[:, 1]
                self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                    raw_probabilities, y_cal
                )
        self.is_fitted = True
        return self

    def evaluate_episodes(self, episodes: pd.DataFrame) -> Dict[str, Any]:
        """Report completed replay outcomes in R, not forward-close returns."""
        evaluated = episodes
        if "operated" in evaluated:
            evaluated = evaluated[evaluated["operated"].astype(bool)]
        X, y_success, returns_r, _ = self._extract_episode_dataset(evaluated)
        if not len(X):
            return {
                "expert": self.name,
                "version": self.version,
                "n_episodes": 0,
                "hit_rate": None,
                "mean_R": None,
                "profit_factor_R": None,
                "brier_score": None,
                "p_exito_calibrado": False,
            }
        gains = float(np.sum(returns_r[returns_r > 0]))
        losses = float(np.abs(np.sum(returns_r[returns_r < 0])))
        result = {
            "expert": self.name,
            "version": self.version,
            "n_episodes": int(len(returns_r)),
            "hit_rate": float(np.mean(y_success)),
            "mean_R": float(np.mean(returns_r)),
            "profit_factor_R": gains / losses if losses > 0 else None,
            "p_exito_calibrado": self.calibrator is not None,
            "mean_R_ci": None,
            "brier_score": None,
        }
        if "decision_date" in evaluated:
            from src.validation.splits import block_bootstrap_mean_interval

            outcomes = evaluated.loc[
                evaluated["outcome_status"].eq("resolved")
                & evaluated["playbook_state"].eq("disparado")
                & pd.to_numeric(evaluated["episode_return_R"], errors="coerce").notna()
            ]
            if len(outcomes) >= 2:
                result["mean_R_ci"] = block_bootstrap_mean_interval(
                    outcomes, value_col="episode_return_R", date_col="decision_date"
                )
        if self.classifier is not None and self.is_fitted:
            probabilities = self._calibrate_probabilities(
                self.classifier.predict_proba(X)[:, 1]
            )
            result["brier_score"] = float(np.mean((probabilities - y_success) ** 2))
        return result

    def _calibrate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return probabilities
        return np.asarray(self.calibrator.predict(probabilities), dtype=float)

    def predict_fitted_distribution(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score calibrated success and return quantiles for a batch of V2 candidates."""
        if self.classifier is None or self.regressor_p10 is None:
            raise ValueError("Se requiere un especialista V2 ajustado")
        probabilities = self._calibrate_probabilities(self.classifier.predict_proba(features)[:, 1])
        quantiles = np.column_stack([
            regressor.predict(features)
            for regressor in (self.regressor_p10, self.regressor_p50, self.regressor_p90)
        ])
        return probabilities, np.sort(quantiles, axis=1)

    def _break_even_probability(self, row: Union[pd.Series, Dict[str, Any]], pb: Dict[str, Any]) -> Optional[float]:
        cost_in_r = row.get("cost_in_r")
        risk = abs(float(pb["invalidacion"]))
        reward = abs(float(pb["objetivo"]))
        if cost_in_r is None or pd.isna(cost_in_r) or risk <= 0 or reward <= 0:
            return None
        cost_in_r = float(cost_in_r)
        if cost_in_r < 0:
            raise ValueError("cost_in_r no puede ser negativo")
        payoff_r = reward / risk
        return (1.0 + cost_in_r) / (payoff_r + 1.0)

    def _conditional_plan(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        pb: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if pb["estado"] == "inactivo":
            return None
        return {
            "entrada": {"tipo": "apertura_siguiente"},
            "direccion": pb["direccion"],
            "invalidacion": float(pb["invalidacion"]),
            "objetivo": float(pb["objetivo"]),
            "tiempo_maximo_dias": self.horizon_days,
            "p_equilibrio": self._break_even_probability(row, pb),
        }

    def predict_instance(self, row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera el contrato de salida estándar para una observación individual.
        """
        pb = self.evaluate_playbook(row)
        estado = pb["estado"]
        aplicabilidad = pb["aplicabilidad"]
        direccion = pb["direccion"]
        invalidacion = pb["invalidacion"]
        objetivo = pb["objetivo"]
        traza = pb["traza_reglas"]

        if estado == "inactivo":
            return {
                "experto": self.name,
                "version": self.version,
                "estado": "inactivo",
                "aplicabilidad": 0.0,
                "direccion": "neutral",
                "p_exito": 0.50,
                "cuantiles_retorno": {"p10": -0.05, "p50": 0.0, "p90": 0.05},
                "invalidacion": 0.0,
                "objetivo": 0.0,
                "incertidumbre": 1.0,
                "traza_reglas": traza,
                "plan": None,
                "espera": {
                    "dias_maximos": self.horizon_days,
                    "politica": "revision diaria con plan fijo; sin politica aprendida",
                },
                "p_equilibrio": None,
                "p_exito_calibrado": self.calibrator is not None,
                "unidad_cuantiles": self.target_unit,
            }

        # Valores base de V1
        p_exito = 0.55 if estado == "disparado" else 0.50
        p10 = -0.04
        p50 = 0.01 if direccion == "largo" else (-0.01 if direccion == "corto" else 0.0)
        p90 = 0.06
        incertidumbre = 0.50

        # Meta-labeling V2
        if self.version == "V2" and self.is_fitted and self.classifier is not None:
            feat_cols = self.get_confirmation_feature_names()
            raw_vals = [float(row.get(c, 0.0) or 0.0) for c in feat_cols]
            x_vals = np.nan_to_num(np.array([raw_vals], dtype=np.float32), nan=0.0)
            if self.regressor_p10 is not None:
                probabilities, quantiles = self.predict_fitted_distribution(x_vals)
                p_exito = float(probabilities[0])
                p10, p50, p90 = quantiles[0]
            else:
                probabilities = self.classifier.predict_proba(x_vals)[:, 1]
                p_exito = float(self._calibrate_probabilities(probabilities)[0])

            incertidumbre = float(1.0 - 2.0 * abs(p_exito - 0.5))

        return {
            "experto": self.name,
            "version": self.version,
            "estado": estado,
            "aplicabilidad": round(float(aplicabilidad), 4),
            "direccion": direccion,
            "p_exito": round(float(p_exito), 4),
            "cuantiles_retorno": {
                "p10": round(float(p10), 4),
                "p50": round(float(p50), 4),
                "p90": round(float(p90), 4),
            },
            "invalidacion": round(float(invalidacion), 4),
            "objetivo": round(float(objetivo), 4),
            "incertidumbre": round(float(incertidumbre), 4),
            "traza_reglas": traza,
            "plan": self._conditional_plan(row, pb),
            "espera": {
                "dias_maximos": self.horizon_days,
                "politica": "revision diaria con plan fijo; sin politica aprendida",
            },
            "p_equilibrio": self._break_even_probability(row, pb),
            "p_exito_calibrado": self.calibrator is not None,
            "unidad_cuantiles": self.target_unit,
        }

    def evaluate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evalúa el desempeño integral del experto de forma vectorizada y optimizada.
        """
        target_col = f"fwd_return_{self.horizon_days}d"
        records_list = df.to_dict("records") if isinstance(df, pd.DataFrame) else df

        candidate_pbs = []
        candidate_rets = []
        candidate_thresholds = []

        for row in records_list:
            pb = self.evaluate_playbook(row)
            if pb["estado"] in ("setup", "disparado"):
                ret = row.get(target_col, None)
                if ret is not None and not np.isnan(ret):
                    candidate_pbs.append(pb)
                    candidate_rets.append(float(ret))
                    threshold = row.get(f"fwd_threshold_{self.horizon_days}d", 0.015)
                    if threshold is None or pd.isna(threshold):
                        threshold = 0.015
                    candidate_thresholds.append(float(threshold))

        n_events = len(candidate_pbs)
        if n_events == 0:
            return {
                "experto": self.name,
                "version": self.version,
                "n_eventos": 0,
                "hit_rate": 0.50,
                "retorno_medio": 0.0,
                "profit_factor": 1.0,
            }

        # Esta métrica es una evaluación descriptiva de candidatos; el replay decidirá la ejecución.
        strat_returns = []
        hits = 0
        total_executed = 0

        for i, pb in enumerate(candidate_pbs):
            ret = candidate_rets[i]
            threshold = candidate_thresholds[i]
            dir_str = pb["direccion"]

            total_executed += 1
            if dir_str == "largo":
                strat_returns.append(ret)
                if ret > threshold:
                    hits += 1
            elif dir_str == "corto":
                strat_returns.append(-ret)
                if ret < -threshold:
                    hits += 1
            else:  # Neutral/volatilidad
                strat_returns.append(abs(ret))
                if abs(ret) > threshold:
                    hits += 1

        if total_executed == 0:
            total_executed = n_events
            strat_returns = candidate_rets
            hits = sum(1 for r in candidate_rets if r > 0)

        hit_rate = hits / total_executed if total_executed > 0 else 0.50
        strat_ret_arr = np.array(strat_returns, dtype=np.float32)
        mean_ret = float(np.mean(strat_ret_arr))

        gains = np.sum(strat_ret_arr[strat_ret_arr > 0])
        losses = np.abs(np.sum(strat_ret_arr[strat_ret_arr < 0]))
        profit_factor = float(gains / (losses + 1e-6)) if losses > 0 else (2.5 if gains > 0 else 1.0)

        return {
            "experto": self.name,
            "version": self.version,
            "n_eventos": total_executed,
            "hit_rate": round(float(hit_rate), 4),
            "retorno_medio": round(float(mean_ret), 4),
            "profit_factor": round(float(profit_factor), 2),
        }
