"""FmaxPredictor: estimate achievable Fmax (and resources/routability) from
design features + knobs.

Used by the optimizer to rank candidate knob sets before spending real flow
time on them. Falls back to a transparent heuristic until enough corpus samples
exist to train a regressor, so cold-start ranking is still sensible.

The predictor grew three heads so the optimizer/reward can screen candidates on
more than clock speed:

* ``predict``          -> Fmax (MHz)
* ``predict_luts``     -> LUT usage (fit resource pressure before P&R)
* ``routed_probability`` -> P(design routes) from features+knobs

``evaluate`` reports honest cross-validated accuracy (MAE / R2 / within-10%)
plus feature importances, so we know whether the model is trustworthy rather
than training blind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from . import features as feat
from .backends.base import FlowOptions

MIN_SAMPLES = 12
DEFAULT_MODEL_PATH = Path("data/model.joblib")

# Ordered knob encoding appended to the design-feature vector.
_OPTION_NAMES = ("abc9", "retime", "dsp", "pipeline_output", "placer_heap")

VECTOR_NAMES: tuple[str, ...] = feat.FEATURE_NAMES + _OPTION_NAMES


def options_vector(options: FlowOptions) -> list[float]:
    return [
        1.0 if options.abc9 else 0.0,
        1.0 if options.retime else 0.0,
        1.0 if options.dsp else 0.0,
        1.0 if options.pipeline_output else 0.0,
        1.0 if options.placer == "heap" else 0.0,
    ]


def make_vector(features: dict[str, float], options: FlowOptions) -> list[float]:
    return feat.to_vector(features) + options_vector(options)


def _row_vector(rec: dict[str, Any]) -> list[float]:
    options = FlowOptions.from_dict(rec.get("options", {}))
    return make_vector(rec.get("features", {}), options)


@dataclass
class ModelReport:
    """Cross-validated accuracy of the Fmax head plus feature importances."""

    n: int
    trained: bool
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    within_10pct: float = 0.0
    baseline_mae: float = 0.0        # naive predict-the-mean MAE, for context
    importances: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def top_features(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.importances.items(), key=lambda kv: -kv[1])[:k]

    def summary(self) -> str:
        if not self.trained:
            return f"model: heuristic fallback ({self.n} samples < {MIN_SAMPLES}); {self.note}".strip()
        skill = (
            f"{(1 - self.mae / self.baseline_mae) * 100:.0f}% better than mean-baseline"
            if self.baseline_mae > 0
            else "n/a"
        )
        top = ", ".join(f"{n}={v:.2f}" for n, v in self.top_features(4))
        return (
            f"model: {self.n} samples | MAE {self.mae:.1f} MHz | R2 {self.r2:.2f} | "
            f"within-10% {self.within_10pct * 100:.0f}% | {skill}\n"
            f"  top features: {top}"
        )


@dataclass
class FmaxPredictor:
    model: Any = None            # fitted sklearn regressor for Fmax, or None
    lut_model: Any = None        # fitted regressor for LUT usage, or None
    route_model: Any = None      # fitted classifier for routability, or None
    trained: bool = False
    n_samples: int = 0

    # ------------------------------------------------------------------ #
    def fit(self, rows: list[dict[str, Any]]) -> "FmaxPredictor":
        """Train on corpus rows (each with 'features', 'options', 'metrics').

        Accepts the full corpus (routed and failed rows). The Fmax and LUT
        regressors train on the successful, routed subset; the routability
        classifier trains on all rows so it can learn what fails to route.
        """
        Xr: list[list[float]] = []
        yf: list[float] = []
        yl: list[float] = []
        Xall: list[list[float]] = []
        yroute: list[int] = []

        for rec in rows:
            metrics = rec.get("metrics", {})
            vec = _row_vector(rec)
            routed = bool(rec.get("success") and metrics.get("routed_ok"))
            Xall.append(vec)
            yroute.append(1 if routed else 0)

            fmax = metrics.get("fmax_mhz")
            if routed and fmax:
                Xr.append(vec)
                yf.append(float(fmax))
                yl.append(float(metrics.get("luts", 0.0)))

        self.n_samples = len(Xr)
        if self.n_samples < MIN_SAMPLES:
            self.model = None
            self.lut_model = None
            self.route_model = None
            self.trained = False
            return self

        self.model = self._new_regressor().fit(Xr, yf)
        if any(v > 0 for v in yl):
            self.lut_model = self._new_regressor().fit(Xr, yl)
        else:
            self.lut_model = None

        # Routability needs both classes to be learnable.
        if len(set(yroute)) == 2 and len(Xall) >= MIN_SAMPLES:
            self.route_model = self._new_classifier().fit(Xall, yroute)
        else:
            self.route_model = None

        self.trained = True
        return self

    @staticmethod
    def _new_regressor():
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
        )

    @staticmethod
    def _new_classifier():
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0
        )

    # ------------------------------------------------------------------ #
    def predict(self, features: dict[str, float], options: FlowOptions) -> float:
        if self.trained and self.model is not None:
            return float(self.model.predict([make_vector(features, options)])[0])
        return self._heuristic(features, options)

    def predict_luts(self, features: dict[str, float], options: FlowOptions) -> float:
        if self.trained and self.lut_model is not None:
            pred = float(self.lut_model.predict([make_vector(features, options)])[0])
            return max(0.0, pred)
        # Cold-start: trust the extracted/estimated LUT feature directly.
        return float(features.get("num_luts", 0.0))

    def routed_probability(
        self, features: dict[str, float], options: FlowOptions
    ) -> float:
        if self.trained and self.route_model is not None:
            proba = self.route_model.predict_proba([make_vector(features, options)])[0]
            classes = list(self.route_model.classes_)
            return float(proba[classes.index(1)]) if 1 in classes else 0.0
        return 1.0  # no evidence of routing failure -> optimistic prior

    @staticmethod
    def _heuristic(features: dict[str, float], options: FlowOptions) -> float:
        """Transparent cold-start estimate mirroring known knob effects."""
        carries = features.get("num_carries", 0.0)
        max_fanout = features.get("max_fanout", 4.0)
        dsp_avail = features.get("num_dsp", 0.0)

        base = 260.0 / (1.0 + carries / 60.0) / (1.0 + max_fanout / 45.0)
        mult = 1.0
        if options.abc9:
            mult *= 1.08
        if options.retime:
            mult *= 1.12
        if options.pipeline_output:
            mult *= 1.20
        if options.dsp and dsp_avail > 0:
            mult *= 1.15
        if options.placer == "heap":
            mult *= 1.03
        return base * mult

    # ------------------------------------------------------------------ #
    def feature_importances(self) -> dict[str, float]:
        if not (self.trained and self.model is not None):
            return {}
        imps = getattr(self.model, "feature_importances_", None)
        if imps is None:
            return {}
        return {name: float(v) for name, v in zip(VECTOR_NAMES, imps)}

    @classmethod
    def evaluate(cls, rows: list[dict[str, Any]], k: int = 5) -> ModelReport:
        """K-fold cross-validated accuracy of the Fmax head.

        This is the honest answer to "should I trust the model's ranking?".
        Runs on held-out folds so the reported error is out-of-sample.
        """
        X: list[list[float]] = []
        y: list[float] = []
        for rec in rows:
            metrics = rec.get("metrics", {})
            fmax = metrics.get("fmax_mhz")
            if rec.get("success") and metrics.get("routed_ok") and fmax:
                X.append(_row_vector(rec))
                y.append(float(fmax))

        n = len(X)
        if n < MIN_SAMPLES:
            return ModelReport(n=n, trained=False, note="not enough samples to evaluate")

        import numpy as np
        from sklearn.model_selection import KFold, cross_val_predict

        folds = max(2, min(k, n))
        kf = KFold(n_splits=folds, shuffle=True, random_state=0)
        preds = cross_val_predict(cls._new_regressor(), X, y, cv=kf)

        y_arr = np.asarray(y, dtype=float)
        err = np.abs(preds - y_arr)
        mae = float(err.mean())
        rmse = float(math.sqrt(float(((preds - y_arr) ** 2).mean())))
        ss_res = float(((y_arr - preds) ** 2).sum())
        ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum()) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        within = float((err <= 0.10 * np.maximum(y_arr, 1e-9)).mean())
        baseline_mae = float(np.abs(y_arr - y_arr.mean()).mean())

        # Importances from a full-data fit (the model we would actually ship).
        fitted = cls._new_regressor().fit(X, y)
        imps = getattr(fitted, "feature_importances_", None)
        importances = (
            {name: float(v) for name, v in zip(VECTOR_NAMES, imps)} if imps is not None else {}
        )

        return ModelReport(
            n=n,
            trained=True,
            mae=mae,
            rmse=rmse,
            r2=r2,
            within_10pct=within,
            baseline_mae=baseline_mae,
            importances=importances,
        )

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "lut_model": self.lut_model,
                "route_model": self.route_model,
                "trained": self.trained,
                "n_samples": self.n_samples,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "FmaxPredictor":
        path = Path(path)
        if not path.exists():
            return cls()
        blob = joblib.load(path)
        return cls(
            model=blob.get("model"),
            lut_model=blob.get("lut_model"),
            route_model=blob.get("route_model"),
            trained=bool(blob.get("trained", False)),
            n_samples=int(blob.get("n_samples", 0)),
        )
