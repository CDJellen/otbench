import numpy as np
import pandas as pd

from otbench.benchmark.models.regression.hybrid_awt import HybridAWTRegressionModel


def _make_X(n: int = 30, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return pd.DataFrame({
        "T_air": rng.uniform(5, 30, n),
        "T_water": rng.uniform(5, 25, n),
        "wind": rng.uniform(0, 10, n),
    })


def test_instantiation_with_extra_kwargs():
    """HybridAWT must not crash when given framework kwargs it doesn't consume."""
    model = HybridAWTRegressionModel(
        name="hybrid",
        target_name="Cn2_3m",
        air_temperature_col_name="T_air",
        water_temperature_col_name="T_water",
        use_log10=True,
        # framework keys that used to crash via del kwargs["verbose"]:
        verbose=False,
        input_size=10,
        predict_residuals=True,
        obs_lat=38.98,
        obs_lon=-76.48,
        d_model=64,
        batch_size=32,
    )
    assert model.name == "hybrid"


def test_instantiation_without_verbose_kwarg():
    """HybridAWT must not crash when 'verbose' is absent from kwargs."""
    model = HybridAWTRegressionModel(
        name="hybrid",
        target_name="Cn2_3m",
        air_temperature_col_name="T_air",
        water_temperature_col_name="T_water",
    )
    assert model.name == "hybrid"


def test_train_predict_basic():
    """Train and predict produce a non-empty output of correct length."""
    rng = np.random.default_rng(42)
    X = _make_X(40, rng)
    y = pd.DataFrame({"Cn2_3m": rng.uniform(1e-16, 1e-14, 40)})

    model = HybridAWTRegressionModel(
        name="hybrid",
        target_name="Cn2_3m",
        air_temperature_col_name="T_air",
        water_temperature_col_name="T_water",
        use_log10=False,
    )
    model.train(X[:30], y[:30])
    preds = model.predict(X[30:])
    assert len(preds) == 10


def test_train_with_nan_awt_rows():
    """Rows where AWT cannot produce a prediction (NaN temps) must be dropped silently."""
    rng = np.random.default_rng(7)
    X = _make_X(40, rng)
    y = pd.DataFrame({"Cn2_3m": rng.uniform(1e-16, 1e-14, 40)})
    # Inject NaN into air temperature for some rows
    X.loc[5:10, "T_air"] = float("nan")

    model = HybridAWTRegressionModel(
        name="hybrid",
        target_name="Cn2_3m",
        air_temperature_col_name="T_air",
        water_temperature_col_name="T_water",
        use_log10=False,
    )
    model.train(X, y)  # must not raise
