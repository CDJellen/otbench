import pytest
import pandas as pd
import numpy as np

from otbench.benchmark.models.forecasting.linear import LinearForecastingModel


@pytest.mark.slow
def test_linear_forecasting_model(task_api):
    """Test the LinearForecastingModel."""
    # create a DataFrame with the required columns
    dates = pd.date_range(start="2024-01-01", periods=50, freq="min")
    X = pd.DataFrame({
        "T_10m": [10 for _ in range(50)],
        "T_0m": [10 for _ in range(50)],
    }, index=dates)
    y = pd.DataFrame({"Cn2_15m": [1.58e-16 for _ in range(50)]}, index=dates)

    # we need a task to prepare forecasting data
    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m")
    X, y = task.prepare_forecasting_data(X, y)

    # create the model
    model = LinearForecastingModel(
        name="linear_forecasting",
        target_name="Cn2_15m",
        window_size=task.window_size,
        forecast_horizon=task.forecast_horizon,
    )
    # check the model name
    assert model.name == "linear_forecasting"

    # train the model (this model doesn't actually train)
    model.train(X, y)
    # predict
    predictions = model.predict(X)
    # check the predictions
    assert len(predictions) == len(y)
    assert np.allclose(predictions, y.values.ravel())


def test_linear_and_mean_window_nan_and_residual_modes():
    """Verify LinearForecastingModel, MeanWindowForecastingModel, and MinuteClimatologyForecastingModel
    handle partial NaNs, residual mode (predict_residuals=True), prefix disambiguation, and timestamp gaps."""
    from otbench.benchmark.models.forecasting.mean_window import MeanWindowForecastingModel
    from otbench.benchmark.models.forecasting.minute_climatology import MinuteClimatologyForecastingModel

    # 1. Partial-NaN window handling + prefix disambiguation (temp_profile_1 vs temp_profile_10)
    X_nan = pd.DataFrame({
        "temp_profile_1 (t-2)": [-15.0, np.nan, np.nan, -15.2],
        "temp_profile_1 (t-1)": [-14.8, -14.7, np.nan, np.nan],
        "temp_profile_1 (t-0)": [-14.6, np.nan, np.nan, -14.8],
        "temp_profile_10 (t-2)": [280.0, 281.0, 282.0, 283.0],
        "temp_profile_10 (t-1)": [280.5, 281.5, 282.5, 283.5],
        "temp_profile_10 (t-0)": [281.0, 282.0, 283.0, 284.0],
    })
    y_nan = pd.DataFrame({"temp_profile_1": [-14.4, -14.6, -14.9, -14.6]})

    lin_mdl = LinearForecastingModel(
        name="lin_nan",
        target_name=["temp_profile_1"],
        window_size=3,
        forecast_horizon=1,
    )
    lin_mdl.train(X_nan, y_nan)
    preds_nan = lin_mdl.predict(X_nan).ravel()
    assert preds_nan.shape == (4,)
    assert np.all(np.isfinite(preds_nan)), "All rows (including 1-valid-lag and 0-valid-lag) must be finite"
    # Row 0 has linear ramp [-15.0, -14.8, -14.6] -> h=1 extrapolation = -14.4
    assert np.isclose(preds_nan[0], -14.4)
    # Row 1 has 1 valid lag (-14.7) -> constant fallback = -14.7
    assert np.isclose(preds_nan[1], -14.7)
    # Row 2 has 0 valid lags -> training mean fallback
    assert np.isclose(preds_nan[2], y_nan["temp_profile_1"].mean())

    # 2. Residual mode parity for LinearForecastingModel and MeanWindowForecastingModel
    X_clean = X_nan.iloc[[0, 3]].copy()
    X_clean["temp_profile_1 (t-1)"] = [-14.8, -15.0]
    y_clean = y_nan.iloc[[0, 3]].copy()

    for ModelCls in (LinearForecastingModel, MeanWindowForecastingModel):
        mdl_dir = ModelCls(
            name=f"{ModelCls.__name__}_dir",
            target_name=["temp_profile_1"],
            window_size=3,
            forecast_horizon=1,
            predict_residuals=False,
        )
        mdl_res = ModelCls(
            name=f"{ModelCls.__name__}_res",
            target_name=["temp_profile_1"],
            window_size=3,
            forecast_horizon=1,
            predict_residuals=True,
        )
        mdl_dir.train(X_clean, y_clean)
        mdl_res.train(X_clean, y_clean)
        np.testing.assert_allclose(mdl_dir.predict(X_clean), mdl_res.predict(X_clean), atol=1e-12)

    # 3. MinuteClimatologyForecastingModel median cadence resilience across initial timestamp gap
    idx = pd.DatetimeIndex([
        "2024-01-01 00:00:00",
        "2024-01-01 06:00:00",  # 6-hour gap between row 0 and row 1
        "2024-01-01 06:01:00",
        "2024-01-01 06:02:00",
        "2024-01-01 06:03:00",
        "2024-01-01 06:04:00",
    ])
    X_clim = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    y_clim = pd.DataFrame({"target": [10.0, 20.0, 21.0, 22.0, 23.0, 24.0]}, index=idx)
    clim_mdl = MinuteClimatologyForecastingModel(
        name="min_clim",
        target_name="target",
        window_size=2,
        forecast_horizon=1,
    )
    clim_mdl.train(X_clim, y_clim)
    preds_clim = clim_mdl.predict(X_clim)
    # Row 1 is at 06:00:00; with median dt = 1 min and horizon = 1, target time is 06:01:00 -> value 21.0
    assert np.isclose(preds_clim[1], 21.0)

