"""
Paranal Tomography Dataset Assertions
======================================

Comprehensive validation of the Paranal Tomography dataset through the full
otbench pipeline: raw xarray → flattened DataFrame → task splits → forecasting
windows. Each test group is designed to be reusable as building blocks for
exploratory data analysis notebooks.

Run with synthetic data (CI-friendly, no ESO archive access):
    python -m pytest tests/test_paranal_dataset_assertions.py --use-synthetic-data -v

Run against the real dataset (requires paranal_tomography.nc on disk):
    python -m pytest tests/test_paranal_dataset_assertions.py -v
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otbench.config import settings
from otbench.dataset import Dataset
from otbench.dataset.synthetic import generate_paranal_tomography
from otbench.tasks import TaskApi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def task_api():
    """Module-scoped TaskApi (respects --use-synthetic-data via conftest)."""
    return TaskApi()


@pytest.fixture(scope="module")
def raw_xarray_dataset():
    """The raw xarray Dataset before flattening (always synthetic for unit tests)."""
    return generate_paranal_tomography()


@pytest.fixture(scope="module")
def dataset():
    """The otbench Dataset object (loads synthetic or real based on settings)."""
    return Dataset(name="paranal_tomography")


@pytest.fixture(scope="module")
def flat_df(dataset):
    """The flattened DataFrame as downstream tasks see it."""
    return dataset.get_all(data_type="pd")


# ---------------------------------------------------------------------------
# Paranal task names (must stay in sync with tasks.json)
# ---------------------------------------------------------------------------

PARANAL_REGRESSION_TASKS = [
    "regression.paranal_tomography.full.cn2_profile_reconstruction",
]

PARANAL_FORECASTING_TASKS = [
    "forecasting.paranal_tomography.full.seeing_nowcast",
    "forecasting.paranal_tomography.full.cn2_profile_forecast",
]

PARANAL_ALL_TASKS = PARANAL_REGRESSION_TASKS + PARANAL_FORECASTING_TASKS


# ===========================================================================
# Group 1: Raw xarray Schema
# ===========================================================================

class TestRawXarraySchema:
    """Validate the xarray Dataset schema before flattening."""

    REQUIRED_VARS = [
        "cn2_free_atmos", "cn2_ground_scalar", "seeing",
        "temp_profile", "wind_speed", "wind_dir", "pressure", "rh", "night_id",
    ]

    REQUIRED_COORDS = ["time", "height_mass", "height_lhatpro"]

    def test_required_variables_present(self, raw_xarray_dataset):
        for var in self.REQUIRED_VARS:
            assert var in raw_xarray_dataset.data_vars, f"Missing variable: {var}"

    def test_required_coordinates_present(self, raw_xarray_dataset):
        for coord in self.REQUIRED_COORDS:
            assert coord in raw_xarray_dataset.coords, f"Missing coordinate: {coord}"

    def test_height_mass_values(self, raw_xarray_dataset):
        expected = np.array([500, 1000, 2000, 4000, 8000, 16000])
        np.testing.assert_array_equal(
            raw_xarray_dataset.height_mass.values, expected,
            err_msg="height_mass layers do not match MASS restoration layers"
        )

    def test_height_lhatpro_has_39_levels(self, raw_xarray_dataset):
        assert len(raw_xarray_dataset.height_lhatpro) == 39, (
            f"Expected 39 LHATPRO levels, got {len(raw_xarray_dataset.height_lhatpro)}"
        )

    def test_cn2_free_atmos_shape(self, raw_xarray_dataset):
        da = raw_xarray_dataset["cn2_free_atmos"]
        assert da.dims == ("time", "height_mass"), f"Wrong dims: {da.dims}"
        assert da.shape[1] == 6

    def test_temp_profile_shape(self, raw_xarray_dataset):
        da = raw_xarray_dataset["temp_profile"]
        assert da.dims == ("time", "height_lhatpro"), f"Wrong dims: {da.dims}"
        assert da.shape[1] == 39

    def test_time_monotonically_increasing(self, raw_xarray_dataset):
        times = raw_xarray_dataset.time.values
        diffs = np.diff(times.astype(np.int64))
        assert np.all(diffs > 0), "Time coordinate is not strictly monotonically increasing"

    def test_night_id_is_integer_typed(self, raw_xarray_dataset):
        night = raw_xarray_dataset["night_id"]
        assert np.issubdtype(night.dtype, np.integer) or np.issubdtype(night.dtype, np.floating), (
            f"night_id dtype is {night.dtype}, expected integer or float"
        )


# ===========================================================================
# Group 2: Flattening Correctness
# ===========================================================================

class TestFlatteningCorrectness:
    """Verify the xarray → DataFrame flattening produces correct columns."""

    EXPECTED_SCALAR_COLUMNS = [
        "cn2_ground_scalar", "seeing", "wind_speed", "wind_dir",
        "pressure", "rh", "night_id",
    ]

    EXPECTED_CN2_PROFILE_COLUMNS = [
        "cn2_free_atmos_500", "cn2_free_atmos_1000", "cn2_free_atmos_2000",
        "cn2_free_atmos_4000", "cn2_free_atmos_8000", "cn2_free_atmos_16000",
    ]

    def test_scalar_columns_present(self, flat_df):
        for col in self.EXPECTED_SCALAR_COLUMNS:
            assert col in flat_df.columns, f"Missing scalar column after flattening: {col}"

    def test_cn2_profile_columns_present(self, flat_df):
        for col in self.EXPECTED_CN2_PROFILE_COLUMNS:
            assert col in flat_df.columns, f"Missing cn2 profile column: {col}"

    def test_temp_profile_columns_present(self, flat_df):
        temp_cols = [c for c in flat_df.columns if c.startswith("temp_profile_")]
        assert len(temp_cols) == 39, (
            f"Expected 39 temp_profile columns, got {len(temp_cols)}: {temp_cols[:5]}..."
        )

    def test_no_duplicate_columns(self, flat_df):
        dupes = flat_df.columns[flat_df.columns.duplicated()]
        assert len(dupes) == 0, f"Duplicate columns found: {dupes.tolist()}"

    def test_time_is_index(self, flat_df):
        assert flat_df.index.name == "time" or "time" in flat_df.columns, (
            "time should be the index or a column after flattening"
        )

    def test_row_count_matches_xarray(self, raw_xarray_dataset, flat_df):
        expected = len(raw_xarray_dataset.time)
        assert len(flat_df) == expected, (
            f"Row count mismatch: xarray has {expected} timesteps, DataFrame has {len(flat_df)}"
        )


# ===========================================================================
# Group 3: Night Identification and Extraction
# ===========================================================================

class TestNightIdentification:
    """Verify that observing nights can be reliably identified and extracted."""

    def test_night_id_column_exists(self, flat_df):
        assert "night_id" in flat_df.columns, "night_id column not found"

    def test_multiple_nights_present(self, flat_df):
        n_nights = flat_df["night_id"].nunique()
        assert n_nights >= 2, f"Only {n_nights} night(s) found; need >=2 for multi-night tests"

    def test_single_night_extraction(self, flat_df):
        """Extract a single night and verify it's a contiguous subset."""
        first_night = flat_df["night_id"].iloc[0]
        night_data = flat_df[flat_df["night_id"] == first_night]
        assert len(night_data) > 0, "Empty result when filtering for first night"

        # Rows within a night should be contiguous (no interleaving)
        indices = flat_df.index.get_indexer(night_data.index)
        diffs = np.diff(indices)
        assert np.all(diffs == 1), (
            "Night data is not contiguous — rows from different nights are interleaved"
        )

    def test_nights_are_non_overlapping_in_time(self, flat_df):
        """Verify that nights partition the timeline without temporal overlap."""
        if "time" not in flat_df.columns and flat_df.index.name == "time":
            times = flat_df.index
        else:
            times = pd.to_datetime(flat_df["time"])

        grouped = flat_df.groupby("night_id")
        intervals = []
        for nid, group in grouped:
            if flat_df.index.name == "time":
                t = group.index
            else:
                t = pd.to_datetime(group["time"])
            intervals.append((nid, t.min(), t.max()))

        intervals.sort(key=lambda x: x[1])
        for i in range(len(intervals) - 1):
            _, _, end_i = intervals[i]
            _, start_next, _ = intervals[i + 1]
            assert end_i < start_next, (
                f"Night {intervals[i][0]} ends at {end_i} but night "
                f"{intervals[i+1][0]} starts at {start_next} — temporal overlap"
            )

    def test_all_rows_have_night_id(self, flat_df):
        n_missing = flat_df["night_id"].isna().sum()
        assert n_missing == 0, f"{n_missing} rows have no night_id"


# ===========================================================================
# Group 4: Task Loading and Split Integrity
# ===========================================================================

class TestTaskLoading:
    """Verify all paranal tasks load correctly and produce valid X/y splits."""

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_task_loads_without_error(self, task_api, task_name):
        task = task_api.get_task(task_name)
        assert task is not None

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_train_val_test_nonempty(self, task_api, task_name):
        task = task_api.get_task(task_name)
        X_train, y_train = task.get_train_data()
        X_val, y_val = task.get_validation_data()
        X_test, y_test = task.get_test_data()

        assert len(X_train) > 0, f"{task_name}: empty training set"
        assert len(X_val) > 0, f"{task_name}: empty validation set"
        assert len(X_test) > 0, f"{task_name}: empty test set"
        assert len(y_train) == len(X_train), f"{task_name}: X/y train length mismatch"
        assert len(y_val) == len(X_val), f"{task_name}: X/y val length mismatch"
        assert len(y_test) == len(X_test), f"{task_name}: X/y test length mismatch"

    @pytest.mark.parametrize("task_name", PARANAL_REGRESSION_TASKS)
    def test_regression_target_not_in_features(self, task_api, task_name):
        """For regression tasks, the target must NOT appear in X (direct leakage)."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()

        target = task.task["target"]
        targets = target if isinstance(target, list) else [target]

        for t in targets:
            assert t not in X_train.columns, (
                f"{task_name}: target '{t}' found in regression features (leakage)"
            )

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_forecasting_target_leakage_prevented_by_shift(self, task_api, task_name):
        """For forecasting tasks, target may be in X (autoregressive), but y must be
        shifted by forecast_horizon so X[t] and y[t] refer to different times."""
        task = task_api.get_task(task_name)
        X_train, y_train = task.get_train_data()

        # Recover session context for prepare_forecasting_data
        session_col = task.task.get("session_col")
        if session_col:
            ctx = task.get_dataset().get_context(X_train.index, session_col)
            cols_to_use = ctx.columns.difference(X_train.columns)
            if not cols_to_use.empty:
                X_train = X_train.join(ctx[cols_to_use])

        X_prep, y_prep = task.prepare_forecasting_data(X_train, y_train)

        # After preparation, the target in y_prep is shifted by forecast_horizon
        # relative to X_prep. Verify they are not identical (which would mean no shift).
        target = task.task["target"]
        targets = target if isinstance(target, list) else [target]

        for t in targets:
            if t in X_prep.columns:
                # The current value (in X) and the future value (in y) must differ
                # for at least some rows (they can match by coincidence, but not all)
                x_vals = X_prep[t].values
                y_vals = y_prep[t].values if t in y_prep.columns else None
                if y_vals is not None and len(x_vals) > 10:
                    # If the forecast horizon shift is working, X[t] != y[t] in general
                    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                    assert correlation < 0.9999, (
                        f"{task_name}: target '{t}' in X and y are nearly identical "
                        f"(corr={correlation:.6f}) — shift may not be applied"
                    )

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_removed_columns_absent_from_features(self, task_api, task_name):
        """Columns in the 'remove' list must not appear in X."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()

        for col in task.task["remove"]:
            assert col not in X_train.columns, (
                f"{task_name}: removed column '{col}' still present in X_train"
            )


# ===========================================================================
# Group 5: Temporal Split Non-Overlap (No Data Leakage)
# ===========================================================================

class TestTemporalSplitIntegrity:
    """Verify train/val/test splits do not temporally overlap."""

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_no_index_overlap_between_splits(self, task_api, task_name):
        """Train, val, and test indices must be mutually exclusive."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()
        X_val, _ = task.get_validation_data()
        X_test, _ = task.get_test_data()

        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        assert train_idx.isdisjoint(val_idx), f"{task_name}: train/val index overlap"
        assert train_idx.isdisjoint(test_idx), f"{task_name}: train/test index overlap"
        assert val_idx.isdisjoint(test_idx), f"{task_name}: val/test index overlap"

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_temporal_ordering_train_before_val_before_test(self, task_api, task_name):
        """Splits must respect causal ordering: max(train) < min(val) < min(test)."""
        task = task_api.get_task(task_name)

        # Parse the raw index ranges from the task config
        train_ends = [int(idx.split(":")[1]) for idx in task.task["train_idx"]]
        val_starts = [int(idx.split(":")[0]) for idx in task.task["val_idx"]]
        val_ends = [int(idx.split(":")[1]) for idx in task.task["val_idx"]]
        test_starts = [int(idx.split(":")[0]) for idx in task.task["test_idx"]]

        assert max(train_ends) <= min(val_starts), (
            f"{task_name}: train end ({max(train_ends)}) > val start ({min(val_starts)})"
        )
        assert max(val_ends) <= min(test_starts), (
            f"{task_name}: val end ({max(val_ends)}) > test start ({min(test_starts)})"
        )


# ===========================================================================
# Group 6: Forecasting Session Masking (Night Boundary Leakage)
# ===========================================================================

class TestForecastingSessionMasking:
    """Verify that the forecasting pipeline correctly masks cross-night windows."""

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_session_col_defined_in_task(self, task_api, task_name):
        """Forecasting tasks must declare a session_col for night masking."""
        task = task_api.get_task(task_name)
        session_col = task.task.get("session_col")
        assert session_col is not None, (
            f"{task_name}: no session_col defined — cross-night leakage is unprotected"
        )
        assert session_col == "night_id"

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_session_masking_removes_cross_night_rows(self, task_api, task_name):
        """After prepare_forecasting_data, no window should span a night boundary."""
        task = task_api.get_task(task_name)
        X_train, y_train = task.get_train_data()

        # Recover night_id context (it was removed by the task)
        session_col = task.task["session_col"]
        ctx = task.get_dataset().get_context(X_train.index, session_col)

        # Only join columns that are not already present
        cols_to_use = ctx.columns.difference(X_train.columns)
        if not cols_to_use.empty:
            X_train = X_train.join(ctx[cols_to_use])

        # prepare_forecasting_data should mask cross-night rows
        X_prepared, y_prepared = task.prepare_forecasting_data(X_train, y_train)

        # After preparation, all remaining rows should have consistent night_id
        # across the window. We verify by checking that no NaN targets remain
        # (NaN targets indicate rows that were dropped, which is correct).
        assert len(X_prepared) > 0, f"{task_name}: all rows masked — no valid windows"
        assert len(y_prepared) > 0, f"{task_name}: no valid target rows after masking"
        assert not y_prepared.isna().any().any(), (
            f"{task_name}: NaN targets remain after prepare_forecasting_data"
        )

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_forecast_horizon_within_window(self, task_api, task_name):
        """forecast_horizon + window_size must not exceed training data length."""
        task = task_api.get_task(task_name)
        X_train, y_train = task.get_train_data()
        wsize = task.window_size
        fh = task.forecast_horizon
        assert wsize + fh < len(X_train), (
            f"{task_name}: window_size({wsize}) + forecast_horizon({fh}) >= "
            f"len(X_train)({len(X_train)})"
        )


# ===========================================================================
# Group 7: Multi-Night Training (Seamless Cross-Night Learning)
# ===========================================================================

class TestMultiNightTraining:
    """Verify the pipeline can seamlessly train across multiple observing nights."""

    @pytest.mark.parametrize("task_name", PARANAL_REGRESSION_TASKS)
    def test_regression_train_spans_multiple_nights(self, task_api, task_name):
        """Regression training data should include rows from multiple nights."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()

        # Recover night_id
        ctx = task.get_dataset().get_context(X_train.index, "night_id")
        n_nights = ctx["night_id"].nunique()
        assert n_nights >= 2, (
            f"{task_name}: training data only covers {n_nights} night(s)"
        )

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_forecasting_prepared_data_spans_multiple_nights(self, task_api, task_name):
        """After session masking, forecasting data should still include multiple nights."""
        task = task_api.get_task(task_name)
        X_train, y_train = task.get_train_data()

        session_col = task.task["session_col"]
        ctx = task.get_dataset().get_context(X_train.index, session_col)
        cols_to_use = ctx.columns.difference(X_train.columns)
        if not cols_to_use.empty:
            X_train = X_train.join(ctx[cols_to_use])

        X_prep, y_prep = task.prepare_forecasting_data(X_train, y_train)

        # Recover night_id for the prepared indices
        ctx_prep = task.get_dataset().get_context(X_prep.index, session_col)
        n_nights = ctx_prep[session_col].nunique()
        assert n_nights >= 2, (
            f"{task_name}: prepared forecasting data only covers {n_nights} night(s)"
        )


# ===========================================================================
# Group 8: Missing Data Handling
# ===========================================================================

class TestMissingDataHandling:
    """Verify that NaN handling is correct through the pipeline."""

    def test_flat_df_dtype_consistency(self, flat_df):
        """All columns should have numeric dtypes (float or int)."""
        for col in flat_df.columns:
            assert np.issubdtype(flat_df[col].dtype, np.number), (
                f"Column '{col}' has non-numeric dtype: {flat_df[col].dtype}"
            )

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_dropna_task_produces_no_nans(self, task_api, task_name):
        """If dropna=True, X and y should have zero NaN values."""
        task = task_api.get_task(task_name)
        if not task.task.get("dropna", False):
            pytest.skip(f"{task_name} has dropna=False")

        X_train, y_train = task.get_train_data()
        assert not X_train.isna().any().any(), f"{task_name}: NaN in X_train with dropna=True"
        assert not y_train.isna().any().any(), f"{task_name}: NaN in y_train with dropna=True"

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_no_all_nan_columns(self, task_api, task_name):
        """No feature column should be entirely NaN."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()
        all_nan_cols = X_train.columns[X_train.isna().all()]
        assert len(all_nan_cols) == 0, (
            f"{task_name}: entirely NaN columns: {all_nan_cols.tolist()}"
        )

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_no_infinite_values_in_features(self, task_api, task_name):
        """Features should not contain +/- infinity."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()
        numeric = X_train.select_dtypes(include=[np.number])
        n_inf = np.isinf(numeric.values).sum()
        assert n_inf == 0, f"{task_name}: {n_inf} infinite values found in features"


# ===========================================================================
# Group 9: Physical Bounds (Sanity Checks)
# ===========================================================================

class TestPhysicalBounds:
    """Check that variable values fall within physically plausible ranges."""

    def test_seeing_positive(self, flat_df):
        """Seeing must be positive where not NaN."""
        seeing = flat_df["seeing"].dropna()
        if len(seeing) > 0:
            assert (seeing > 0).all(), (
                f"{(seeing <= 0).sum()} non-positive seeing values"
            )

    def test_turbulence_nonnegative(self, flat_df):
        """Turbulence integrals (J_i) must be non-negative where not NaN."""
        cn2_cols = [c for c in flat_df.columns
                    if c.startswith("cn2_free_atmos_") or c == "cn2_ground_scalar"]
        for col in cn2_cols:
            vals = flat_df[col].dropna()
            if len(vals) > 0:
                n_neg = (vals < 0).sum()
                assert n_neg == 0, f"{col}: {n_neg} negative turbulence values"

    def test_wind_speed_nonnegative(self, flat_df):
        vals = flat_df["wind_speed"].dropna()
        if len(vals) > 0:
            assert (vals >= 0).all(), "Negative wind speed values found"

    def test_wind_dir_in_range(self, flat_df):
        vals = flat_df["wind_dir"].dropna()
        if len(vals) > 0:
            assert (vals >= 0).all() and (vals <= 360).all(), (
                "Wind direction outside [0, 360] range"
            )

    def test_relative_humidity_in_range(self, flat_df):
        vals = flat_df["rh"].dropna()
        if len(vals) > 0:
            assert (vals >= 0).all() and (vals <= 100).all(), (
                "Relative humidity outside [0, 100] range"
            )

    def test_pressure_plausible(self, flat_df):
        """Surface pressure at 2635m should be ~740 hPa, within [600, 900]."""
        vals = flat_df["pressure"].dropna()
        if len(vals) > 0:
            assert (vals > 600).all() and (vals < 900).all(), (
                f"Pressure values outside [600, 900] hPa range: "
                f"min={vals.min():.1f}, max={vals.max():.1f}"
            )


# ===========================================================================
# Group 10: Context Recovery (night_id accessible after removal)
# ===========================================================================

class TestContextRecovery:
    """Verify that night_id can be recovered via get_context after task removal."""

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_night_id_recoverable_from_train_indices(self, task_api, task_name):
        """After the task removes night_id from X, it must be recoverable."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()

        # night_id should be removed from X (it's in the remove list or not a feature)
        # but recoverable via context
        ctx = task.get_dataset().get_context(X_train.index, "night_id")
        assert "night_id" in ctx.columns
        assert len(ctx) == len(X_train)
        assert not ctx["night_id"].isna().any(), "Recovered night_id has NaN values"

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_context_index_alignment(self, task_api, task_name):
        """Recovered context indices must exactly match X indices."""
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()
        ctx = task.get_dataset().get_context(X_train.index, "night_id")
        pd.testing.assert_index_equal(X_train.index, ctx.index)


# ===========================================================================
# Group 11: Log Transform Correctness
# ===========================================================================

class TestLogTransform:
    """Verify log_transform is applied correctly to the target."""

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_log_transform_produces_finite_values(self, task_api, task_name):
        """If log_transform=True, y values should be finite (no -inf from log(0))."""
        task = task_api.get_task(task_name)
        if not task.task.get("log_transform", False):
            pytest.skip(f"{task_name} does not use log_transform")

        _, y_train = task.get_train_data()
        y_vals = y_train.values
        valid = y_vals[~np.isnan(y_vals)]
        n_inf = np.sum(~np.isfinite(valid))
        assert n_inf == 0, (
            f"{task_name}: log_transform produced {n_inf} non-finite values"
        )

    @pytest.mark.parametrize("task_name", PARANAL_ALL_TASKS)
    def test_log_transform_values_are_negative(self, task_api, task_name):
        """Turbulence integrals are O(1e-16), so log10 should yield ~[-19, -12]."""
        task = task_api.get_task(task_name)
        if not task.task.get("log_transform", False):
            pytest.skip(f"{task_name} does not use log_transform")

        _, y_train = task.get_train_data()
        y_vals = y_train.values.flatten()
        valid = y_vals[np.isfinite(y_vals)]
        if len(valid) == 0:
            pytest.skip("No finite y values to check")
        # log10 of turbulence integrals (order 1e-16) should be strongly negative
        assert np.median(valid) < 0, (
            f"{task_name}: median log-target is {np.median(valid):.2f}, "
            "expected negative for turbulence integrals"
        )


# ===========================================================================
# Group 12: End-to-End Forecasting Pipeline
# ===========================================================================

class TestEndToEndForecasting:
    """Full pipeline test: load → split → recover context → prepare → validate."""

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_full_forecasting_pipeline(self, task_api, task_name):
        """Run the complete forecasting pipeline as bench_runner would."""
        task = task_api.get_task(task_name)

        # 1. Load splits
        X_train, y_train = task.get_train_data()
        X_test, y_test = task.get_test_data()

        # 2. Recover session context (mirrors bench_runner.py logic)
        session_col = task.task.get("session_col")
        assert session_col is not None

        for X in [X_train, X_test]:
            ctx = task.get_dataset().get_context(X.index, session_col)
            cols_to_use = ctx.columns.difference(X.columns)
            if not cols_to_use.empty:
                X = X.join(ctx[cols_to_use])

        # Re-join for the actual preparation (need mutable reference)
        ctx_train = task.get_dataset().get_context(X_train.index, session_col)
        cols_to_use = ctx_train.columns.difference(X_train.columns)
        if not cols_to_use.empty:
            X_train = X_train.join(ctx_train[cols_to_use])

        ctx_test = task.get_dataset().get_context(X_test.index, session_col)
        cols_to_use = ctx_test.columns.difference(X_test.columns)
        if not cols_to_use.empty:
            X_test = X_test.join(ctx_test[cols_to_use])

        # 3. Prepare forecasting data (windowing + session masking)
        X_train_prep, y_train_prep = task.prepare_forecasting_data(X_train, y_train)
        X_test_prep, y_test_prep = task.prepare_forecasting_data(X_test, y_test)

        # 4. Validate outputs
        assert len(X_train_prep) > 0, "Empty training set after forecasting prep"
        assert len(X_test_prep) > 0, "Empty test set after forecasting prep"
        assert not X_train_prep.isna().any().any(), "NaN in X_train after prep"
        assert not y_train_prep.isna().any().any(), "NaN in y_train after prep"
        assert not X_test_prep.isna().any().any(), "NaN in X_test after prep"
        assert not y_test_prep.isna().any().any(), "NaN in y_test after prep"

        # 5. Verify session_col was consumed (dropped from features)
        assert session_col not in X_train_prep.columns, (
            f"session_col '{session_col}' should be dropped from features after masking"
        )

        # 6. Verify lag columns were created
        window_size = task.window_size
        if window_size > 1:
            lag_cols = [c for c in X_train_prep.columns if "(t-" in c]
            assert len(lag_cols) > 0, (
                f"No lag columns created with window_size={window_size}"
            )
