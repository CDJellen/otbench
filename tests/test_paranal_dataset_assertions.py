"""
Paranal Tomography Dataset Assertions
======================================

Comprehensive validation of the Paranal Tomography dataset through the full
otbench pipeline: raw xarray → flattened DataFrame → task splits → forecasting
windows.  Each test group is designed to be reusable as building blocks for
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
from otbench.tasks import TaskApi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def task_api():
    """Module-scoped TaskApi (respects --use-synthetic-data via conftest)."""
    return TaskApi()


@pytest.fixture(scope="module")
def dataset():
    """The otbench Dataset object (loads synthetic or real based on settings)."""
    return Dataset(name="paranal_tomography")


@pytest.fixture(scope="module")
def raw_xarray_dataset(dataset):
    """The raw xarray Dataset from the SAME source as flat_df.

    Uses dataset.get_xarray() so the fixture and the flattening pipeline
    always operate on identical data.  Skips if the dataset is not
    xarray-backed (e.g. CSV-only datasets).
    """
    ds = dataset.get_xarray()
    if ds is None:
        pytest.skip("Dataset is not xarray-backed")
    return ds


@pytest.fixture(scope="module")
def flat_df(dataset):
    """A flat DataFrame slice for schema and column validation.

    Bounded to _FLAT_DF_SAMPLE rows so the fixture never OOMs on the real
    dataset (~914 MB for 2.2 M rows × 52 columns × float64).

    Synthetic run : all 2 000 rows.
    Real run      : first 5 000 rows (column / dtype checks are fully valid).
    """
    _FLAT_DF_SAMPLE = 5000
    return dataset.get_sample_df(_FLAT_DF_SAMPLE)


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
        # Load just the time coordinate (cheap even for lazy datasets)
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
    """Verify the xarray → DataFrame flattening produces correct columns and row count."""

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

    def test_row_count_matches_xarray_slice(self, dataset, raw_xarray_dataset):
        """Flattening must preserve the row count of the source slice exactly.

        We verify a bounded slice (N rows) rather than the full dataset so the
        test never OOMs and the assertion is tight: the flattened output must
        have exactly N rows, not N ± anything from an outer-join artefact.
        """
        n = min(500, len(raw_xarray_dataset.time))
        sample_df = dataset.get_sample_df(n)
        assert len(sample_df) == n, (
            f"Flattening {n}-row xarray slice produced {len(sample_df)} rows — "
            "likely an outer-join index mismatch between variables"
        )

    def test_52_total_columns(self, flat_df):
        """The canonical flat schema has 52 columns (6 MASS + 39 LHATPRO + 7 scalar)."""
        assert len(flat_df.columns) == 52, (
            f"Expected 52 columns, got {len(flat_df.columns)}: {flat_df.columns.tolist()}"
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
        if flat_df.index.name == "time":
            pass  # index already is time
        else:
            pytest.skip("flat_df does not have time as index")

        grouped = flat_df.groupby("night_id")
        intervals = []
        for nid, group in grouped:
            t = group.index
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

        target = task.task["target"]
        targets = target if isinstance(target, list) else [target]

        for t in targets:
            if t in X_prep.columns:
                x_vals = X_prep[t].values
                y_vals = y_prep[t].values if t in y_prep.columns else None
                if y_vals is not None and len(x_vals) > 10:
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

        session_col = task.task["session_col"]
        ctx = task.get_dataset().get_context(X_train.index, session_col)

        cols_to_use = ctx.columns.difference(X_train.columns)
        if not cols_to_use.empty:
            X_train = X_train.join(ctx[cols_to_use])

        X_prepared, y_prepared = task.prepare_forecasting_data(X_train, y_train)

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

        X_train, y_train = task.get_train_data()
        X_test, y_test = task.get_test_data()

        session_col = task.task.get("session_col")
        assert session_col is not None

        ctx_train = task.get_dataset().get_context(X_train.index, session_col)
        cols_to_use = ctx_train.columns.difference(X_train.columns)
        if not cols_to_use.empty:
            X_train = X_train.join(ctx_train[cols_to_use])

        ctx_test = task.get_dataset().get_context(X_test.index, session_col)
        cols_to_use = ctx_test.columns.difference(X_test.columns)
        if not cols_to_use.empty:
            X_test = X_test.join(ctx_test[cols_to_use])

        X_train_prep, y_train_prep = task.prepare_forecasting_data(X_train, y_train)
        X_test_prep, y_test_prep = task.prepare_forecasting_data(X_test, y_test)

        assert len(X_train_prep) > 0, "Empty training set after forecasting prep"
        assert len(X_test_prep) > 0, "Empty test set after forecasting prep"
        assert not X_train_prep.isna().any().any(), "NaN in X_train after prep"
        assert not y_train_prep.isna().any().any(), "NaN in y_train after prep"
        assert not X_test_prep.isna().any().any(), "NaN in X_test after prep"
        assert not y_test_prep.isna().any().any(), "NaN in y_test after prep"

        assert session_col not in X_train_prep.columns, (
            f"session_col '{session_col}' should be dropped from features after masking"
        )

        window_size = task.window_size
        if window_size > 1:
            lag_cols = [c for c in X_train_prep.columns if "(t-" in c]
            assert len(lag_cols) > 0, (
                f"No lag columns created with window_size={window_size}"
            )


# ===========================================================================
# Group 13: Night-Aware Modeling
# ===========================================================================

class TestNightAwareModeling:
    """Validate that observing-night boundaries are correctly enforced throughout
    the forecasting pipeline.

    Paranal data is collected in discrete observing nights (UTC-16h shift yields
    a local-noon epoch, stored as YYYYMMDD integer ``night_id``).  The forecasting
    tasks exploit this structure in two ways:

    1. ``night_id`` is NOT in the ``remove`` list for forecasting tasks, so it
       travels through ``get_train_data()`` into X, enabling session-aware lag
       construction.
    2. ``prepare_forecasting_data()`` groups by ``night_id``, builds lags *within*
       each night, then discards the column — preventing any atmospheric state from
       night N from contaminating night N+1's feature vectors.

    These tests codify both the invariants above and the arithmetic governing how
    many valid samples survive the window/horizon trimming within each night.
    """

    # ------------------------------------------------------------------
    # Invariant 1: night_id routing through the pipeline
    # ------------------------------------------------------------------

    def test_regression_task_excludes_night_id_from_features(self, task_api):
        """Regression task has night_id in its remove list; X must not contain it.

        For the regression task there is no temporal session structure to exploit —
        each row is an independent nowcast — so night_id is removed to avoid
        target leakage through temporal clustering.
        """
        task = task_api.get_task(
            "regression.paranal_tomography.full.cn2_profile_reconstruction"
        )
        X_train, _ = task.get_train_data()
        assert "night_id" not in X_train.columns, (
            "night_id must be absent from regression task features: it is in "
            "the task's 'remove' list and could induce target leakage through "
            "temporal clustering."
        )

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_forecasting_task_night_id_in_raw_features(self, task_api, task_name):
        """Forecasting tasks do NOT remove night_id; it must be present in raw X.

        night_id is the session_col that gates per-night lag construction.
        If it were absent, prepare_forecasting_data would fall back to global
        lagging and silently allow cross-night contamination.
        """
        task = task_api.get_task(task_name)
        X_train, _ = task.get_train_data()
        assert "night_id" in X_train.columns, (
            f"{task_name}: night_id must be present in raw X so that "
            "prepare_forecasting_data can group by session and build lags "
            "within each observing night."
        )

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_night_id_absent_from_prepared_features(self, task_api, task_name):
        """After prepare_forecasting_data, night_id must be dropped from features.

        night_id is a bookkeeping label, not a predictive signal, and must not
        reach the model's feature matrix after session-aware lag construction.
        """
        task = task_api.get_task(task_name)
        X_raw, y_raw = task.get_train_data()
        X_prep, _ = task.prepare_forecasting_data(X_raw.copy(), y_raw.copy())
        assert "night_id" not in X_prep.columns, (
            f"{task_name}: night_id must be dropped from X after "
            "prepare_forecasting_data — it is a session key, not a feature."
        )

    # ------------------------------------------------------------------
    # Invariant 2: cross-night lag contamination is absent
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_no_cross_night_lag_contamination(self, task_api, task_name):
        """At the first valid prepared row of each night, the deepest lag must
        equal that night's own opening observation — not the previous night's tail.

        Explanation
        -----------
        With window_size W, the deepest lag column is ``feature (t-(W-1))``.
        For the first valid prepared row of night N (session-local row W-1), a
        correctly session-aware pipeline produces::

            feature (t-(W-1)) == raw_feature[night_N_start]   ✓  (same night)

        Under broken global lagging the identical position would yield::

            feature (t-(W-1)) == raw_feature[night_N_start - (W-1)]  ✗  (previous night)

        The test walks prepared data night-by-night, accumulating row offsets
        from the exact per-session count formula max(0, N_i - (W-1) - H).
        """
        task = task_api.get_task(task_name)
        W = task.window_size
        H = task.forecast_horizon

        if W < 2:
            pytest.skip(f"{task_name}: window_size={W} < 2, no lag columns present")

        X_raw, y_raw = task.get_train_data()
        session_col = task.task.get("session_col")

        assert session_col in X_raw.columns, (
            f"'{session_col}' not in X_raw — verify that '{task_name}' does not "
            "include night_id in its remove list."
        )

        # Choose the first non-session feature as sentinel
        feature_candidates = [c for c in X_raw.columns if c != session_col]
        assert feature_candidates, f"{task_name}: no usable feature columns"
        sentinel_feature = feature_candidates[0]
        deepest_lag_col = f"{sentinel_feature} (t-{W - 1})"

        night_ids = X_raw[session_col]
        unique_nights = night_ids.unique()  # insertion-order (night 1, 2, …)

        # Record each night's opening value for the sentinel feature
        night_opening_val = {
            nid: X_raw.loc[night_ids == nid, sentinel_feature].iloc[0]
            for nid in unique_nights
        }

        # Per-session valid row counts
        rows_per_night = {
            nid: max(0, int((night_ids == nid).sum()) - (W - 1) - H)
            for nid in unique_nights
        }

        X_prep, _ = task.prepare_forecasting_data(X_raw.copy(), y_raw.copy())

        assert deepest_lag_col in X_prep.columns, (
            f"{task_name}: expected lag column '{deepest_lag_col}' not found. "
            f"Available columns: {list(X_prep.columns[:10])}…"
        )

        cursor = 0
        for nid in unique_nights:
            n_valid = rows_per_night[nid]
            if n_valid == 0:
                continue

            actual_lag_val = X_prep.iloc[cursor][deepest_lag_col]
            expected_lag_val = night_opening_val[nid]

            assert np.isclose(actual_lag_val, expected_lag_val, rtol=1e-6), (
                f"{task_name} — Night {nid}: deepest lag '{deepest_lag_col}' "
                f"at first prepared row = {actual_lag_val:.6g}, "
                f"expected {expected_lag_val:.6g} (night's own opening value). "
                "Cross-night lag contamination detected: lags must be built "
                "per-session, not globally across the full split."
            )

            cursor += n_valid

    # ------------------------------------------------------------------
    # Invariant 3: per-session sample count arithmetic
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_per_session_sample_count_exact(self, task_api, task_name):
        """Prepared sample count equals Σ max(0, N_i − (W−1) − H) over nights.

        This formula counts the rows that survive within each session after
        discarding the initial W-1 lag-NaN rows and the final H shift-NaN rows.
        Equality with the global formula N_total − (W−1) − H would indicate that
        cross-night boundary rows are *not* being dropped, i.e. per-session
        logic is silently inactive.
        """
        task = task_api.get_task(task_name)
        W = task.window_size
        H = task.forecast_horizon

        X_raw, y_raw = task.get_train_data()
        session_col = task.task.get("session_col")
        assert session_col in X_raw.columns

        night_ids = X_raw[session_col]
        expected = sum(
            max(0, count - (W - 1) - H)
            for count in night_ids.value_counts(sort=False).values
        )

        X_prep, _ = task.prepare_forecasting_data(X_raw.copy(), y_raw.copy())
        actual = len(X_prep)

        assert actual == expected, (
            f"{task_name}: expected {expected} prepared rows "
            f"(Σ max(0, N_i − {W - 1} − {H})), got {actual}. "
            "If actual > expected, cross-night boundary rows are not being "
            "excluded — check that per-session lag construction is active."
        )

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_every_training_night_yields_at_least_one_sample(self, task_api, task_name):
        """Each observing night in the training split must be long enough to
        contribute at least one valid prepared sample.

        The minimum viable night length is (W−1) + H + 1 minutes.  A night
        shorter than this produces zero samples and is silently skipped by
        prepare_forecasting_data, which can mask data coverage problems.
        """
        task = task_api.get_task(task_name)
        W = task.window_size
        H = task.forecast_horizon
        min_length = (W - 1) + H + 1

        X_raw, _ = task.get_train_data()
        session_col = task.task.get("session_col")
        assert session_col in X_raw.columns

        night_ids = X_raw[session_col]
        short_nights = {
            nid: count
            for nid, count in night_ids.value_counts(sort=False).items()
            if count < min_length
        }

        assert not short_nights, (
            f"{task_name}: the following training nights are too short to yield "
            f"any prepared samples (need ≥ {min_length} rows, W={W}, H={H}): "
            f"{short_nights}. Consider widening the training window or flagging "
            "these nights as incomplete in the dataset."
        )

    # ------------------------------------------------------------------
    # Invariant 4: intra-night temporal cadence
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("task_name", PARANAL_FORECASTING_TASKS)
    def test_intra_night_temporal_cadence_is_1min(self, task_api, task_name):
        """Within every observing night the median inter-sample interval must be
        exactly 1 minute.

        The forecast_horizon is expressed in number of steps (minutes), so any
        deviation from 1-minute cadence would silently mis-scale all look-ahead
        distances.  For example, a task with forecast_horizon=5 nominally predicts
        "5 minutes ahead"; a 2-minute cadence would make it 10 minutes ahead.
        """
        task = task_api.get_task(task_name)
        X_raw, _ = task.get_train_data()

        if not isinstance(X_raw.index, pd.DatetimeIndex):
            pytest.skip(f"{task_name}: index is not DatetimeIndex, cannot check cadence")

        session_col = task.task.get("session_col")
        assert session_col in X_raw.columns

        night_ids = X_raw[session_col]
        bad_nights = {}

        for nid, idx in night_ids.groupby(night_ids, sort=False).groups.items():
            times = X_raw.index[X_raw.index.isin(idx)]
            if len(times) < 2:
                continue
            diffs = pd.Series(times).diff().dropna()
            median_gap = diffs.median()
            if median_gap != pd.Timedelta("1min"):
                bad_nights[nid] = str(median_gap)

        assert not bad_nights, (
            f"{task_name}: the following nights have intra-night cadence ≠ 1 min "
            f"(forecast_horizon steps would be mis-scaled): {bad_nights}"
        )


# ===========================================================================
# Group 14: Lazy Loading and Dataset API
# ===========================================================================

class TestLazyLoadingAPI:
    """Verify the lazy xarray path and new Dataset convenience methods."""

    def test_get_xarray_returns_dataset(self, dataset):
        """paranal_tomography is xarray-backed with the real dataset (lazy=true).

        Skipped for synthetic data because the synthetic path eagerly flattens
        the in-memory xr.Dataset to a pd.DataFrame before storing it in
        Dataset._data, so get_xarray() correctly returns None there.
        """
        if settings.USE_SYNTHETIC_DATA:
            pytest.skip("Synthetic data uses eager flattening, not lazy xarray backing")
        ds = dataset.get_xarray()
        assert ds is not None, (
            "dataset.get_xarray() returned None — paranal_tomography should be "
            "xarray-backed with 'lazy': true in datasets.json"
        )
        assert isinstance(ds, xr.Dataset)

    def test_get_xarray_has_correct_variables(self, dataset):
        ds = dataset.get_xarray()
        if ds is None:
            pytest.skip("Not xarray-backed")
        expected_vars = {"cn2_free_atmos", "cn2_ground_scalar", "seeing",
                         "temp_profile", "wind_speed", "wind_dir",
                         "pressure", "rh", "night_id"}
        assert expected_vars.issubset(set(ds.data_vars))

    def test_get_sample_df_returns_bounded_frame(self, dataset):
        """get_sample_df must return exactly the requested number of rows."""
        n = 100
        sample = dataset.get_sample_df(n)
        assert isinstance(sample, pd.DataFrame)
        assert len(sample) == n, (
            f"get_sample_df({n}) returned {len(sample)} rows"
        )

    def test_get_sample_df_columns_match_full_schema(self, dataset, flat_df):
        """Sample df columns must exactly match the full flat schema."""
        sample = dataset.get_sample_df(50)
        assert set(sample.columns) == set(flat_df.columns), (
            f"Column mismatch:\n"
            f"  sample-only: {set(sample.columns) - set(flat_df.columns)}\n"
            f"  flat_df-only: {set(flat_df.columns) - set(sample.columns)}"
        )

    def test_get_all_pd_returns_dataframe(self, dataset):
        """get_all(data_type='pd') must return a flat DataFrame even for lazy datasets."""
        # We use get_sample_df to avoid OOM, but verify the API contract
        result = dataset.get_sample_df(10)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 52
