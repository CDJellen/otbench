import sys

import numpy as np
import pandas as pd

# Support both standalone execution (from this directory) and package import.
try:
    from otbench.data.paranal_tomography.fetch_eso import ESOFetcher
    from otbench.data.paranal_tomography.eso import ESOParanalLoader
except ImportError:
    from fetch_eso import ESOFetcher
    from eso import ESOParanalLoader

START = "2017-01-01"
END = "2022-01-01"
OUTPUT_DIR = "raw"
NC_FILENAME = "paranal_tomography.nc"


def validate_dataset(ds) -> bool:
    """Run basic sanity checks on the constructed dataset."""
    ok = True

    # Check required variables exist
    required_vars = [
        "cn2_free_atmos", "cn2_ground_scalar", "seeing",
        "temp_profile", "wind_speed", "wind_dir", "pressure", "rh", "night_id",
    ]
    for v in required_vars:
        if v not in ds.data_vars:
            print(f"  [FAIL] Missing variable: {v}")
            ok = False

    # Check required coordinates
    for c in ["time", "height_mass", "height_lhatpro"]:
        if c not in ds.coords:
            print(f"  [FAIL] Missing coordinate: {c}")
            ok = False

    # Check dimensions
    if "height_mass" in ds.coords and len(ds.height_mass) != 6:
        print(f"  [FAIL] height_mass should have 6 layers, got {len(ds.height_mass)}")
        ok = False
    if "height_lhatpro" in ds.coords and len(ds.height_lhatpro) != 39:
        print(f"  [FAIL] height_lhatpro should have 39 layers, got {len(ds.height_lhatpro)}")
        ok = False

    # Check time is monotonically increasing
    if "time" in ds.coords:
        times = ds.time.values
        if len(times) > 1 and not np.all(np.diff(times.astype(np.int64)) >= 0):
            print("  [FAIL] Time coordinate is not monotonically non-decreasing")
            ok = False

    # Check that turbulence values are non-negative where not NaN
    for turb_var in ["cn2_free_atmos", "cn2_ground_scalar"]:
        if turb_var in ds.data_vars:
            vals = ds[turb_var].values
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0 and np.any(valid < 0):
                n_neg = np.sum(valid < 0)
                print(f"  [WARN] {turb_var} has {n_neg} negative values (physically impossible)")

    # Check seeing is positive where not NaN
    if "seeing" in ds.data_vars:
        seeing = ds["seeing"].values
        valid = seeing[~np.isnan(seeing)]
        if len(valid) > 0 and np.any(valid <= 0):
            n_neg = np.sum(valid <= 0)
            print(f"  [WARN] seeing has {n_neg} non-positive values")

    if ok:
        print("  [PASS] All validation checks passed")
    return ok


def run_pipeline(output_dir: str = OUTPUT_DIR, dataset_filename: str = NC_FILENAME):
    print("=" * 60)
    print("Paranal Tomography Dataset Pipeline")
    print("=" * 60)

    print("\n-- Phase 1: Data Acquisition")
    fetcher = ESOFetcher(output_dir=output_dir)

    instruments = ["mass_paranal", "meteo_paranal", "lhatpro_paranal"]

    for instrument in instruments:
        try:
            fetcher.fetch_campaign(instrument, start_date=START, end_date=END, freq="MS")
        except Exception as e:
            print(f"Critical Download Failure for {instrument}: {e}")

    print("\n-- Phase 2: Transformation")
    try:
        loader = ESOParanalLoader(raw_dir=output_dir)
        ds = loader.build_dataset()

        print("\n-- Phase 3: Validation")
        validate_dataset(ds)

        print("\n-- Phase 4: Serialization")
        ds.to_netcdf(dataset_filename)
        print(f"[SUCCESS] Artifact created: {dataset_filename}")
        print(f"Dimensions: {dict(ds.sizes)}")
        print(f"Time Range: {pd.to_datetime(ds.time.values.min())} to {pd.to_datetime(ds.time.values.max())}")

    except Exception as e:
        print(f"\n[FAIL] ETL Failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline(output_dir=OUTPUT_DIR, dataset_filename=NC_FILENAME)
