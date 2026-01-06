import os
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path
from typing import List, Optional

# --- Configuration: Physical Constants ---
# MASS: Coarse layers at 0.5, 1, 2, 4, 8, 16 km
MASS_HEIGHTS = [500, 1000, 2000, 4000, 8000, 16000]

# LHATPRO: 39 thermodynamic layers (Standard radiometer channels)
LHATPRO_HEIGHTS = [
    0, 10, 30, 50, 75, 100, 125, 150, 200, 250, 325, 400, 475, 550, 625, 700, 
    800, 900, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2200, 2500, 2800, 
    3100, 3500, 3900, 4400, 5000, 5600, 6200, 7000, 8000, 9000, 10000
]

# SLODAR: 8 Fine layers. 
# Note: Physical heights vary by star separation. We use logical indices 1-8 
# representing the surface layer shear profile (approx 40m - 500m).
SLODAR_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]

class ESOParanalLoader:
    def __init__(self, raw_dir: str = "raw"):
        self.raw_dir = Path(raw_dir)
        
    def load_csv_pattern(self, pattern: str, time_col: str = "Date time") -> pd.DataFrame:
        """
        Loads, stitches, and sanitizes files matching a pattern.
        """
        search_path = self.raw_dir / pattern
        files = sorted(glob(str(search_path)))
        
        if not files:
            print(f"  [WARN] No files found for pattern: {pattern}")
            return pd.DataFrame()
            
        print(f"  Found {len(files)} files for {pattern}. Stitching...")
        
        dfs = []
        for f in files:
            try:
                # 1. Load with low_memory=False to prevent mixed-type inference issues
                df = pd.read_csv(f, low_memory=False)
                
                # 2. Normalize Time Column Name
                if time_col not in df.columns:
                    # Fallback heuristic for variable ESO headers
                    candidates = [c for c in df.columns if "date" in c.lower() and "time" in c.lower()]
                    if candidates:
                        df = df.rename(columns={candidates[0]: "time"})
                    else:
                        continue 
                else:
                    df = df.rename(columns={time_col: "time"})

                # 3. Coerce Time & Drop Invalid
                df["time"] = pd.to_datetime(df["time"], errors='coerce', utc=True)
                df = df.dropna(subset=["time"])
                
                dfs.append(df)
            except Exception as e:
                print(f"  [ERR] Failed to load {f}: {e}")
        
        if not dfs:
            return pd.DataFrame()

        # 4. Global Concatenation & Sort
        full_df = pd.concat(dfs, ignore_index=True)
        full_df = full_df.sort_values("time")
        
        return full_df

    def _regularize_time(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """
        Snaps timestamps to a fixed grid to ensure regular steps for RNNs/ODEs.
        Handles collisions by taking the first observation in the bin.
        """
        if df.empty:
            return df
            
        # Round to nearest grid point
        df["time"] = df["time"].dt.round(freq)
        
        # Deduplicate: If multiple readings fall in the same minute, take the first
        # (This is safer than mean() for categorical/flag columns)
        df = df.groupby("time").first().reset_index()
        return df

    def build_dataset(self) -> xr.Dataset:
        print("1. Loading Raw Campaigns...")
        meteo = self.load_csv_pattern("*meteo_paranal*.csv")
        mass = self.load_csv_pattern("*mass_paranal*.csv")
        lhatpro = self.load_csv_pattern("*lhatpro_paranal*.csv")
        slodar = self.load_csv_pattern("*slodar_paranal*.csv")

        # --- PRE-MERGE CLEANUP ---
        # Drop colliding metadata columns that interfere with merging
        drop_cols = ["Integration time [s]", "Platform", "LHATPRO ID", 
                     "Telescope Azimuth", "Telescope Elevation"]
        
        for df in [meteo, lhatpro, slodar, mass]:
            if df.empty: continue
            cols_to_drop = [c for c in drop_cols if c in df.columns]
            df.drop(columns=cols_to_drop, inplace=True)

        print("2. Regularizing Time Grid (1-min resolution)...")
        # ML Requirement: Regular sequence steps for Backprop-through-time
        if mass.empty:
            raise ValueError("MASS data missing. Cannot build benchmark without Ground Truth.")

        mass = self._regularize_time(mass)
        meteo = self._regularize_time(meteo)
        lhatpro = self._regularize_time(lhatpro)
        slodar = self._regularize_time(slodar)

        print("3. Aligning Streams (Causal Backward Merge)...")
        # Physics Requirement: Turbulence at time t depends on Atmosphere at t or t-delta.
        # We strictly define MASS as the "Clock".
        merged = mass.copy()
        
        # Merge Strategy:
        # direction='backward': Finds the closest reading in the PAST.
        # tolerance='2min': If data is older than 2 mins, it's stale. Drop it (NaN).
        tolerance = pd.Timedelta("2min")
        
        if not meteo.empty:
            merged = pd.merge_asof(merged, meteo, on="time", tolerance=tolerance, 
                                   direction="backward", suffixes=("", "_meteo"))
        if not slodar.empty:
            merged = pd.merge_asof(merged, slodar, on="time", tolerance=tolerance, 
                                   direction="backward", suffixes=("", "_slodar"))
        # LHATPRO is slower, allow 5 min tolerance for thermodynamics
        if not lhatpro.empty:
            merged = pd.merge_asof(merged, lhatpro, on="time", tolerance=pd.Timedelta("5min"), 
                                   direction="backward", suffixes=("", "_lhatpro"))

        # Input Viability Check: We need wind speed to model anything.
        # We look for standard column names
        wind_candidates = [c for c in merged.columns if "Wind Speed" in c]
        if wind_candidates:
            merged = merged.dropna(subset=[wind_candidates[0]])
        else:
            print("  [WARN] No Wind Speed column found. Dataset may be defective.")

        print("4. Identifying Observing Sessions (Night ID)...")
        # ML Requirement: Reset RNN state on large gaps to prevent "Daylight Poisoning"
        # Calculate time delta between rows
        merged["dt_sec"] = merged["time"].diff().dt.total_seconds()
        
        # Define a "New Session" if gap > 1 hour (3600s)
        # First row is always a new session
        merged["new_session"] = (merged["dt_sec"] > 3600).fillna(True).astype(int)
        
        # Cumulative sum creates a unique ID for each contiguous night/block
        merged["night_id"] = merged["new_session"].cumsum()
        
        print(f"  Identified {merged['night_id'].max()} distinct observing sessions.")

        print("5. Vectorizing Data Streams...")
        
        # Helper to extract tensor and handle missing cols
        def extract_tensor(cols):
            valid_cols = [c for c in cols if c in merged.columns]
            if not valid_cols:
                return np.full((len(merged), len(cols)), np.nan)
            return merged[valid_cols].reindex(columns=cols, fill_value=np.nan).values

        # Helper to safely get columns with regex-like fallback
        def get_col(candidates, default=np.nan):
            if isinstance(candidates, str): candidates = [candidates]
            for c in candidates:
                if c in merged.columns: return merged[c].values
                matches = [col for col in merged.columns if c in col]
                if matches: 
                    # specific sort to prefer shorter (exact) matches
                    matches.sort(key=len)
                    return merged[matches[0]].values
            return np.full(len(merged), default)

        # --- A. MASS PROFILE (High Altitude) ---
        mass_cols = [f"Layer {i} Cn2 [10**(-15)m**(1/3)]" for i in range(1, 7)]
        cn2_mass_tensor = extract_tensor(mass_cols) * 1e-15 # Scale to SI

        # --- B. SLODAR PROFILE (Boundary Layer) ---
        # Using the specific keys from the ESO fetcher payload
        slodar_cols = [f"Cn2 in layer {i}" for i in SLODAR_LAYERS]
        # Fallback for alternative naming
        if not any(c in merged.columns for c in slodar_cols):
             slodar_cols = [f"Cn2 in layer {i} [10**(-15)m**(1/3)]" for i in SLODAR_LAYERS]
        
        cn2_slodar_tensor = extract_tensor(slodar_cols) * 1e-15 # Scale to SI
        
        # --- C. LHATPRO PROFILE (Thermodynamics) ---
        temp_cols = [f"Temperature [K] at {h}[m]" for h in LHATPRO_HEIGHTS]
        temp_tensor = extract_tensor(temp_cols)

        # --- D. SCALAR GROUND LAYER ---
        # MASS integrated ground layer (0-500m residual)
        cn2_ground_scalar = get_col(["Layer 0 Cn2", "Layer 0 Cn2 [10**(-15)m**(1/3)]"]) * 1e-15

        print("6. Serializing to NetCDF...")
        
        ds = xr.Dataset(
            data_vars={
                # --- TARGETS (The Physics to Predict) ---
                "cn2_free_atmos":    (("time", "height_mass"), cn2_mass_tensor),
                "cn2_boundary":      (("time", "height_slodar"), cn2_slodar_tensor),
                "cn2_ground_scalar": (("time",), cn2_ground_scalar),
                "seeing":            (("time",), get_col(["MASS-DIMM Seeing", "Seeing"])),
                
                # --- INPUTS (The Drivers) ---
                "temp_profile":      (("time", "height_lhatpro"), temp_tensor),
                "wind_speed":        (("time",), get_col(["Wind Speed at 30m", "Wind Speed"])),
                "wind_dir":          (("time",), get_col(["Wind Direction at 30m", "Wind Direction"])),
                "pressure":          (("time",), get_col(["Air Pressure at ground", "Air Pressure"])),
                "rh":                (("time",), get_col(["Relative Humidity at 2m", "Relative Humidity"])),
                
                # --- METADATA (For Batching) ---
                "night_id":          (("time",), merged["night_id"].values),
            },
            coords={
                "time": merged["time"].values,
                "height_mass": MASS_HEIGHTS,
                "height_lhatpro": LHATPRO_HEIGHTS,
                "height_slodar": SLODAR_LAYERS # Logical indices
            },
            attrs={
                "project": "otbench v2",
                "site": "ESO Paranal",
                "description": "Tomographic Benchmark (MASS+SLODAR+LHATPRO)",
                "processing": "Causal Backward Merge (2min tolerance), 1-min Regularization"
            }
        )
        
        return ds

if __name__ == "__main__":
    try:
        loader = ESOParanalLoader(raw_dir="raw")
        ds = loader.build_dataset()
        print("\nDataset Summary:")
        print(ds)
        
        output_file = "paranal_v2_tomography.nc"
        ds.to_netcdf(output_file)
        print(f"\n[SUCCESS] Written to {output_file}")
        
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Pipeline aborted: {e}")
        import traceback
        traceback.print_exc()
