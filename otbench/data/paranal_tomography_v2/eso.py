import os
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path
from typing import List

# --- Configuration ---
MASS_HEIGHTS = [500, 1000, 2000, 4000, 8000, 16000]
LHATPRO_HEIGHTS = [
    0, 10, 30, 50, 75, 100, 125, 150, 200, 250, 325, 400, 475, 550, 625, 700, 
    800, 900, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2200, 2500, 2800, 
    3100, 3500, 3900, 4400, 5000, 5600, 6200, 7000, 8000, 9000, 10000
]

class ESOParanalLoader:
    def __init__(self, raw_dir: str = "raw"):
        self.raw_dir = Path(raw_dir)
        
    def load_csv_pattern(self, pattern: str, time_col: str = "Date time") -> pd.DataFrame:
        search_path = self.raw_dir / pattern
        files = sorted(glob(str(search_path)))
        if not files:
            return pd.DataFrame()
            
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                # Normalize Time
                if time_col not in df.columns:
                    candidates = [c for c in df.columns if "date" in c.lower() and "time" in c.lower()]
                    if candidates: df = df.rename(columns={candidates[0]: "time"})
                    else: continue 
                else:
                    df = df.rename(columns={time_col: "time"})

                df["time"] = pd.to_datetime(df["time"], errors='coerce', utc=True)
                df = df.dropna(subset=["time"])
                dfs.append(df)
            except Exception:
                continue
        
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True).sort_values("time")

    def _regularize_time(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        if df.empty: return df
        df["time"] = df["time"].dt.round(freq)
        # Deduplicate
        return df.groupby("time").first().reset_index()

    def build_dataset(self) -> xr.Dataset:
        print("1. Loading Raw Campaigns (MASS + LHATPRO + Meteo)...")
        meteo = self.load_csv_pattern("*meteo_paranal*.csv")
        mass = self.load_csv_pattern("*mass_paranal*.csv")
        lhatpro = self.load_csv_pattern("*lhatpro_paranal*.csv")

        # --- PRE-MERGE CLEANUP ---
        drop_cols = ["Integration time [s]", "Platform", "LHATPRO ID", "Telescope Azimuth", "Telescope Elevation"]
        for df in [meteo, lhatpro, mass]:
            if df.empty: continue
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        print("2. Regularizing Time Grid (1-min)...")
        if mass.empty: raise ValueError("MASS data missing.")
        mass = self._regularize_time(mass)
        meteo = self._regularize_time(meteo)
        lhatpro = self._regularize_time(lhatpro)

        print("3. Aligning Streams (Causal Backward Merge)...")
        merged = mass.copy()
        tolerance = pd.Timedelta("2min")
        
        if not meteo.empty:
            merged = pd.merge_asof(merged, meteo, on="time", tolerance=tolerance, direction="backward", suffixes=("", "_meteo"))
        if not lhatpro.empty:
            merged = pd.merge_asof(merged, lhatpro, on="time", tolerance=pd.Timedelta("5min"), direction="backward", suffixes=("", "_lhatpro"))

        # Input Viability Check
        wind_col = next((c for c in merged.columns if "Wind Speed" in c), None)
        if wind_col: merged = merged.dropna(subset=[wind_col])

        print("4. Identifying Observing Sessions...")
        merged["dt_sec"] = merged["time"].diff().dt.total_seconds()
        merged["new_session"] = (merged["dt_sec"] > 3600).fillna(True).astype(int)
        merged["night_id"] = merged["new_session"].cumsum()

        print("5. Vectorizing Data Streams...")
        def extract_tensor(cols):
            valid_cols = [c for c in cols if c in merged.columns]
            if not valid_cols: return np.full((len(merged), len(cols)), np.nan)
            return merged[valid_cols].reindex(columns=cols, fill_value=np.nan).values

        def get_col(candidates, default=np.nan):
            if isinstance(candidates, str): candidates = [candidates]
            for c in candidates:
                if c in merged.columns: return merged[c].values
                matches = [col for col in merged.columns if c in col]
                if matches: 
                    matches.sort(key=len)
                    return merged[matches[0]].values
            return np.full(len(merged), default)

        # --- A. MASS PROFILE ---
        mass_cols = [f"Layer {i} Cn2 [10**(-15)m**(1/3)]" for i in range(1, 7)]
        cn2_mass_tensor = extract_tensor(mass_cols) * 1e-15 

        # --- B. THERMODYNAMICS ---
        temp_cols = [f"Temperature [K] at {h}[m]" for h in LHATPRO_HEIGHTS]
        temp_tensor = extract_tensor(temp_cols)

        # --- C. SCALARS ---
        # Note: MASS "Layer 0" is the ground integral.
        cn2_ground_scalar = get_col(["Layer 0 Cn2", "Layer 0 Cn2 [10**(-15)m**(1/3)]"]) * 1e-15

        print("6. Serializing to NetCDF...")
        ds = xr.Dataset(
            data_vars={
                "cn2_free_atmos":    (("time", "height_mass"), cn2_mass_tensor),
                "cn2_ground_scalar": (("time",), cn2_ground_scalar),
                "seeing":            (("time",), get_col(["MASS-DIMM Seeing", "Seeing"])),
                "temp_profile":      (("time", "height_lhatpro"), temp_tensor),
                "wind_speed":        (("time",), get_col(["Wind Speed at 30m", "Wind Speed"])),
                "wind_dir":          (("time",), get_col(["Wind Direction at 30m", "Wind Direction"])),
                "pressure":          (("time",), get_col(["Air Pressure at ground", "Air Pressure"])),
                "rh":                (("time",), get_col(["Relative Humidity at 2m", "Relative Humidity"])),
                "night_id":          (("time",), merged["night_id"].values),
            },
            coords={
                "time": merged["time"].values,
                "height_mass": MASS_HEIGHTS,
                "height_lhatpro": LHATPRO_HEIGHTS,
            },
            attrs={
                "project": "otbench v2",
                "site": "ESO Paranal",
                "description": "5-Year Tomography Benchmark (MASS+LHATPRO)",
                "processing": "Causal Backward Merge, 1-min Regularization"
            }
        )
        return ds

if __name__ == "__main__":
    loader = ESOParanalLoader(raw_dir="raw")
    ds = loader.build_dataset()
    ds.to_netcdf("paranal_v2_tomography.nc")
    print("Success.")
