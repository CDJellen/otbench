import os
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
from typing import List, Dict

# --- Configuration: Physical Constants ---
# MASS: Coarse layers at 0.5, 1, 2, 4, 8, 16 km
MASS_HEIGHTS = [500, 1000, 2000, 4000, 8000, 16000]

# LHATPRO: 39 thermodynamic layers
LHATPRO_HEIGHTS = [
    0, 10, 30, 50, 75, 100, 125, 150, 200, 250, 325, 400, 475, 550, 625, 700, 
    800, 900, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2200, 2500, 2800, 
    3100, 3500, 3900, 4400, 5000, 5600, 6200, 7000, 8000, 9000, 10000
]

# SLODAR: 8 Fine layers. Heights are dynamic (dependent on "Cn2 layer thickness"), 
# but typically centered at [40, 80, 120, 160, 200, 250, 300, 400]. 
# We will index them simply as 1..8 for now.
SLODAR_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]

class ESOParanalLoader:
    def __init__(self, raw_dir: str = "raw"):
        self.raw_dir = raw_dir
        
    def load_csv_pattern(self, pattern: str, time_col: str = "Date time") -> pd.DataFrame:
        """
        Loads and stitches all files matching a pattern (e.g. 'mass_paranal_*.csv').
        """
        search_path = os.path.join(self.raw_dir, pattern)
        files = sorted(glob(search_path))
        
        if not files:
            print(f"  [WARN] No files found for pattern: {pattern}")
            return pd.DataFrame()
            
        print(f"  Found {len(files)} files for {pattern}. Stitching...")
        
        dfs = []
        for f in files:
            try:
                # 1. Load low_memory=False to prevent type guessing errors
                df = pd.read_csv(f, low_memory=False)
                
                # 2. Rename Time immediately
                # Handle ESO's variable time column names
                if time_col not in df.columns:
                    # Fallback search
                    candidates = [c for c in df.columns if "date" in c.lower() and "time" in c.lower()]
                    if candidates:
                        df = df.rename(columns={candidates[0]: "time"})
                    else:
                        continue # Skip bad file
                else:
                    df = df.rename(columns={time_col: "time"})

                # 3. Coerce Time
                df["time"] = pd.to_datetime(df["time"], errors='coerce', utc=True)
                df = df.dropna(subset=["time"])
                dfs.append(df)
            except Exception as e:
                print(f"  [ERR] Failed to load {f}: {e}")
        
        if not dfs:
            return pd.DataFrame()

        # Concatenate and Sort
        full_df = pd.concat(dfs, ignore_index=True)
        full_df = full_df.sort_values("time").drop_duplicates(subset="time", keep="first")
        return full_df

    def build_dataset(self) -> xr.Dataset:
        print("1. Loading Raw Campaigns...")
        # Note the wildcards (*) to catch the monthly chunk files
        meteo = self.load_csv_pattern("*meteo_paranal*.csv")
        mass = self.load_csv_pattern("*mass_paranal*.csv")
        lhatpro = self.load_csv_pattern("*lhatpro_paranal*.csv")
        slodar = self.load_csv_pattern("*slodar_paranal*.csv")

        # --- PRE-MERGE CLEANUP ---
        # Drop colliding metadata
        drop_cols = ["Integration time [s]", "Platform", "LHATPRO ID", "Telescope Azimuth", "Telescope Elevation"]
        for df in [meteo, lhatpro, slodar, mass]:
            if df.empty: continue
            cols = [c for c in drop_cols if c in df.columns]
            df.drop(columns=cols, inplace=True)

        print("2. Aligning Timestamps (Tolerance: 5min)...")
        if mass.empty:
            raise ValueError("MASS data missing. Cannot build benchmark without Ground Truth.")

        merged = mass.copy()
        
        # Merge Sequence
        if not meteo.empty:
            merged = pd.merge_asof(merged, meteo, on="time", tolerance=pd.Timedelta("5min"), direction="nearest", suffixes=("", "_meteo"))
        if not slodar.empty:
            merged = pd.merge_asof(merged, slodar, on="time", tolerance=pd.Timedelta("5min"), direction="nearest", suffixes=("", "_slodar"))
        if not lhatpro.empty:
            merged = pd.merge_asof(merged, lhatpro, on="time", tolerance=pd.Timedelta("10min"), direction="nearest", suffixes=("", "_lhatpro"))

        # Strict Drop for Input Viability
        wind_col = next((c for c in merged.columns if "Wind Speed at 30m" in c), None)
        if wind_col:
            merged = merged.dropna(subset=[wind_col])

        print("3. Vectorizing Data Streams...")
        
        # --- A. MASS PROFILE (High Altitude) ---
        mass_cols = [f"Layer {i} Cn2 [10**(-15)m**(1/3)]" for i in range(1, 7)]
        # Add Layer 0 (Ground) separately or prepended? Let's keep separate for clarity.
        
        # --- B. SLODAR PROFILE (Boundary Layer) ---
        # Keys match the 'tab_cnsqs' logic we fetched: "Cn2 in layer 1" ... "Cn2 in layer 8"
        slodar_cols = [f"Cn2 in layer {i} [10**(-15)m**(1/3)]" for i in SLODAR_LAYERS]
        
        # --- C. LHATPRO PROFILE (Thermodynamics) ---
        temp_cols = [f"Temperature [K] at {h}[m]" for h in LHATPRO_HEIGHTS]
        
        # Helper to extract tensor and handle missing cols
        def extract_tensor(cols):
            valid_cols = [c for c in cols if c in merged.columns]
            if not valid_cols:
                return np.full((len(merged), len(cols)), np.nan)
            # If some columns missing, fill with nan
            return merged[valid_cols].reindex(columns=cols, fill_value=np.nan).values

        cn2_mass_tensor = extract_tensor(mass_cols)
        cn2_slodar_tensor = extract_tensor(slodar_cols)
        temp_tensor = extract_tensor(temp_cols)

        # --- D. UNIT SCALING (The Physicist Check) ---
        # Convert 10^-15 units to SI (m^-2/3)
        cn2_mass_tensor *= 1e-15
        cn2_slodar_tensor *= 1e-15
        
        # Handle scalar ground layer from MASS
        mass_ground_col = "Layer 0 Cn2 [10**(-15)m**(1/3)]"
        cn2_ground_mass = merged[mass_ground_col].values * 1e-15 if mass_ground_col in merged.columns else np.full(len(merged), np.nan)

        print("4. Serializing to NetCDF...")
        
        # Helper for scalar columns
        def get_col(partial_name):
            matches = [c for c in merged.columns if partial_name in c]
            return merged[matches[0]].values if matches else np.full(len(merged), np.nan)

        ds = xr.Dataset(
            data_vars={
                # --- TARGETS ---
                "cn2_free_atmos": (("time", "height_mass"), cn2_mass_tensor),
                "cn2_boundary":   (("time", "height_slodar"), cn2_slodar_tensor),
                "cn2_ground_scalar": (("time",), cn2_ground_mass),
                "seeing":         (("time",), get_col("MASS-DIMM Seeing")),
                
                # --- INPUTS ---
                "temp_profile":   (("time", "height_lhatpro"), temp_tensor),
                "wind_speed":     (("time",), get_col("Wind Speed at 30m")),
                "wind_dir":       (("time",), get_col("Wind Direction at 30m")),
                "pressure":       (("time",), get_col("Air Pressure at ground")),
                "rh":             (("time",), get_col("Relative Humidity at 2m")),
            },
            coords={
                "time": merged["time"].values,
                "height_mass": MASS_HEIGHTS,
                "height_lhatpro": LHATPRO_HEIGHTS,
                "height_slodar": SLODAR_LAYERS # Logical indices 1..8
            },
            attrs={
                "project": "otbench v2",
                "site": "ESO Paranal",
                "description": "Tomographic Benchmark (MASS+SLODAR+LHATPRO)"
            }
        )
        
        return ds

if __name__ == "__main__":
    loader = ESOParanalLoader(raw_dir="raw")
    ds = loader.build_dataset()
    print(ds)
    ds.to_netcdf("paranal_v2_tomography.nc")
    print("Success.")
