from glob import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import xarray as xr

# --- CONFIGURATION ---

# MASS Layer Heights (meters above observatory)
# These are the nominal layer centers for the 6 free-atmosphere restoration layers.
MASS_HEIGHTS = [500, 1000, 2000, 4000, 8000, 16000]

# LHATPRO Radiometric Profile Heights (meters above observatory)
# Non-uniform vertical grid with fine resolution in the boundary layer.
LHATPRO_HEIGHTS = [
    0, 10, 30, 50, 75, 100, 125, 150, 200, 250, 325, 400, 475, 550, 625, 700,
    800, 900, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2200, 2500, 2800,
    3100, 3500, 3900, 4400, 5000, 5600, 6200, 7000, 8000, 9000, 10000
]

# Paranal Observatory longitude (west), used to derive observing-night boundaries.
PARANAL_LONGITUDE_DEG_W = 70.4

# --- EXPLICIT COLUMN MAPPINGS ---
# These map the ESO Archive CSV headers to our internal variable names.
# Every column we extract must be listed here; no fuzzy/substring matching.

METEO_COLUMN_MAP = {
    "wind_speed":  "Wind Speed at 30m [m/s]",
    "wind_dir":    "Wind Direction at 30m (0/360) [deg]",
    "pressure":    "Air Pressure at ground [hPa]",
    "rh":          "Relative Humidity at 2m [%]",
}

MASS_SCALAR_MAP = {
    "cn2_ground_scalar": "Layer 0 Cn2 [10**(-15)m**(1/3)]",
    "seeing":            'MASS-DIMM Seeing ["]',
}

# Merge tolerances: the maximum temporal gap we accept when aligning streams.
# All merges use direction="backward" to enforce strict causality (no future data).
LHATPRO_MERGE_TOLERANCE = pd.Timedelta("5min")
METEO_MERGE_TOLERANCE = pd.Timedelta("2min")
MASS_MERGE_TOLERANCE = pd.Timedelta("2min")


class ESOParanalLoader:

    def __init__(self, raw_dir: str):
        self.raw_dir = Path(raw_dir)

    def load_csv_pattern(self, pattern: str, time_col: str = "Date time") -> pd.DataFrame:
        """Loads and sorts raw CSV shards into a single DataFrame."""
        search_path = self.raw_dir / pattern
        files = sorted(glob(str(search_path)))
        if not files:
            print(f"[WARN] No files found for {pattern}")
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)

                # Normalize Time Column
                target_col = None
                if time_col in df.columns:
                    target_col = time_col
                else:
                    candidates = [c for c in df.columns if "Date" in c and "time" in c]
                    if candidates:
                        target_col = candidates[0]

                if target_col:
                    df = df.rename(columns={target_col: "time"})
                    df["time"] = pd.to_datetime(df["time"], errors='coerce', utc=True)
                    df = df.dropna(subset=["time"])
                    dfs.append(df)
            except (pd.errors.ParserError, ValueError, UnicodeDecodeError) as e:
                print(f"[WARN] Failed to parse {f}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True).sort_values("time")

    def _regularize_time(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Round timestamps to the nearest minute and deduplicate.

        When multiple readings fall within the same minute, the first
        observation (in original CSV ordering) is retained. This is a
        deliberate design choice for 1-min cadence benchmarks.
        """
        if df.empty:
            return df
        df = df.copy()
        df["time"] = df["time"].dt.round(freq)
        return df.groupby("time").first().reset_index()

    def _resolve_column(self, df: pd.DataFrame, exact_name: str) -> Optional[np.ndarray]:
        """Resolve a column by exact name. Returns None if not found."""
        if exact_name in df.columns:
            return df[exact_name].values
        return None

    def build_dataset(self) -> xr.Dataset:
        print("1. Loading Raw Campaigns...")
        meteo = self.load_csv_pattern("*meteo_paranal*.csv")
        mass = self.load_csv_pattern("*mass_paranal*.csv")
        lhatpro = self.load_csv_pattern("*lhatpro_paranal*.csv")

        if mass.empty:
            raise ValueError("Critical Error: No MASS data found.")
        if lhatpro.empty:
            raise ValueError("Critical Error: No LHATPRO data found.")

        print(f"   Raw rows  — MASS: {len(mass)}, LHATPRO: {len(lhatpro)}, Meteo: {len(meteo)}")

        # Cleanup known metadata cols that cause merge conflicts
        drop_cols = ["LHATPRO ID", "Platform"]
        for df in [meteo, lhatpro, mass]:
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

        print("2. Regularizing Time Grid (1-min)...")
        mass = self._regularize_time(mass)
        meteo = self._regularize_time(meteo)
        lhatpro = self._regularize_time(lhatpro)

        print(f"   After regularization — MASS: {len(mass)}, LHATPRO: {len(lhatpro)}, Meteo: {len(meteo)}")

        print("3. Aligning Streams (Causal Backward Merge)...")
        # Build the time backbone from the union of all instrument timestamps.
        idx_mass = pd.Index(mass["time"])
        idx_lhatpro = pd.Index(lhatpro["time"])
        union_index = idx_mass.union(idx_lhatpro)
        if not meteo.empty:
            idx_meteo = pd.Index(meteo["time"])
            union_index = union_index.union(idx_meteo)

        merged = pd.DataFrame({"time": union_index}).sort_values("time").reset_index(drop=True)
        print(f"   Backbone Size: {len(merged)} epochs")

        # --- CAUSAL MERGES ---
        # All three merges use direction="backward" so that each row only
        # contains information available at or before that timestamp.

        # A. LHATPRO (thermodynamic state — persists between readings)
        merged = pd.merge_asof(
            merged, lhatpro, on="time",
            tolerance=LHATPRO_MERGE_TOLERANCE,
            direction="backward",
            suffixes=("", "_lhatpro"),
        )

        # B. Surface meteorology (also a persistent state measurement)
        if not meteo.empty:
            merged = pd.merge_asof(
                merged, meteo, on="time",
                tolerance=METEO_MERGE_TOLERANCE,
                direction="backward",
                suffixes=("", "_meteo"),
            )

        # C. MASS turbulence measurements (discrete event observations)
        # CRITICAL: direction="backward" prevents future data leakage.
        # A row at time T only sees MASS observations taken at or before T.
        merged = pd.merge_asof(
            merged, mass, on="time",
            tolerance=MASS_MERGE_TOLERANCE,
            direction="backward",
            suffixes=("", "_mass"),
        )

        # --- D. DAYLIGHT PRUNING ---
        # Drop rows where the LHATPRO radiometer has no reading (instrument off
        # or gap in the union where only sparse meteo/MASS might exist).
        temp_col_0m = "Temperature [K] at 0[m]"
        if temp_col_0m in merged.columns:
            pre_len = len(merged)
            merged = merged.dropna(subset=[temp_col_0m])
            print(f"   Pruned {pre_len - len(merged)} rows (missing LHATPRO thermodynamics)")

        print(f"   Final Aligned Size: {len(merged)} epochs")

        print("4. Identifying Observing Sessions...")
        # An observing "night" straddles midnight.  To assign a unique date to
        # each night we shift UTC timestamps so that the day boundary falls at
        # local solar noon (when the telescope is idle).
        #
        # Paranal is at 70.4 deg W.  Local solar noon ≈ UTC + 70.4/15 h ≈ UTC 16:41.
        # We round to 16 h for a clean boundary; the ~41 min error is irrelevant
        # because the observatory never observes near noon.
        local_noon_shift = pd.Timedelta(hours=16)
        shifted_time = merged["time"] - local_noon_shift
        merged["night_id"] = shifted_time.dt.strftime('%Y%m%d').astype(int)

        n_nights = merged["night_id"].nunique()
        print(f"   Identified {n_nights} unique observing sessions")

        print("5. Vectorizing Data Streams...")

        # --- Vector extraction helper ---
        def extract_numpy(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
            """Extract an ordered set of columns as a 2D numpy array."""
            present = [c for c in cols if c in df.columns]
            if not present:
                return np.full((len(df), len(cols)), np.nan)
            return df[cols].reindex(columns=cols).values

        # --- A. MASS Profile (Free Atmosphere Turbulence Integrals) ---
        # The ESO MASS archive labels these "Cn2" but the units [10^-15 m^(1/3)]
        # reveal they are layer-integrated turbulence strengths:
        #   J_i = integral_{layer_i}( C_n^2(h) dh )
        # We scale from the archive unit to SI [m^(1/3)].
        mass_cols = [f"Layer {i} Cn2 [10**(-15)m**(1/3)]" for i in range(1, 7)]
        if mass_cols[0] not in merged.columns:
            mass_cols = [f"Layer {i} Cn2" for i in range(1, 7)]
        cn2_mass_tensor = extract_numpy(merged, mass_cols) * 1e-15
        # Negative turbulence integrals are physically impossible (instrument error).
        cn2_mass_tensor[cn2_mass_tensor < 0] = np.nan

        # --- B. LHATPRO Profile (Thermodynamic Vertical Profile) ---
        temp_cols = [f"Temperature [K] at {h}[m]" for h in LHATPRO_HEIGHTS]
        temp_tensor = extract_numpy(merged, temp_cols)

        # --- C. Scalar Variables (explicit column resolution) ---
        ground_col = MASS_SCALAR_MAP["cn2_ground_scalar"]
        cn2_ground_arr = self._resolve_column(merged, ground_col)
        if cn2_ground_arr is None:
            # Fallback for older CSV format without units in header
            cn2_ground_arr = self._resolve_column(merged, "Layer 0 Cn2")
        if cn2_ground_arr is None:
            cn2_ground_arr = np.full(len(merged), np.nan)
        cn2_ground = cn2_ground_arr.astype(np.float64) * 1e-15
        cn2_ground[cn2_ground < 0] = np.nan

        seeing_col = MASS_SCALAR_MAP["seeing"]
        seeing = self._resolve_column(merged, seeing_col)
        if seeing is None:
            seeing = self._resolve_column(merged, "MASS-DIMM Seeing")
        if seeing is None:
            seeing = np.full(len(merged), np.nan)
        seeing = seeing.astype(np.float64)

        w_spd_col = METEO_COLUMN_MAP["wind_speed"]
        w_spd = self._resolve_column(merged, w_spd_col)
        if w_spd is None:
            print(f"   [WARN] Column '{w_spd_col}' not found; wind_speed will be NaN")
            w_spd = np.full(len(merged), np.nan)
        w_spd = w_spd.astype(np.float64)

        w_dir_col = METEO_COLUMN_MAP["wind_dir"]
        w_dir = self._resolve_column(merged, w_dir_col)
        if w_dir is None:
            print(f"   [WARN] Column '{w_dir_col}' not found; wind_dir will be NaN")
            w_dir = np.full(len(merged), np.nan)
        w_dir = w_dir.astype(np.float64)

        pres_col = METEO_COLUMN_MAP["pressure"]
        pres = self._resolve_column(merged, pres_col)
        if pres is None:
            print(f"   [WARN] Column '{pres_col}' not found; pressure will be NaN")
            pres = np.full(len(merged), np.nan)
        pres = pres.astype(np.float64)

        rh_col = METEO_COLUMN_MAP["rh"]
        rh = self._resolve_column(merged, rh_col)
        if rh is None:
            print(f"   [WARN] Column '{rh_col}' not found; rh will be NaN")
            rh = np.full(len(merged), np.nan)
        rh = rh.astype(np.float64)

        # --- D. Data Quality Summary ---
        n = len(merged)
        mass_valid = np.count_nonzero(~np.isnan(cn2_mass_tensor[:, 0]))
        lhatpro_valid = np.count_nonzero(~np.isnan(temp_tensor[:, 0]))
        seeing_valid = np.count_nonzero(~np.isnan(seeing))
        meteo_valid = np.count_nonzero(~np.isnan(w_spd))
        print(f"   Coverage — MASS: {mass_valid}/{n} ({100*mass_valid/n:.1f}%), "
              f"LHATPRO: {lhatpro_valid}/{n} ({100*lhatpro_valid/n:.1f}%), "
              f"Seeing: {seeing_valid}/{n} ({100*seeing_valid/n:.1f}%), "
              f"Meteo: {meteo_valid}/{n} ({100*meteo_valid/n:.1f}%)")

        print("6. Serializing to xarray Dataset...")
        ds = xr.Dataset(
            data_vars={
                "cn2_free_atmos": xr.Variable(
                    ("time", "height_mass"), cn2_mass_tensor,
                    attrs={
                        "long_name": "Free-atmosphere layer-integrated turbulence strength (MASS)",
                        "units": "m^(1/3)",
                        "note": (
                            "Despite the conventional 'cn2' name, these are "
                            "layer-integrated turbulence strengths "
                            "J_i = integral(C_n^2 dh) with units m^(1/3), "
                            "not C_n^2 density (which has units m^(-2/3)). "
                            "The total free-atmosphere integral is sum(J_i)."
                        ),
                    },
                ),
                "cn2_ground_scalar": xr.Variable(
                    ("time",), cn2_ground,
                    attrs={
                        "long_name": "Ground-layer integrated turbulence strength (MASS Layer 0)",
                        "units": "m^(1/3)",
                        "note": (
                            "Turbulence integral for the ground layer (0 to ~500 m), "
                            "derived from the MASS-DIMM combined measurement. "
                            "Same unit convention as cn2_free_atmos."
                        ),
                    },
                ),
                "seeing": xr.Variable(
                    ("time",), seeing,
                    attrs={
                        "long_name": "MASS-DIMM integrated seeing",
                        "units": "arcsec",
                    },
                ),
                "temp_profile": xr.Variable(
                    ("time", "height_lhatpro"), temp_tensor,
                    attrs={
                        "long_name": "LHATPRO temperature vertical profile",
                        "units": "K",
                    },
                ),
                "wind_speed": xr.Variable(
                    ("time",), w_spd,
                    attrs={
                        "long_name": "Wind speed at 30 m tower",
                        "units": "m/s",
                    },
                ),
                "wind_dir": xr.Variable(
                    ("time",), w_dir,
                    attrs={
                        "long_name": "Wind direction at 30 m tower (meteorological convention)",
                        "units": "deg",
                    },
                ),
                "pressure": xr.Variable(
                    ("time",), pres,
                    attrs={
                        "long_name": "Surface atmospheric pressure",
                        "units": "hPa",
                    },
                ),
                "rh": xr.Variable(
                    ("time",), rh,
                    attrs={
                        "long_name": "Relative humidity at 2 m",
                        "units": "%",
                    },
                ),
                "night_id": xr.Variable(
                    ("time",), merged["night_id"].values,
                    attrs={
                        "long_name": "Observing-session identifier (YYYYMMDD of shifted local date)",
                    },
                ),
            },
            coords={
                "time": merged["time"].values,
                "height_mass": ("height_mass", MASS_HEIGHTS, {
                    "long_name": "MASS restoration layer center heights",
                    "units": "m",
                }),
                "height_lhatpro": ("height_lhatpro", LHATPRO_HEIGHTS, {
                    "long_name": "LHATPRO radiometric profile heights",
                    "units": "m",
                }),
            },
            attrs={
                "project": "otbench v2",
                "site": "ESO Paranal Observatory",
                "site_latitude": "-24.6272",
                "site_longitude": "-70.4048",
                "site_altitude_m": "2635",
                "description": "Optical Turbulence Tomography Benchmark (MASS + LHATPRO + Meteo)",
                "processing": (
                    f"Causal backward merge with tolerances: "
                    f"LHATPRO {LHATPRO_MERGE_TOLERANCE}, "
                    f"Meteo {METEO_MERGE_TOLERANCE}, "
                    f"MASS {MASS_MERGE_TOLERANCE}. "
                    f"Time regularized to 1-min cadence."
                ),
            },
        )
        return ds
