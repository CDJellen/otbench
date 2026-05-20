from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd
import xarray as xr
from otbench.config import settings


def generate_paranal_tomography(seed: int = 2020) -> xr.Dataset:
    """
    Generates a synthetic Paranal Tomography dataset for testing purposes.
    
    The schema matches the one defined in otbench/data/paranal_tomography/README.md.
    Values are physically plausible (e.g., positive Cn2) but random.
    
    Args:
        seed: Random seed for determinism.
        
    Returns:
        xr.Dataset: The synthetic dataset.
    """
    rng = np.random.default_rng(seed)

    n_time = 2000
    times = pd.date_range("2017-06-01T00:01:00", periods=n_time, freq="min")

    height_mass = np.array([500, 1000, 2000, 4000, 8000, 16000])
    # Real LHATPRO non-uniform vertical grid (fine in boundary layer, coarser aloft)
    height_lhatpro = np.array([
        0, 10, 30, 50, 75, 100, 125, 150, 200, 250, 325, 400, 475, 550, 625, 700,
        800, 900, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2200, 2500, 2800,
        3100, 3500, 3900, 4400, 5000, 5600, 6200, 7000, 8000, 9000, 10000
    ])

    def random_cn2(shape):
        return rng.lognormal(mean=-16, sigma=1, size=shape)

    data_vars = {
        "cn2_free_atmos": (("time", "height_mass"), random_cn2((n_time, len(height_mass)))),
        "cn2_ground_scalar": (("time",), random_cn2(n_time)),
        "seeing": (("time",), rng.uniform(0.4, 1.5, n_time)),  # Arcseconds
        "temp_profile": (("time", "height_lhatpro"), rng.normal(273, 5, (n_time, len(height_lhatpro)))),
        "wind_speed": (("time",), rng.uniform(0, 20, n_time)),
        "wind_dir": (("time",), rng.uniform(0, 360, n_time)),
        "pressure": (("time",), rng.normal(740, 5, n_time)),  # hPa at altitude
        "rh": (("time",), rng.uniform(0, 100, n_time)),
        "night_id": (("time",), np.repeat(np.arange(1, 11), n_time // 10)),  # 10 nights, equal split
    }

    coords = {
        "time": times,
        "height_mass": height_mass,
        "height_lhatpro": height_lhatpro,
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    ds.attrs = {
        "project": "otbench v2",
        "site": "ESO Paranal",
        "description": "Synthetic Tomographic Benchmark (MASS+LHATPRO)",
        "processing": "Synthetic generator",
        "is_synthetic": True
    }

    return ds


# Registry for synthetic data generators
# Dictionary mapping dataset name (as in datasets.json) to generator function
SYNTHETIC_REGISTRY: Dict[str, Callable[[], xr.Dataset]] = {"paranal_tomography": generate_paranal_tomography}
