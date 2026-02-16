# Paranal Tomography Dataset

## Overview

This dataset constitutes the canonical benchmarking corpus for optical turbulence forecasting at the ESO Paranal Observatory. It combines layer-resolved turbulence profiles from MASS with thermodynamic vertical profiles from LHATPRO and surface meteorological measurements, enabling evaluation of tomographic reconstruction and forecasting algorithms. The schema is optimized for `xarray` ingestion and enforces strict causal ordering for time-series analysis.

## Provenance and Acquisition

Data are derived from co-located site-monitoring instruments at Paranal ($24^\circ 37'38''$S, $70^\circ 24'15''$W, 2635 m):

*   **MASS-DIMM (Multi-Aperture Scintillation Sensor / Differential Image Motion Monitor)**: Provides turbulence profiling of the free atmosphere (6 restoration layers from 500 m to 16 km) and an integrated ground-layer measurement. Also provides integrated seeing.
*   **LHATPRO (Low Humidity and Temperature Profiling Microwave Radiometer)**: Radiometric thermodynamic vertical profiling at 39 heights from ground level to 10 km.
*   **Paranal Meteorological Station**: Surface weather measurements including wind (30 m and 10 m towers), pressure, temperature, and humidity.

Raw data are fetched from the [ESO Ambient Conditions Database](https://archive.eso.org/cms/eso-data/ambient-conditions.html) via `fetch_eso.py` and transformed by `eso.py`.

## Physical Quantities

The MASS instrument measures scintillation indices and restores the vertical distribution of turbulence across discrete atmospheric layers. The restored quantity for each layer is the **turbulence integral**:

$$J_i = \int_{\text{layer}_i} C_n^2(h)\,dh$$

with SI units of $\mathrm{m}^{1/3}$. This is *not* the $C_n^2$ density (which has units $\mathrm{m}^{-2/3}$) but rather the path-integrated turbulence strength over each layer's vertical extent. The total free-atmosphere turbulence integral is obtained by summation: $J_{\mathrm{free}} = \sum_i J_i$.

The ESO archive labels these columns as "Cn2" (following widespread community convention), and the dataset variable names preserve this convention (`cn2_free_atmos`, `cn2_ground_scalar`) for compatibility. See the variable `note` attributes in the NetCDF file for the precise physical definition.

## Processing Methodology

The ingestion pipeline (`eso.py`) enforces causal integrity:

1.  **Regularization**: Each instrument stream is independently rounded to a 1-minute cadence and deduplicated (first observation per minute retained).
2.  **Sparse Union Backbone**: A master time index is constructed from the union of all instrument timestamps.
3.  **Causal Backward Merge**: Each instrument stream is aligned to the backbone using `pd.merge_asof` with `direction="backward"`, ensuring that a row at time $T$ contains only information available at or before $T$. No future data leakage occurs.
    *   LHATPRO: 5-minute tolerance (radiometric state persists between readings)
    *   Meteorology: 2-minute tolerance (surface conditions are persistent state)
    *   MASS: 2-minute tolerance (discrete measurement events)
4.  **Daylight Pruning**: Rows with no LHATPRO temperature reading (radiometer off during daytime) are removed.
5.  **Night Identification**: Observing sessions are assigned by shifting UTC timestamps by 16 hours (approximate local solar noon at 70.4$^\circ$W) so that the day boundary falls during daytime inactivity.

## Schema Definition

The dataset conforms to the following `xarray.Dataset` specification:

### Dimensions

*   `time`: Number of 1-minute epochs (varies with raw data coverage)
*   `height_mass`: 6 (free-atmosphere restoration layers)
*   `height_lhatpro`: 39 (thermodynamic profile levels)

### Coordinates

| Coordinate | Type | Values |
| :--- | :--- | :--- |
| `time` | `datetime64[ns]` | 1-minute cadence, UTC |
| `height_mass` | `int64` | `[500, 1000, 2000, 4000, 8000, 16000]` m |
| `height_lhatpro`| `int64` | `[0, 10, 30, 50, 75, 100, ..., 8000, 9000, 10000]` m |

### Data Variables

| Variable | Dimensions | Units | Description |
| :--- | :--- | :--- | :--- |
| `cn2_free_atmos` | `(time, height_mass)` | $\mathrm{m}^{1/3}$ | Layer-integrated turbulence strength $J_i$ for each MASS restoration layer. |
| `cn2_ground_scalar`| `(time)` | $\mathrm{m}^{1/3}$ | Ground-layer (0--500 m) turbulence integral from MASS-DIMM. |
| `seeing` | `(time)` | arcsec | MASS-DIMM integrated astronomical seeing. |
| `temp_profile` | `(time, height_lhatpro)`| K | LHATPRO radiometric temperature vertical profile. |
| `wind_speed` | `(time)` | m/s | Wind speed at 30 m tower. |
| `wind_dir` | `(time)` | deg | Wind direction at 30 m tower (meteorological convention, 0/360). |
| `pressure` | `(time)` | hPa | Surface atmospheric pressure. |
| `rh` | `(time)` | % | Relative humidity at 2 m. |
| `night_id` | `(time)` | int | Observing-session identifier (YYYYMMDD format of shifted local date). |

### Global Attributes

*   **project**: `otbench v2`
*   **site**: `ESO Paranal Observatory`
*   **site_latitude**: `-24.6272`
*   **site_longitude**: `-70.4048`
*   **site_altitude_m**: `2635`
*   **description**: `Optical Turbulence Tomography Benchmark (MASS + LHATPRO + Meteo)`
*   **processing**: Causal backward merge with per-instrument tolerances; 1-min regularization.

## Reproducing the Dataset

```bash
cd otbench/data/paranal_tomography
python pipeline.py
```

This fetches raw monthly CSVs from the ESO archive (if not already cached in `raw/`) and produces `paranal_tomography.nc`.
