# Paranal Tomography Dataset

## Overview

This dataset constitutes the canonical benchmarking corpus for optical turbulence forecasting at the ESO Paranal Observatory. It aggregates vertical profiles of the refractive index structure constant ($C_n^2$) and meteorological covariates, enabling precise evaluation of tomographic reconstruction and prediction algorithms. The schema is optimized for `xarray` ingestion, maintaining strict causal ordering for time-series analysis.

## Provenance and Acquisition

Data are derived from a co-located suite of site testing instrumentation at Paranal ($24^\circ 37'38''S, 70^\circ 24'15''W$):
*   **MASS (Multi-Aperture Scintillation Sensor)**: Low-resolution turbulence profiling of the free atmosphere.
*   **SLODAR (Slope Detection and Ranging)**: High-resolution profiling of the surface layer.
*   **LHATPRO (Low Humidity and Temperature Profiling Microwave Radiometer)**: Radiometric thermodynamic vertical profiling.

## Processing Methodology

The ingestion pipeline enforces rigorous causal integrity:
1.  **Causal Backward Merge**: Observations are unified with a strict 2-minute causal tolerance window.
2.  **Regularization**: The time domain is regularized to a 1-minute cadence to ensure uniform temporal pacing.

## Schema Definition

The dataset conforms to the following `xarray.Dataset` specification:

### Dimensions

*   `time`: 375,758 epochs (~2017-06 to 2020-03)
*   `height_mass`: 6 strata (Free Atmosphere)
*   `height_slodar`: 8 strata (Boundary Layer)
*   `height_lhatpro`: 39 strata (Thermodynamic Profile)

### Coordinates

| Coordinate | Type | domain |
| :--- | :--- | :--- |
| `time` | `datetime64[ns]` | `2017-06-01T00:01:00` ... `2020-03-...` |
| `height_mass` | `int64` | `[500, 1000, 2000, 4000, 8000, 16000]` |
| `height_lhatpro`| `int64` | `[0, 10, 30, 50 ... 7000, 8000, 9000, 10000]` |
| `height_slodar` | `int64` | `[1, 2, 3, 4, 5, 6, 7, 8]` |

### Data Variables

| Variable | Dimensions | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `cn2_free_atmos` | `(time, height_mass)` | `float64` | $C_n^2$ profiles derived from MASS. |
| `cn2_boundary` | `(time, height_slodar)` | `float64` | $C_n^2$ profiles derived from SLODAR. |
| `cn2_ground_scalar`| `(time)` | `float64` | Scalar ground-level $C_n^2$ values. |
| `seeing` | `(time)` | `float64` | Integrated astronomical seeing. |
| `temp_profile` | `(time, height_lhatpro)`| `float64` | Vertical temperature profile. |
| `wind_speed` | `(time)` | `float64` | Scalar wind speed. |
| `wind_dir` | `(time)` | `float64` | Wind direction azimuth. |
| `pressure` | `(time)` | `float64` | Surface atmospheric pressure. |
| `rh` | `(time)` | `float64` | Surface relative humidity. |
| `night_id` | `(time)` | `int64` | Sequential identifier for contiguous observing nights. |

### Global Attributes

*   **project**: `otbench v2`
*   **site**: `ESO Paranal`
*   **description**: `Tomographic Benchmark (MASS+SLODAR+LHATPRO)`
*   **processing**: `Causal Backward Merge (2min tolerance), 1-min Regularization`
