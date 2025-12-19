### **1. Surface Meteorology**

* **File Name:** `paranal_meteo_surface.csv`
* **Description:** High-frequency surface weather logs from the Vaisala station. It captures the mechanical drivers of ground-layer turbulence (wind shear at 10m/30m) and the thermal stability conditions (temperature gradients between 2m and ground).
* **Key Columns:** `Wind Speed at 30m`, `Ambient Temperature at ground`.

### **2. Boundary Layer Profiler (SLODAR)**

* **File Name:** `paranal_slodar_boundary_layer.csv`
* **Description:** High-resolution optical turbulence profiles focusing specifically on the lowest 500 meters of the atmosphere. Unlike MASS (which sees high altitude), this dataset resolves the "surface layer" structure critical for ground-to-ground laser propagation.
* **Key Columns:** `Cn2 fraction below 500m`, `Surface layer profile`.

### **3. Thermodynamic Profiler (LHATPRO)**

* **File Name:** `paranal_lhatpro_profiles.csv`
* **Description:** Vertical profiles of temperature and humidity up to 10 km altitude, derived from microwave radiometry. This provides the *thermodynamic input vector* () for your physics-informed model, allowing it to see stability indices aloft without launching a weather balloon.
* **Key Columns:** `Temperature [K] at 0[m]...10000[m]`, `Absolute Humidity`.

### **4. Integrated Turbulence & Free Atmosphere (MASS-DIMM)**

* **File Name:** `paranal_mass_dimm_profile.csv`
* **Description:** The primary "Ground Truth" target for the full atmospheric column. It combines the **DIMM** (total integrated seeing) with the **MASS** (low-resolution vertical profile at 0.5km, 1km, ..., 16km), enabling the Vertical Profiling task.
* **Key Columns:** `MASS-DIMM Seeing`, `Layer 0 Cn2` (Ground), `Layer 1...6 Cn2` (High Altitude).
