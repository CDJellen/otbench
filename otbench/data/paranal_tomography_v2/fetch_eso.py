import os
import requests
import hashlib
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List

class ESOFetcher:
    BASE_URL = "https://archive.eso.org/wdb/wdb/asm"
    
    def __init__(self, output_dir="raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) otbench/v2 Research",
            "Origin": "https://archive.eso.org",
            "Referer": "https://archive.eso.org/wdb/wdb/asm/mass_paranal/form"
        })
        
        # Payload builders (same as before)
        self.builders = {
            "mass_paranal": self._get_mass_payload,
            "meteo_paranal": self._get_meteo_payload,
            "lhatpro_paranal": self._get_lhatpro_payload,
            "slodar_paranal": self._get_slodar_payload
        }

    # ... [Insert _get_mass_payload, _get_meteo_payload, etc. from previous turn here] ...
    def _get_lhatpro_payload(self) -> Dict[str, Any]:
        payload = {"wdbo": "csv/download", "max_rows_returned": 100000, "platform": "%", "tab_lhatpro_id": "on", "integration": ""}
        for i in range(1, 40):
            payload[f"tab_ahum{i}"] = "on"
            payload[f"tab_temp{i}"] = "on"
        return payload

    def _get_meteo_payload(self) -> Dict[str, Any]:
        return {"wdbo": "csv/download", "max_rows_returned": 10000000, "integration": "", "tab_press": "on", "tab_temp1": "on", "tab_temp2": "on", "tab_temp3": "on", "tab_rhum2": "on", "tab_wind_dir1": "on", "tab_wind_dir1_180": "on", "tab_wind_dir2": "on", "tab_wind_dir2_180": "on", "tab_wind_speed1": "on", "tab_wind_speed2": "on"}

    def _get_mass_payload(self) -> Dict[str, Any]:
        return {"wdbo": "csv/download", "max_rows_returned": 10000000, "tab_fracgl": "on", "tab_turbfwhm": "on", "tab_tab_airmass": "on", "tab_dimm_turb": "on", "tab_dimm_alt": "on", "tab_turb1": "on", "tab_alt1": "on", "tab_turb2": "on", "tab_alt2": "on", "tab_turb3": "on", "tab_alt3": "on", "tab_turb4": "on", "tab_alt4": "on", "tab_turb5": "on", "tab_alt5": "on", "tab_turb6": "on", "tab_alt6": "on"}

    def _get_slodar_payload(self) -> Dict[str, Any]:
        """
        Corrected payload using exact field names from the ESO SLODAR form.
        """
        return {
            "wdbo": "csv/download",
            "max_rows_returned": 10000000,
            "integration": "",
            
            # --- The Scientific Targets (Mapped from your curl) ---
            "tab_fracgl500": "on",  # Cn2 fraction below 500m
            "tab_hrsfit": "on",     # Surface layer profile (High Res Surface Fit)
            "tab_step": "on",       # Cn2 layer thickness
            
            # The Vertical Profile Layers (Legacy name: cnsqs = Cn squared)
            "tab_cnsqs1": "on", "tab_cnsqs2": "on", 
            "tab_cnsqs3": "on", "tab_cnsqs4": "on",
            "tab_cnsqs5": "on", "tab_cnsqs6": "on", 
            "tab_cnsqs7": "on", "tab_cnsqs8": "on",
            
            # Optional: Uncomment if you want raw flux/noise data for debugging
            # "tab_flux1": "on", "tab_flux2": "on",
        }

    def fetch_campaign(self, instrument: str, start_date: str, end_date: str, freq: str = "MS"):
        """
        Fetches data in chunks (Monthly by default) and computes SHAs.
        
        Args:
            freq: Pandas frequency string. 'MS' = Month Start (Monthly chunks).
        """
        if instrument not in self.builders:
            print(f"Skipping {instrument}: No payload builder.")
            return

        # 1. Create Time Windows
        # logical_dates contains the start of each chunk
        logical_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        manifest = {}

        print(f"Starting Campaign: {instrument} | {len(logical_dates)} Chunks")

        for i, date in enumerate(logical_dates):
            # Define window: From this month start to next month start
            chunk_start = date.strftime("%Y-%m-%d")
            
            # Handle the last chunk gracefully
            if i + 1 < len(logical_dates):
                chunk_end = logical_dates[i+1].strftime("%Y-%m-%d")
            else:
                # If it's the last start date, go to the global end_date
                chunk_end = end_date

            # 2. Build Filename (e.g., paranal_meteo_2017-01.csv)
            # Using YYYY-MM ensures standard sorting
            file_suffix = date.strftime("%Y-%m")
            filename = self.output_dir / f"{instrument}_{file_suffix}.csv"
            
            # 3. Check Manifest/Cache (Optimization)
            # If file exists, we could check SHA here. For now, we overwrite or skip.
            if filename.exists() and filename.stat().st_size > 1000:
                print(f"  [SKIP] {filename.name} exists.")
                # Ideally, verify SHA here before skipping
                continue

            # 4. Download
            sha = self._download_chunk(instrument, chunk_start, chunk_end, filename)
            
            if sha:
                manifest[filename.name] = sha

        # 5. Save Manifest
        manifest_path = self.output_dir / f"{instrument}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Campaign Complete. Manifest saved to {manifest_path}")

    def _download_chunk(self, instrument: str, start: str, end: str, path: Path) -> str:
        """
        Downloads a chunk, validates it is not an ESO error message, and returns SHA256.
        """
        # [Same URL logic as before...]
        url_part = "lhatpro_profiles_paranal" if instrument == "lhatpro_paranal" else instrument
        url = f"{self.BASE_URL}/{url_part}/query"
        
        payload = self.builders[instrument]()
        payload["start_date"] = f"{start}..{end}"

        print(f"  Fetching {start} -> {end} ...")
        
        sha256_hash = hashlib.sha256()
        
        try:
            with self.session.post(url, data=payload, stream=True) as r:
                r.raise_for_status()
                
                # --- THE BOUNCER: Check for Fake CSVs ---
                # We peek at the first byte stream without consuming it improperly
                iterator = r.iter_content(chunk_size=8192)
                try:
                    first_chunk = next(iterator)
                except StopIteration:
                    print("  [WARN] Received empty response.")
                    return None

                # Check for the ESO error signature
                # Decode safe ascii to check string content
                heading = first_chunk[:512].decode('utf-8', errors='ignore')
                if "No data returned" in heading or "0 records were found" in heading:
                    print(f"  [SKIP] No data found for this period (Server Message).")
                    return None

                # If valid, write the first chunk and proceed
                with open(path, 'wb') as f:
                    f.write(first_chunk)
                    sha256_hash.update(first_chunk)
                    
                    # Stream the rest
                    for chunk in iterator:
                        f.write(chunk)
                        sha256_hash.update(chunk)
                            
            return sha256_hash.hexdigest()

        except Exception as e:
            print(f"  [FAIL] {e}")
            # cleanup partial files if needed
            if path.exists():
                path.unlink()
            return None

if __name__ == "__main__":
    fetcher = ESOFetcher()
    
    # Example: Fetch 1 Year of data in Monthly chunks
    # This prevents timeouts and allows easy resume
    fetcher.fetch_campaign("meteo_paranal", "2017-01-01", "2018-01-01", freq="MS")
    fetcher.fetch_campaign("lhatpro_paranal", "2017-01-01", "2018-01-01", freq="MS")
    fetcher.fetch_campaign("mass_paranal", "2017-01-01", "2018-01-01", freq="MS")
    fetcher.fetch_campaign("slodar_paranal", "2017-01-01", "2018-01-01", freq="MS")
