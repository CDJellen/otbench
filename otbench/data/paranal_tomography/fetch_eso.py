import os
import requests
import hashlib
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

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
        
        # Payload builders
        self.builders = {
            "mass_paranal": self._get_mass_payload,
            "meteo_paranal": self._get_meteo_payload,
            "lhatpro_paranal": self._get_lhatpro_payload
            # SLODAR removed for v2.0 (Scale over Resolution)
        }

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

    def fetch_campaign(self, instrument: str, start_date: str, end_date: str, freq: str = "MS"):
        """Fetches data in chunks (Monthly by default) and computes SHAs."""
        if instrument not in self.builders:
            print(f"Skipping {instrument}: No payload builder.")
            return

        logical_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        manifest = {}

        print(f"Starting Campaign: {instrument} | {len(logical_dates)} Chunks")

        for i, date in enumerate(logical_dates):
            chunk_start = date.strftime("%Y-%m-%d")
            if i + 1 < len(logical_dates):
                chunk_end = logical_dates[i+1].strftime("%Y-%m-%d")
            else:
                chunk_end = end_date

            file_suffix = date.strftime("%Y-%m")
            filename = self.output_dir / f"{instrument}_{file_suffix}.csv"
            
            if filename.exists() and filename.stat().st_size > 1000:
                print(f"  [SKIP] {filename.name} exists.")
                continue

            sha = self._download_chunk(instrument, chunk_start, chunk_end, filename)
            if sha:
                manifest[filename.name] = sha

        manifest_path = self.output_dir / f"{instrument}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Campaign Complete. Manifest saved to {manifest_path}")

    def _download_chunk(self, instrument: str, start: str, end: str, path: Path) -> str:
        url_part = "lhatpro_profiles_paranal" if instrument == "lhatpro_paranal" else instrument
        url = f"{self.BASE_URL}/{url_part}/query"
        
        payload = self.builders[instrument]()
        payload["start_date"] = f"{start}..{end}"

        print(f"  Fetching {start} -> {end} ...")
        sha256_hash = hashlib.sha256()
        
        try:
            with self.session.post(url, data=payload, stream=True) as r:
                r.raise_for_status()
                iterator = r.iter_content(chunk_size=8192)
                try:
                    first_chunk = next(iterator)
                except StopIteration:
                    return None

                heading = first_chunk[:512].decode('utf-8', errors='ignore')
                if "No data returned" in heading:
                    return None

                with open(path, 'wb') as f:
                    f.write(first_chunk)
                    sha256_hash.update(first_chunk)
                    for chunk in iterator:
                        f.write(chunk)
                        sha256_hash.update(chunk)
                            
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"  [FAIL] {e}")
            if path.exists(): path.unlink()
            return None

if __name__ == "__main__":
    fetcher = ESOFetcher()
    # Strategic Pivot: 5-Year Campaign (2017-2022)
    START = "2017-01-01"
    END = "2022-01-01"
    
    fetcher.fetch_campaign("meteo_paranal", START, END, freq="MS")
    fetcher.fetch_campaign("lhatpro_paranal", START, END, freq="MS")
    fetcher.fetch_campaign("mass_paranal", START, END, freq="MS")
