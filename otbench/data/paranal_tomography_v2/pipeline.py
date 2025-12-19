import sys
from fetch_eso import ESOFetcher
from eso import ESOParanalLoader

def run_pipeline():
    print("=== STAGE 1: INGESTION ===")
    fetcher = ESOFetcher(output_dir="raw")
    
    # We define the target range for the benchmark
    START = "2017-01-01"
    END = "2017-01-16" # Expanding to 4 years for production
    
    # Download all streams
    # Note: You must ensure PAYLOADS in fetch_eso.py covers all instruments
    for instrument in ["mass_paranal", "meteo_paranal", "slodar_paranal", "lhatpro_paranal"]:
        try:
            fetcher.fetch(instrument, start_date=START, end_date=END)
        except Exception as e:
            print(f"Critical Download Failure for {instrument}: {e}")
            sys.exit(1)

    print("\n=== STAGE 2: TRANSFORMATION ===")
    try:
        loader = ESOParanalLoader(raw_dir="raw")
        ds = loader.build_dataset()
        
        output_file = "paranal_v2_tomography.nc"
        ds.to_netcdf(output_file)
        print(f"\n[SUCCESS] Artifact created: {output_file}")
        print(f"Dimensions: {dict(ds.sizes)}")
        
    except Exception as e:
        print(f"ETL Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
