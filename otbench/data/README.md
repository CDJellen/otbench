## Datasets

This directory contains data used in `otbench` regression and forecasting tasks. Existing datasets are serialized in the NetCDF4 format and seek to comply with the CF conventions. As such, each dataset is self-describing, but also includes a `README.md` file with additional information and a `citation.md` file with a suggested citation. The `README.md` file is intended to be human-readable, while the `citation.md` file is intended to be machine-readable.

### Adding new Tasks from existing Datasets

To add a new task from an existing dataset:

1. Open `otbench/config/tasks.json`.
2. Navigate to the appropriate task type (`regression` or `forecasting`).
3. Add a new entry under the dataset name with the following required fields:
   - `description` — short description of the task.
   - `description_long` — detailed description including evaluation methodology.
   - `ds_name` — must match a key in `datasets.json`.
   - `obs_tz`, `obs_lat`, `obs_lon` — observing site metadata.
   - `train_idx`, `val_idx`, `test_idx` — index ranges as `"start:stop"` strings.
   - `dropna` — whether to drop rows with any missing values.
   - `log_transform` — whether to apply a base-10 log transform to the target.
   - `eval_metrics` — list of metric names from `otbench.eval`.
   - `target` — column name (string) or list of column names for vector targets.
   - `remove` — list of columns to exclude from the feature set.
4. For forecasting tasks, also provide `window_size` and `forecast_horizon`.
5. Run the test suite with `--use-synthetic-data` to verify the new task loads correctly.

### Adding new Datasets

To add a new dataset:

1. **Prepare the data** as a NetCDF4 file with a `time` dimension. Include CF-compliant attributes (units, long_name) on each variable.
2. **Create a directory** under `otbench/data/<dataset_name>/` containing:
   - The `.nc` data file.
   - A `README.md` describing the dataset, provenance, and variables.
   - A `citation.md` with BibTeX or plain-text citation.
3. **Register the dataset** in `otbench/config/datasets.json` with:
   - `local_data_path` — filename of the `.nc` file.
   - `citation_path` — path to citation file.
   - `remote_data_path` — URL where the raw data can be obtained.
   - `feature_map` — physical column mappings used by the benchmark runner (see existing entries for the schema).
   - `lazy` (optional, default `false`) — set to `true` for large datasets to enable memory-mapped loading.
4. **Add a synthetic data generator** in `otbench/dataset/synthetic.py` by adding a function to `SYNTHETIC_REGISTRY`. This enables CI testing without the real data file.
5. **Add at least one task** in `tasks.json` (see above).
6. **Run the test suite** with `--use-synthetic-data` to verify end-to-end.
