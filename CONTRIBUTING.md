# Contributing to `otbench`

Thank you for your interest in contributing to `otbench`; we encourage bug reports, feature requests, code contributions, new benchmarks, and new datasets. Please read the following guidelines before submitting your contribution.

## Bug reports and feature requests

Please submit bug reports and feature requests as [GitHub issues](https://github.com/cdjellen/otbench/issues). When submitting a bug report, please include as much information as possible, including:
* A description of the bug (e.g., what you expected to happen and what actually happened)
* A minimal example that reproduces the bug
* The version of `otbench` you are using
* Your environment and package versions (e.g., Python version, NumPy version, etc.), preferably as a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually) or using `pip list --format=freeze`

## Code contributions

We welcome code contributions via [GitHub pull requests](https://github.com/cdjellen/otbench/pulls). Please follow the style of the existing code and include tests for any new functionality. We describe the minimal style guidelines in `setup.cfg` and use `yapf` to enforce them. You can run `yapf` on the entire codebase with the following command:

```bash
yapf -i -r otbench
```

### New features

If you are adding a new feature, please include the following:
* A description of the feature
* A minimal example that demonstrates the feature
* Tests for the feature (using `pytest`, see the `tests` directory for examples)

### Adding a new task

If you are adding a new task which builds from an existing data set, and relies exclusively on existing functionality, please include the following:
* A description of the task
* A minimal example that demonstrates the task
* Update the tasks manifest (`otbench/config/tasks.json`) to include the new task
* Tests for the task (using `pytest`, see the `tests` directory for examples)
* [Optional] Integration with the existing benchmarking framework (see `otbench/benchmark/bench_runner.py`)
* [Optional] Performance metrics for benchmark models (see `otbench/benchmark/experiments.json`)

If you are adding a new task which requires new functionality, please also include the following:
* A description of the new functionality
* Integration with exiting functionality, such that new code resides near related code
* Tests for the new functionality (using `pytest`, see the `tests` directory for examples)

If you are adding a new task that requires a new data set, please first review the [data set contribution guidelines](###adding-a-new-data-set). We recommend that you first add the data set, then add the task. We appreciate that this may not always be possible, and we will work with you to ensure that the task and dataset are added in a timely manner. Please consider including the following:
* A description of the data set
* A `README.md` and `citation.md` file in the data set directory (`otbench/data/<dataset_name>`)
* A new entry in the data sets manifest (`otbench/config/datasets.json`)
* Preferably, the data set itself (e.g., as a `.nc` file or a `.zip` file)
* If the data set is too large to include in the repository, please include a script that downloads the data set (e.g., `otbench/data/<dataset_name>/download.sh`)

## Data set contributions

### Adding a new data set

Before contributing a new data set, please consider the following:
* Is the data set publicly available?
* Is the data set small enough to include in the repository?
* If the data set is too large to include in the repository, is the data set easy to download?
* Is the data set relevant to the optical turbulence community?
* Does the dataset include macro-meterological data (e.g., wind speed, wind direction, temperature, etc.) along with scintillation or other turbulence data?

If you are adding a new data set, please see the `usna_cn2_sm` as an example. We recommend that you include the following:
* A description of the data set in the `README.md` file
* Detailed instructions for downloading the data set in the `README.md` file, if the data set is not included in the repository
* A citation for the data set in the `citation.md` file
* The data set itself (e.g., as a `.nc` file or a `.zip` file), preferably in a format that is easy to read in Python. If it is possible to format the data set as a `.nc` file, please consider doing so.

## Model contributions

### Adding a new benchmark model

Benchmark models and reference implementations for new approaches are highly encouraged. By including a reference implementation, we can ensure that the benchmarking framework is fair and that the results are reproducible. This will not only help compare the new approach to existing approaches, but also enable future researchers to compare their approaches to the new approach. If you are adding a new benchmark model, please include the following:
* A description of the benchmark model
* A minimal example that demonstrates the benchmark model
* A reference implementation of the benchmark model (preferably in Python)
* Preferably include tests for the new model (using `pytest`, see the `tests` directory for examples)
* [Optional] Performance metrics for the benchmark model (see `otbench/benchmark/experiments.json`) generated using the `otbench.benchmark.bench_runner` module.
* [Optional] A reference to the paper that describes the benchmark model in a `citation.md` file

## Documentation and other contributions

We welcome documentation contributions, including:
* Improvements to the documentation
* New tutorials
* New examples
* Edits to in-line documentation and docstrings

## Concluding remarks

We appreciate your interest in contributing to `otbench`. If you have any questions, please feel free to open an issue or contact the maintainers directly. We will make every effort to respond to your questions and contributions in a timely manner, and to work with you to ensure that your contributions are included in the repository.

Thank you for your interest in improving `otbench` and in advancing the field of optical turbulence research!
