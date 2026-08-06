# ai_library

`ai_library` is a Python package that exposes model training, inference, metrics recording, cron scheduling, and package-based configuration management.

## Package overview

The package is structured so that `import ai_library` is safe and does not execute heavy runtime logic. Core user-facing functionality is available through lazy exports exposed from `ai_library/__init__.py`.

### Available top-level APIs

- `ai_library.validate_config(config_path=None)`
  - Loads and validates the package default `ai_library/config.yaml` if no path is provided.
  - Returns the parsed configuration dictionary.

- `ai_library.update_config(config_path=None, updates=...)`
  - Updates the package config file by default.
  - Supports a dictionary, a `[key, value]` pair, or a list of `[key, value]` pairs.

- `ai_library.show_config(config_path=None)`
  - Prints the current configuration to stdout.

- `ai_library.train()`
  - Loads configuration from the package config.
  - Reads data, builds the selected pipeline, and trains the model.

- `ai_library.infer()`
  - Loads package configuration.
  - Loads a saved model and performs inference.

- `ai_library.record`
  - Lazy-imported recorder module.
  - Use `ai_library.record.main()` or run the module directly to start metric collection.

- `ai_library.add_to_cron()` / `ai_library.remove_from_cron()`
  - Manage cron scheduling for recurring training runs.
  - Adds or removes a cron job that runs `python3 -m ai_library.codebase.setup.train`.

## Installation

From source:

```bash
python3 -m pip install -e .
```

From a built wheel or sdist:

```bash
python3 -m pip install dist/ai-library-swch-0.1.0-py3-none-any.whl
```

## Configuration

The default configuration file lives inside the package at `ai_library/config.yaml`.
This package is designed so that the default config is editable through the helper API:

```python
from ai_library import validate_config, update_config, show_config

config = validate_config()
print(config)

update_config(None, {"pipeline_type": "gmlp"})
show_config()
```

> Note: editing `ai_library/config.yaml` via `update_config(None, ...)` works cleanly during development or editable installs. If the package is installed from a read-only wheel, consider using an explicit config path or a custom location.

## Usage

### Training

```bash
python3 -m ai_library.codebase.setup.train
```

Or programmatically:

```python
from ai_library import train
train()
```

### Inference

```bash
python3 -m ai_library.codebase.setup.infer
```

Or programmatically:

```python
from ai_library import infer
infer()
```

### Metric recording

```bash
python3 -m ai_library.codebase.setup.record --out-dir ./data --run-seconds 60
```

Or programmatically:

```python
from ai_library import record
record.main()
```

### Cron scheduling

```python
from ai_library import add_to_cron, remove_from_cron
add_to_cron()
remove_from_cron()
```

## File structure

- `ai_library/`
  - `__init__.py` — lazy exports and package-level API surface.
  - `config.yaml` — default package config.
  - `codebase/`
    - `helpers/` — helper utilities such as `config_helper.py`.
    - `models/` — model pipeline classes.
    - `setup/` — training, inference, recording, and cron management modules.

## Notes

- `ai_library.record` is lazy-loaded so importing `ai_library` does not import heavy recorder dependencies until you actually access it.
- The package currently relies on package-relative config loading, so the package config file is the main runtime configuration source.
- Model training and inference rely on the dataset, saved model files, and other paths configured in `ai_library/config.yaml`.

## Recommended next improvements

- Add a runtime override for external config files.
- Add automated tests for training and inference flows.
- Document environment dependencies and the exact `requirements.txt` contents.
