("""Utility: validate and load configuration.

This module provides a single convenience function `validate_config`
which accepts a path to a YAML config file, verifies it exists,
parses it, and performs basic sanity checks on required keys and
value types.

The function returns the parsed config dict when validation succeeds
and raises clear exceptions (`FileNotFoundError`, `ValueError`) for
common problems so callers can fail fast.
""")

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import yaml

def validate_config(config_path: Path) -> Dict[str, Any]:
	"""Load and validate a YAML config file.

	Args:
		config_path: Path to the YAML configuration file.

	Returns:
		The parsed configuration dictionary.

	Raises:
		FileNotFoundError: if the config file does not exist.
		ValueError: if required keys are missing or types/values are invalid.
	"""
	p = Path(config_path)
	if not p.exists():
		raise FileNotFoundError(f"Configuration file not found: {p}")

	with p.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}

	required_keys = [
		"window",
		"horizon",
		"prediction_target",
		"pipeline_type",
		"splits",
		"cols_to_drop",
		"stride",
		"parquet_path",
		"data_frequency",
		"STANDARD_METRICS",
		"data_dir",
		"saved_files_dir",
		"poll_interval_seconds",
		"flush_max_rows",
		"flush_max_seconds",
	]
	missing = [k for k in required_keys if k not in config or config[k] in (None, "")]
	if missing:
		raise ValueError(f"Missing required config values: {', '.join(missing)}")

	pipeline_type = config["pipeline_type"]
	if pipeline_type not in {"gmlp", "xgb", "itransformer"}:
		raise ValueError(f"Unsupported pipeline_type: {pipeline_type}")

	for key in ("window", "horizon", "stride"):
		value = config[key]
		if not isinstance(value, int) or value <= 0:
			raise ValueError(f"{key} must be a positive integer, got {value!r}")

	splits = config["splits"]
	if not isinstance(splits, (list, tuple)) or len(splits) != 3:
		raise ValueError(f"splits must be a list or tuple of 3 values, got {splits!r}")
	if any(not isinstance(v, (int, float)) or not 0 < v < 1 for v in splits):
		raise ValueError(f"Each split value must be between 0 and 1, got {splits!r}")
	total = float(sum(splits))
	eps = 1e-8
	if total > 1.0 + eps:
		raise ValueError(f"Split values sum to more than 1.0 (got {total})")

	if not isinstance(config["cols_to_drop"], list):
		raise ValueError("cols_to_drop must be a list")

	if not isinstance(config.get("STANDARD_METRICS"), list):
		raise ValueError("STANDARD_METRICS must be a list of metric names")

	# data_dir and saved_files_dir should be non-empty strings
	for dir_key in ("data_dir", "saved_files_dir", "parquet_path"):
		if not isinstance(config.get(dir_key), str) or not config.get(dir_key).strip():
			raise ValueError(f"{dir_key} must be a non-empty string path")

	if not isinstance(config.get("poll_interval_seconds"), (int, float)) or config.get("poll_interval_seconds") <= 0:
		raise ValueError("poll_interval_seconds must be a positive number")

	if not isinstance(config.get("flush_max_rows"), int) or config.get("flush_max_rows") <= 0:
		raise ValueError("flush_max_rows must be a positive integer")

	if not isinstance(config.get("flush_max_seconds"), (int, float)) or config.get("flush_max_seconds") <= 0:
		raise ValueError("flush_max_seconds must be a positive number")

	# Optional: verify model-specific sections are mappings if present
	for model_key in ("gmlp", "xgb"):
		if model_key in config and config[model_key] is not None and not isinstance(config[model_key], dict):
			raise ValueError(f"{model_key} section must be a mapping of parameters")

	return config
