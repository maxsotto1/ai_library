import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

import ai_library
from ai_library import validate_config, update_config, show_config


def package_config_path() -> Path:
    return Path(ai_library.__file__).resolve().parent / "config.yaml"


def backup_package_config() -> str:
    return package_config_path().read_text(encoding="utf-8")


def restore_package_config(text: str) -> None:
    package_config_path().write_text(text, encoding="utf-8")


def write_package_config(config: dict) -> None:
    package_config_path().write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def create_synthetic_parquet(path: Path) -> None:
    timestamps = pd.date_range("2026-07-10 00:00:00", periods=24, freq="300s")
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append({"ts": ts, "metric": "cpu_util_instance", "value": float(i)})
        rows.append({"ts": ts, "metric": "extra_metric", "value": float(i * 2)})
    df = pd.DataFrame(rows)
    df.to_parquet(path)


def run_config_tests(temp_dir: Path) -> None:
    temp_config = temp_dir / "config_test.yaml"
    config = {
        "pipeline_type": "xgb",
        "window": 4,
        "horizon": 2,
        "stride": 1,
        "prediction_target": "cpu_util_instance",
        "STANDARD_METRICS": ["cpu_util_instance"],
        "cols_to_drop": ["ts"],
        "splits": [0.7, 0.1, 0.2],
        "xgb": {
            "model_params": {
                "n_estimators": 1,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0.1,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "tree_method": "hist",
                "predictor": "cpu_predictor",
                "random_state": 42,
                "n_jobs": 1,
                "objective": "reg:squarederror",
            }
        },
        "retrain_frequency": "30m",
        "data_frequency": "300s",
        "parquet_path": str(temp_dir / "data.parquet"),
        "data_dir": str(temp_dir / "collector"),
        "saved_files_dir": str(temp_dir / "saved_models"),
        "poll_interval_seconds": 5.0,
        "flush_max_rows": 100,
        "flush_max_seconds": 10.0,
    }
    temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    parsed = validate_config(temp_config)
    assert parsed["pipeline_type"] == "xgb"

    update_config(temp_config, {"horizon": 3})
    parsed = validate_config(temp_config)
    assert parsed["horizon"] == 3

    print("[config] validate_config/update_config/show_config passed")
    show_config(temp_config)


def run_cron_tests(temp_dir: Path) -> None:
    import ai_library.codebase.setup.cron_manager as cron_manager

    def fake_check_output(args, stderr=None):
        return b""

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.returncode = 0

        def communicate(self, input=None):
            self.stdin = input
            return (None, None)

    with patch("ai_library.codebase.setup.cron_manager.subprocess.check_output", fake_check_output):
        with patch("ai_library.codebase.setup.cron_manager.subprocess.Popen", return_value=FakePopen()):
            cron_manager.add_to_cron()
            cron_manager.remove_from_cron()

    print("[cron] add_to_cron/remove_from_cron dry-run passed")


def run_record_tests(temp_dir: Path) -> None:
    # The installed package exposes record via the package attribute,
    # but it is not a physical submodule at ai_library/record.py.
    record = ai_library.record

    assert record._coerce_value(3.14)[0] == 3.14
    assert record._coerce_value("42")[0] == 42.0
    row = record._make_row("cpu_util_instance", "node1", "standard", 1.0, 42)
    assert row["metric"] == "cpu_util_instance"

    out_dir = temp_dir / "record_output"
    sink = record.ParquetSink(str(out_dir), max_rows=1, max_seconds=0.0)
    sink.append([row])
    sink.maybe_flush(force=True)
    assert out_dir.exists()
    assert any(out_dir.rglob("*.parquet"))

    print("[record] internal sink and helper functions passed")


def run_pipeline_constructor_tests() -> None:
    # When the package is installed the package-relative imports inside
    # modules use a top-level `codebase` package name. Create a runtime
    # alias so those imports resolve against the installed `ai_library`.
    import sys
    sys.modules.setdefault("codebase", ai_library.codebase)

    from ai_library.codebase.models.gmlp_class import gMLP_pipeline
    from ai_library.codebase.models.iTransformer_class import iTransformer_pipeline

    gmlp = gMLP_pipeline()
    itrans = iTransformer_pipeline()
    assert gmlp.model is None
    assert itrans.model is None

    print("[pipeline] gMLP and iTransformer constructors passed")


def run_train_infer_tests(temp_dir: Path) -> None:
    temp_config = temp_dir / "config_train.yaml"
    config = {
        "pipeline_type": "xgb",
        "window": 4,
        "horizon": 2,
        "stride": 1,
        "prediction_target": "cpu_util_instance",
        "STANDARD_METRICS": ["cpu_util_instance"],
        "cols_to_drop": ["ts"],
        "splits": [0.7, 0.1, 0.2],
        "xgb": {
            "model_params": {
                "n_estimators": 1,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0.1,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "tree_method": "hist",
                "predictor": "cpu_predictor",
                "random_state": 42,
                "n_jobs": 1,
                "objective": "reg:squarederror",
            }
        },
        "retrain_frequency": "30m",
        "data_frequency": "300s",
        "parquet_path": str(temp_dir / "data.parquet"),
        "data_dir": str(temp_dir / "collector"),
        "saved_files_dir": str(temp_dir / "saved_models"),
        "poll_interval_seconds": 5.0,
        "flush_max_rows": 100,
        "flush_max_seconds": 10.0,
    }
    temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    original_validate = validate_config

    def patched_validate_config(config_path=None):
        return original_validate(temp_config if config_path is None else config_path)

    try:
        # patch the package-level validate_config for this run
        ai_library.validate_config = patched_validate_config
        # Create a local config.yaml in the temp dir so modules that call
        # `open("config.yaml")` (package code) will find it when we chdir.
        local_config_path = temp_dir / "config.yaml"
        local_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            create_synthetic_parquet(temp_dir / "data.parquet")
            ai_library.train()
            infer_output = ai_library.infer()
        finally:
            os.chdir(cwd)
        assert isinstance(infer_output, tuple) and len(infer_output) == 5
        predictions = infer_output[0]
        assert len(predictions) > 0
    finally:
        ai_library.validate_config = original_validate

    print("[train/infer] xgboost train and inference cycle passed")


def main() -> None:
    original_config_text = backup_package_config()
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        try:
            run_config_tests(temp_dir)
            run_cron_tests(temp_dir)
            run_record_tests(temp_dir)
            run_pipeline_constructor_tests()
            run_train_infer_tests(temp_dir)
            print("\nAll package smoke tests passed successfully.")
        except Exception as exc:
            print(f"Test failed: {exc}")
            raise
        finally:
            restore_package_config(original_config_text)


if __name__ == "__main__":
    main()
