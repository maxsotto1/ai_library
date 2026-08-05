import pandas as pd
from codebase.models.gmlp_class import gMLP_pipeline
from codebase.models.xgb_class import XGBoost_pipeline
from pathlib import Path
import yaml
from codebase.models.iTransformer_class import iTransformer_pipeline
import codebase.setup.read
from codebase.helpers.pivot_df import pivot_df

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
#get train window, train_horizon, targets, pipeline_type from config.yaml
train_window = config["window"]
train_horizon = config["horizon"]
targets = config["prediction_target"]
pipeline_type = config["pipeline_type"]
splits = config["splits"]
cols_to_drop = config["cols_to_drop"]
stride = config["stride"]
parquet_path = config["parquet_path"]
resample_frequency = config["data_frequency"]
def get_last_window_data_and_train(train_window, train_horizon, targets, pipeline_type, splits, cols_to_drop):
    if pipeline_type == "gmlp":
        pipeline = gMLP_pipeline()
    elif pipeline_type == "xgb":
        pipeline = XGBoost_pipeline()
    elif pipeline_type == "itransformer":
        pipeline = iTransformer_pipeline()
    df = pd.read_parquet(parquet_path)
    # Basic safety checks: parquet should contain time column and enough rows
    if df.empty:
        raise ValueError(f"No data found in parquet file: {parquet_path}")
    if "ts" not in df.columns:
        raise ValueError(f"Parquet file {parquet_path} missing required 'ts' column")
    required_rows = int(train_window) + int(train_horizon)
    if len(df) < required_rows:
        raise ValueError(
            f"Not enough rows in parquet file: found {len(df)}, need at least {required_rows} (window + horizon)"
        )

    df = pivot_df(df)
    df = df.set_index("ts").resample(resample_frequency).mean().interpolate("linear").bfill().ffill().reset_index()
    if pipeline_type == "gmlp":
        dls, test_dl = pipeline.preprocess_splits(df, targets, splits, train_horizon, train_window, stride, cols_to_drop)
        model, rmse, conformal_q = pipeline.train(dls, test_dl)
    elif pipeline_type == "xgb":
        train_ds, val_ds, test_ds = pipeline.preprocess_splits(df,targets, splits, train_horizon, train_window, stride, cols_to_drop)
        model, rmse, conformal_q = pipeline.train(train_ds, val_ds, test_ds)
    elif pipeline_type == "itransformer":
        dls, test_dls = pipeline.preprocess_splits(df, targets, splits, train_horizon, train_window, stride, cols_to_drop)
        model, rmse, conformal_q = pipeline.train(dls, test_dls)
    print(f"Trained {pipeline_type} model with RMSE: {rmse}")

get_last_window_data_and_train(train_window, train_horizon, targets, pipeline_type, splits, cols_to_drop)