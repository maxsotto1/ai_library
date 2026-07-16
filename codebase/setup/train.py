import pandas as pd
from gmlp_class import gMLP_pipeline
from xgb_class import xgb_pipeline
import sys
from pathlib import Path
import yaml
import read

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
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

def get_last_window_data(train_window, train_horizon, targets, pipeline_type, splits, cols_to_drop):
    if pipeline_type == "gmlp":
        pipeline = gMLP_pipeline()
    elif pipeline_type == "xgb":
        pipeline = xgb_pipeline()
    df = pd.read_parquet("all_data.parquet")
    if pipeline_type == "gmlp":
        dls, test_dl = pipeline.preprocess_splits(df, targets, splits=splits, n_future=train_horizon, n_past=train_window, stride=stride, cols_to_drop=cols_to_drop)
    elif pipeline_type == "xgb":
        train_ds, val_ds, test_ds = pipeline.preprocess_splits(df, targets, n_future=train_horizon, n_past=train_window)
    return (dls, test_dl) if pipeline_type == "gmlp" else (train_ds, val_ds, test_ds)