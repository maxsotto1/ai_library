from models.xgb_class import XGBoost_pipeline
from models.gmlp_class import gMLP_pipeline
from models.iTransformer_class import iTransformer_pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_parquet("data/burn_cpu_burn_data/a.parquet")

    #setup gmlp pipeline
    gmlp_pipeline = gMLP_pipeline()
    dls, test_dls = gmlp_pipeline.preprocess_splits(df,["cpu_01_busy"],[0.7,0.1,0.2],96,96,1,["m_id", "sample_time"])
    model, rmse = gmlp_pipeline.train(dls, test_dls)
    preprocess_inference = gmlp_pipeline.preprocess_inference(df.iloc[-384:-96],["cpu_01_busy"],96,["m_id", "sample_time"])
    preds_gmlp = gmlp_pipeline.infer(preprocess_inference)
    print(rmse)
    
    #setup xgboost pipeline
    xgboost_pipeline = XGBoost_pipeline()
    train, val, test = xgboost_pipeline.preprocess_splits(df,["cpu_01_busy"],[0.7,0.1,0.2],96,288,96,["m_id", "sample_time"])
    model, rmse = xgboost_pipeline.train(train,val,test)
    preprocess_inference = xgboost_pipeline.preprocess_inference(df.iloc[-384:-96],["cpu_01_busy"],288,["m_id", "sample_time"])
    preds_xgb = xgboost_pipeline.infer(preprocess_inference)
    print(rmse)

    #setup iTransformer pipeline
    itransformer_pipeline = iTransformer_pipeline()
    dls, test_dls = itransformer_pipeline.preprocess_splits(df,["cpu_01_busy"],[0.7,0.1,0.2],96,96,1,["m_id", "sample_time"])
    model, rmse = itransformer_pipeline.train(dls, test_dls)
    preprocess_inference = itransformer_pipeline.preprocess_inference(df.iloc[-384:-96],["cpu_01_busy"],96,["m_id", "sample_time"])
    preds_itf = itransformer_pipeline.infer(preprocess_inference)
    print(rmse)

    truths = df.iloc[-96:][["cpu_01_busy"]].values

    truths = truths.reshape(-1)
    preds_gmlp = np.array(preds_gmlp).reshape(-1)
    preds_xgb = np.array(preds_xgb).reshape(-1)
    preds_itf = np.array(preds_itf).reshape(-1)

    # gMLP metrics
    gmlp_rmse = np.sqrt(mean_squared_error(truths, preds_gmlp))
    gmlp_mae = mean_absolute_error(truths, preds_gmlp)
    gmlp_r2 = r2_score(truths, preds_gmlp)

    # XGBoost metrics
    xgb_rmse = np.sqrt(mean_squared_error(truths, preds_xgb))
    xgb_mae = mean_absolute_error(truths, preds_xgb)
    xgb_r2 = r2_score(truths, preds_xgb)

    # iTransformer metrics
    itf_rmse = np.sqrt(mean_squared_error(truths, preds_itf))
    itf_mae = mean_absolute_error(truths, preds_itf)
    itf_r2 = r2_score(truths, preds_itf)

    print("gMLP:")
    print(f"  RMSE: {gmlp_rmse:.4f}")
    print(f"  MAE : {gmlp_mae:.4f}")
    print(f"  R²  : {gmlp_r2:.4f}")

    print("\nXGBoost:")
    print(f"  RMSE: {xgb_rmse:.4f}")
    print(f"  MAE : {xgb_mae:.4f}")
    print(f"  R²  : {xgb_r2:.4f}")

    print("\niTransformer:")
    print(f"  RMSE: {itf_rmse:.4f}")
    print(f"  MAE : {itf_mae:.4f}")
    print(f"  R²  : {itf_r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(truths, label="Ground Truth", marker='o')
    plt.plot(preds_gmlp, label="gMLP Predictions", marker='x')
    plt.plot(preds_xgb, label="XGBoost Predictions", marker='s')
    plt.plot(preds_itf, label="iTransformer Predictions", marker='^')
    plt.title("gMLP vs XGBoost Predictions")
    plt.xlabel("Time Steps")
    plt.ylabel("CPU Busy")
    plt.legend()
    plt.grid()
    plt.savefig("gmlp_vs_xgb_predictions.png")
