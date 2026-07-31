from math import sqrt
from sklearn.metrics import mean_squared_error
from codebase.helpers.sliding_window import apply_sliding_window
import numpy as np
import xgboost as xgb
import os
import yaml
from codebase.helpers.to_saved_files import atomic_save
class XGBoost_pipeline:
        def __init__(self):
            self.scaler_y = None
            self.scaler_x = None
            self.clipping_min = None
            self.clipping_max = None
            self.model = None
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                self.params = config.get("xgb", {}).get("model_params", {})
                self.saved_files_dir = config.get("saved_files_dir")
                os.makedirs(self.saved_files_dir, exist_ok=True)

        def preprocess_splits(
            self,
            df,
            targets,
            splits=(0.7, 0.1, 0.2),
            n_future=96,
            n_past=288,
            stride=96,
            exclude_columns=None
            ):
            """
            Create all sliding windows first and then split
            the resulting samples into train/val/test.
            """

            X, Y, scaler_x, scaler_y = apply_sliding_window(
                df,
                targets,
                n_future,
                n_past,
                stride,
                exclude_columns,
            )

            self.scaler_x = scaler_x
            self.scaler_y = scaler_y

            n_samples = len(X)

            train_end = int(splits[0] * n_samples)
            val_end = int((splits[0] + splits[1]) * n_samples)

            x_train, y_train = X[:train_end], Y[:train_end]
            self.clipping_min = np.quantile(x_train, 0.01, axis=0)
            self.clipping_max = np.quantile(x_train, 0.99, axis=0)

            x_val, y_val = np.clip(X[train_end:val_end],self.clipping_min,self.clipping_max), Y[train_end:val_end]
            x_test, y_test = np.clip(X[val_end:],self.clipping_min,self.clipping_max), Y[val_end:]
            
            train_ds = (x_train, y_train)
            val_ds = (x_val, y_val)
            test_ds = (x_test, y_test)

            return train_ds, val_ds, test_ds
        
        def train(self, train_ds, val_ds, test_ds):

            x_train = train_ds[0]
            x_train = x_train.reshape(x_train.shape[0],-1)

            y_train = train_ds[1].astype(np.float32)
            y_train = y_train.reshape(y_train.shape[0],-1)

            x_val = val_ds[0]
            x_val = x_val.reshape(x_val.shape[0],-1)

            y_val = val_ds[1].astype(np.float32)
            y_val = y_val.reshape(y_val.shape[0],-1)

            x_test = test_ds[0]
            x_test = x_test.reshape(x_test.shape[0],-1)
            
            y_test = test_ds[1].astype(np.float32)
            y_test = y_test.reshape(y_test.shape[0],-1)

            model = xgb.XGBRegressor(**self.params)

            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            preds = self.scaler_y.inverse_transform(preds)
            targets = self.scaler_y.inverse_transform((y_test))
            rmse_val = sqrt(mean_squared_error(preds, targets))
            self.model = model
            atomic_save(self.model, f"{self.saved_files_dir}/trained_model_xgb.pkl")
            atomic_save(self.scaler_x, f"{self.saved_files_dir}/scaler_x_xgb.pkl")
            atomic_save(self.scaler_y, f"{self.saved_files_dir}/scaler_y_xgb.pkl")
            atomic_save(self.clipping_min, f"{self.saved_files_dir}/clipping_min_xgb.pkl")
            atomic_save(self.clipping_max, f"{self.saved_files_dir}/clipping_max_xgb.pkl")
            return model, rmse_val
    

        def preprocess_inference(self, df, targets, n_past, exclude_columns=None):

            exclude_columns = list(exclude_columns or [])
            target_cols = targets if isinstance(targets, (list, tuple)) else [targets]

            # 1. keep all required columns (features + targets)
            df_full = df.drop(columns=exclude_columns)

            # 2. scale features ONLY
            feature_df = df_full.drop(columns=target_cols)
            X_feat = self.scaler_x.transform(feature_df)
            # 3. scale targets separately
            Y = self.scaler_y.transform(df_full[target_cols])

            # 4. recombine in correct column order (VERY important)
            X = np.concatenate([X_feat, Y], axis=1)

            # 5. window: transpose to (channels, seq_len), clip with stored quantiles,
            # then flatten for XGBoost
            window = X[-n_past:]
            window = window.T[None, ...]  # shape (1, channels, seq_len)
            window = np.clip(window, self.clipping_min, self.clipping_max).astype(np.float32)
            window = window.reshape(1, -1)  # Flatten to (1, channels * seq_len)

            return window

        def infer(self, window):

            if self.model is None:
                raise RuntimeError("model was not trained yet, please run the train method first.")
            
            # XGBoost predict directly on numpy array
            preds = self.model.predict(window)
            preds = preds.reshape(-1, 1)
            preds = self.scaler_y.inverse_transform(preds)

            return preds