from tsai.data.core import TSDatasets, TSDataLoaders, TSDataLoader
from tsai.models.gMLP import gMLP
from tsai.learner import Learner
from fastai.callback.tracker import EarlyStoppingCallback
from torch.nn import MSELoss
from fastai.metrics import rmse
from math import sqrt
from sklearn.metrics import mean_squared_error
from codebase.preprocess.sliding_window import apply_sliding_window
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from codebase.helpers.to_saved_files import atomic_save

class gMLP_pipeline:
        def __init__(self):
            self.scaler_y = None
            self.scaler_x = None
            self.clipping_min = None
            self.clipping_max = None
            self.model = None

        def preprocess_splits(
            self,
            df,
            targets,
            splits=(0.7, 0.1, 0.2),
            n_future=96,
            n_past=288,
            stride=96,
            exclude_columns=None,
            bs=12,
            ):
            """
            Create all sliding windows first and then split
            the resulting samples into train/val/test.
            """
            print(df.head())
            print(f"Targets: {targets}")
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
            
            train_ds = TSDatasets(x_train, y_train)
            val_ds = TSDatasets(x_val, y_val)
            test_ds = TSDatasets(x_test, y_test)

            dls = TSDataLoaders.from_dsets(
                train_ds,
                val_ds,
                bs=bs
            )

            test_dl = TSDataLoader(test_ds, bs=bs)

            return dls, test_dl
        
        def train(self, dls, test_dl, lr=1e-3, epochs=10, patience=10):
        
            xb, yb = dls.one_batch()
            print(xb.shape, yb.shape)
            print("X batch sample data:\n", xb[:2])
            print("Y batch sample data:\n", yb[:2])
            model = gMLP(
                c_in=xb.shape[1],
                c_out=yb.shape[1],
                seq_len=xb.shape[2],
                d_model=64,
                d_ffn=128,
                depth=2,
                patch_size=1
            )

            learn = Learner(
                    dls,
                    model,
                    loss_func=MSELoss(),
                    metrics=rmse,
                    cbs=[EarlyStoppingCallback(monitor="valid_loss", patience=patience)]
                )

            learn.fit_one_cycle(epochs, lr)
            preds, targets = learn.get_preds(dl=test_dl)
            preds = self.scaler_y.inverse_transform(preds)
            targets = self.scaler_y.inverse_transform(targets)
            rmse_val = sqrt(mean_squared_error(preds, targets))
            
            ####plotting (for debugging)
            '''
            plt.figure(figsize=(8, 3))
            plt.plot(targets.reshape(-1,1)[::96], label="target")
            plt.plot(preds.reshape(-1,1)[::96], label="preds", alpha=0.7)
            plt.legend()
            plt.xlabel("forecast step (aligned)")
            plt.ylabel("cpu_01_busy")
            plt.tight_layout()
            plt.show()
            '''

            self.model = model
            atomic_save(self.scaler_x, "scaler_x.pkl")  
            atomic_save(self.scaler_y, "scaler_y.pkl")
            atomic_save(self.model, "trained_model.pth", use_pytorch=True)
            atomic_save(self.clipping_min, "clipping_min.pkl")
            atomic_save(self.clipping_max, "clipping_max.pkl")
            return model, rmse_val
        
        def preprocess_inference(self, df, targets, n_past, exclude_columns=None):

            exclude_columns = list(exclude_columns or [])
            target_cols = targets if isinstance(targets, (list, tuple)) else [targets]

            # 1. keep all required columns (features + targets)
            df_full = df.drop(columns=exclude_columns)

            # 2. scale features ONLY
            feature_df = df_full.drop(columns=target_cols)
            X_feat = self.scaler_x.transform(feature_df)
            X_feat = np.clip(X_feat, self.clipping_min, self.clipping_max).astype(np.float32)

            # 3. scale targets separately
            Y = self.scaler_y.transform(df_full[target_cols])

            # 4. recombine in correct column order (VERY important)
            X = np.concatenate([X_feat, Y], axis=1)

            # 6. window
            window = X[-n_past:]
            window = window.T[None, ...]
            window = np.clip(window, self.clipping_min, self.clipping_max).astype(np.float32)

            return window

        def infer(self, window, device="cpu"):

            window = torch.tensor(window, dtype=torch.float32).to(device)

            if self.model is None:
                raise RuntimeError("model was not trained yet, please run the train method first.")
            
            model = self.model.to(device)
            model.eval()

            with torch.no_grad():
                preds = model(window)

            preds = preds.cpu().numpy().reshape(-1,1)
            preds = self.scaler_y.inverse_transform(preds)

            return preds