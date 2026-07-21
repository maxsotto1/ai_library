
import pandas as pd
import pyarrow.dataset as ds
import yaml

DATA_DIR = "data/orchestrator_data/collector"
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
#get these from config.yaml instead of hardcoding them here!!
STANDARD_METRICS = config["STANDARD_METRICS"]

# 1. Load the dataset
dataset = ds.dataset(DATA_DIR, format="parquet", partitioning="hive")
#dataset = ds.dataset(DATA_DIR, format="parquet")
print("Schema:\n", dataset.schema, "\n")

# 2. Filter for ALL standard metrics using .isin() for partition pruning
table = dataset.to_table(
    filter=(ds.field("metric").isin(STANDARD_METRICS))
    # & (ds.field("date") == "2026-06-10")
)
unique_metrics = table.column("metric").unique()
print("Unique metrics in PyArrow Table:", unique_metrics)
print("Table Column Names:", table.column_names)
# 3. Process the DataFrame
df = table.to_pandas()

# Clean up the NaN values in the node column since we are in standard mode
df["node"] = df["node"].fillna("all_nodes")

df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
#df["ts"] = pd.to_datetime(df["collected_at"], utc=True)
df = df.sort_values("ts").reset_index(drop=True)

print(df.tail(10))
print(f"\n{len(df)} samples, {df['ts'].min()} .. {df['ts'].max()}")

# 4. For forecasting: Resample per metric (nodes are unified under 'all_nodes')
series = (
    df.set_index("ts")
    .groupby("metric")["value"]
    .resample("300s")
    .mean()
)
print("\nResampled (300s) preview:\n", series)
print("Metrics actually found in the dataset:", df["metric"].unique())
# 5. Save the full dataframe to a Parquet file without the row index numbers
df.to_parquet("data/orchestrator_data/data.parquet", index=False)