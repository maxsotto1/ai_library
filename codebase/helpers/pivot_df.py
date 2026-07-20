def pivot_df(df):
    df_indexed = df.set_index("ts")
    resampled = df_indexed.groupby("metric")["value"].resample("300s").mean()
    resampled_df = resampled.reset_index()
    pivoted_df = resampled_df.pivot(index="ts", columns="metric", values="value")
    plot_complete_timesteps(pivoted_df)
    pivoted_df = pivoted_df[pivoted_df['ts'] >= '2026-07-10']
    return pivoted_df

import matplotlib.pyplot as plt

def plot_complete_timesteps(pivoted_df):
    # 1. Drop any rows that contain NaN in ANY column
    complete_df = pivoted_df.dropna()
    
    if complete_df.empty:
        print("No timesteps exist where all metrics are non-NaN simultaneously!")
        return

    # 2. Set 'ts' as index if it isn't already, so it plots nicely on the X-axis
    if 'ts' in complete_df.columns:
        complete_df = complete_df.set_index('ts')

    # 3. Plot the complete periods
    plt.figure(figsize=(12, 6))
    
    # Loop through each metric column and plot it
    for column in complete_df.columns:
        plt.plot(complete_df.index, complete_df[column], label=column, marker='.', linestyle='-')
        
    plt.title("Metrics Over Complete Timesteps (No NaN Rows)")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save or display it
    plt.savefig("complete_metrics_plot.png")
