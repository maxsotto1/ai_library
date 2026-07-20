def pivot_df(df):
    df_indexed = df.set_index("ts")
    resampled = df_indexed.groupby("metric")["value"].resample("300s").mean()
    resampled_df = resampled.reset_index()
    pivoted_df = resampled_df.pivot(index="ts", columns="metric", values="value")
    return pivoted_df
