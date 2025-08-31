# utils/helpers.py
def safe_merge(df1, df2, on, how='left'):
    """
    Perform a safe merge by warning about duplicate keys and nulls post-merge.
    """
    import pandas as pd
    merged = pd.merge(df1, df2, on=on, how=how, indicator=True)
    dup_keys = merged[merged.duplicated(on, keep=False)]
    if not dup_keys.empty:
        print(f"[Warning] Duplicate keys found during merge on {on}.")
    nulls = merged[merged.isnull().any(axis=1)]
    if not nulls.empty:
        print(f"[Warning] Null values introduced during merge on {on}.")
    return merged

def log_stage(message):
    print(f"[LOG] {message}")

def round_spread_line(prob, translation_dict):
    """
    Round a probability to 3 decimals and map to implied spread
    """
    return translation_dict.get(round(prob, 3), None)

def weighted_average(values, weights):
    """
    Compute weighted average with aligned lists.
    """
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)