# output/write_results.py
def save_dataframe(df, filepath, index=False):
    """
    Save DataFrame to a CSV file.
    """
    import pandas as pd
    try:
        df.to_csv(filepath, index=index)
        print(f"[INFO] Data saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save DataFrame: {e}")

def save_model_pickle(obj, filepath):
    """
    Save any Python object to a pickle file.
    """
    import pickle
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"[INFO] Model object saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save pickle: {e}")

def load_model_pickle(filepath):
    """
    Load any Python object from a pickle file.
    """
    import pickle
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load pickle: {e}")
        return None