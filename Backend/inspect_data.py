from .data_utils import load_raw_data
from pathlib import Path

if __name__ == "__main__":
    df = load_raw_data()
    print("Columns:\n", df.columns.tolist())
    print("\nFirst 5 rows:\n", df.head())