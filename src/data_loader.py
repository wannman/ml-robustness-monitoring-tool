import pandas as pd
from pathlib import Path

def load_data(file_path: Path) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(file_path)
        return df
    
    except Exception as ex:
        print("Failed loading data")
        return pd.DataFrame()

