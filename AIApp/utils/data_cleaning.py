import pandas as pd

def clean_data(file_path: str) -> dict:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        df = df.dropna().drop_duplicates()
        return {"cleaned_data": df.to_dict()}
    return {"error": "Only CSV supported for cleaning"}