"""Data Preparation: clean the data and create a stratified train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID"]

def clean_data(df):
    df = df.copy()
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

def main():
    df = pd.read_csv(DATA_PATH)
    df_clean = clean_data(df)
    X = df_clean.drop(columns=["ProdTaken"])
    y = df_clean["ProdTaken"]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    print(f"Train shape: {Xtrain.shape}  Test shape: {Xtest.shape}")

if __name__ == "__main__":
    main()
