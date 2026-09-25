"""Data Preparation: clean the data and create a stratified train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

# Neither column carries predictive signal: Unnamed: 0 is a row-index export
# artifact, and CustomerID is a unique identifier
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID"]


def clean_data(df):
    """Drop unnecessary columns, fix known label typos, dedupe, and impute."""
    df = df.copy()
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    # Fix inconsistent category labels found in the raw data
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Remove exact duplicate rows (117 found in the current snapshot)
    df = df.drop_duplicates()

    # Defensive imputation in case a future data drop has missing values,
    # even though the current snapshot has none
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

    # Stratify so both splits keep the same ~19%/81% purchase-rate balance
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Saved to the working directory — in CI, actions/upload-artifact hands
    # these four files to the model_building job as the "data-splits" artifact
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    print(f"Train shape: {Xtrain.shape}  Test shape: {Xtest.shape}")


if __name__ == "__main__":
    main()
