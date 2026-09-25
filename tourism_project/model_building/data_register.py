"""Data Registration: validate schema (columns + dtypes) and print a dataset summary."""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"

# All columns named in the business data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

# Expected data type category for each column: "numeric" or "categorical".
# Catches data-drift issues a name-only check would miss, e.g. a numeric
# column silently turning into text in a future data drop.
EXPECTED_DTYPES = {
    "CustomerID": "numeric", "ProdTaken": "numeric", "Age": "numeric",
    "CityTier": "numeric", "DurationOfPitch": "numeric",
    "NumberOfPersonVisiting": "numeric", "NumberOfFollowups": "numeric",
    "PreferredPropertyStar": "numeric", "NumberOfTrips": "numeric",
    "Passport": "numeric", "PitchSatisfactionScore": "numeric",
    "OwnCar": "numeric", "NumberOfChildrenVisiting": "numeric",
    "MonthlyIncome": "numeric",
    "TypeofContact": "categorical", "Occupation": "categorical",
    "Gender": "categorical", "ProductPitched": "categorical",
    "MaritalStatus": "categorical", "Designation": "categorical",
}


def main():
    df = pd.read_csv(DATA_PATH)

    # Gate 1: every expected column must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"VALIDATION FAILED - missing columns: {missing}")
        sys.exit(1)

    # Gate 2: every column's type must match what the rest of the pipeline
    # expects (a numeric column that silently turned into text would break
    # StandardScaler downstream, for example)
    type_mismatches = []
    for col, expected_kind in EXPECTED_DTYPES.items():
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        actual_kind = "numeric" if is_numeric else "categorical"
        if actual_kind != expected_kind:
            type_mismatches.append(
                "{}: expected {}, found {} ({})".format(col, expected_kind, actual_kind, df[col].dtype)
            )
    if type_mismatches:
        print("VALIDATION FAILED - column type mismatches:")
        for m in type_mismatches:
            print("  - " + m)
        sys.exit(1)

    # Both gates passed - print a summary so a human (or the CI log) can
    # sanity-check the dataset before the pipeline proceeds
    print("VALIDATION PASSED - all expected columns are present with the correct type.")
    print("")
    print("Shape:", df.shape)
    print(df.dtypes)
    print("")
    print("Missing values:")
    print(df.isnull().sum())
    print("")
    print("Target ({}) distribution:".format(TARGET_COLUMN))
    print(df[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    main()
