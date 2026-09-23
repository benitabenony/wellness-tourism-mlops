"""Data Registration: validate schema and print a dataset summary."""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

def main():
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"VALIDATION FAILED - missing columns: {missing}")
        sys.exit(1)
    print("VALIDATION PASSED - all expected columns are present.\n")
    print("Shape:", df.shape)
    print(df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print(f"\nTarget ({TARGET_COLUMN}) distribution:\n", df[TARGET_COLUMN].value_counts())

if __name__ == "__main__":
    main()
