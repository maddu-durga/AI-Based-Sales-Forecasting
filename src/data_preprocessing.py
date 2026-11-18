import pandas as pd
from pathlib import Path


def load_sales_data():

    # Locate the project directory
    project_path = Path(__file__).resolve().parent.parent

    # Read sales dataset
    file_path = project_path / "data" / "sales_data.csv"

    df = pd.read_csv(file_path)

    print("\nOriginal Sales Dataset:")
    print(df.head())

    # Convert Date column into datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert Sales column into numeric format
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    # Remove missing values
    df = df.dropna(subset=["Date", "Sales"])

    # Remove duplicate records
    df = df.drop_duplicates(subset=["Date"])

    # Sort records by date
    df = df.sort_values(by="Date")

    # Reset index
    df = df.reset_index(drop=True)

    print("\nCleaned Sales Dataset:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTotal Sales:", df["Sales"].sum())

    return df


if __name__ == "__main__":
    load_sales_data()