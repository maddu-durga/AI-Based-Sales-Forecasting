import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find project directory
BASE_DIR = Path(__file__).resolve().parent.parent


def analyze_sales():

    # Load sales dataset
    file_path = BASE_DIR / "data" / "sales_data.csv"
    df = pd.read_csv(file_path)

    # Convert date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Calculate sales statistics
    total_sales = df["Sales"].sum()
    average_sales = df["Sales"].mean()
    highest_sales = df["Sales"].max()
    lowest_sales = df["Sales"].min()

    # Find highest and lowest sales months
    highest_month = df.loc[df["Sales"].idxmax(), "Date"]
    lowest_month = df.loc[df["Sales"].idxmin(), "Date"]

    # Display results
    print("\n========== SALES ANALYSIS ==========")

    print("Total Sales:", total_sales)
    print("Average Sales:", round(average_sales, 2))
    print("Highest Sales:", highest_sales)
    print("Lowest Sales:", lowest_sales)

    print("Highest Sales Month:",
          highest_month.strftime("%B %Y"))

    print("Lowest Sales Month:",
          lowest_month.strftime("%B %Y"))

    # Create reports folder if necessary
    reports_path = BASE_DIR / "reports"
    reports_path.mkdir(exist_ok=True)

    # Create sales trend graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["Sales"],
        marker="o",
        label="Monthly Sales"
    )

    plt.title("Monthly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales Amount")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save graph
    graph_path = reports_path / "sales_trend.png"
    plt.savefig(graph_path)

    plt.close()

    print("\nSales trend graph saved successfully!")
    print("Graph location:", graph_path)


if __name__ == "__main__":
    analyze_sales()