import pandas as pd
import joblib

from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent


def predict_future_sales():

    # Load historical sales data
    data_path = BASE_DIR / "data" / "sales_data.csv"
    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Load trained model
    model_path = BASE_DIR / "models" / "sales_model.pkl"
    model = joblib.load(model_path)

    # Last three sales values
    sales_history = df["Sales"].astype(float).tolist()

    # Last available date
    last_date = df["Date"].iloc[-1]

    predictions = []

    # Predict next three months
    for i in range(1, 4):

        future_date = last_date + pd.DateOffset(months=i)

        # Prepare model input
        input_data = pd.DataFrame([{
            "Year": future_date.year,
            "Month": future_date.month,
            "Previous_Month": sales_history[-1],
            "Two_Months_Ago": sales_history[-2],
            "Three_Months_Ago": sales_history[-3]
        }])

        # Predict sales
        predicted_sales = model.predict(input_data)[0]

        # Store prediction
        predictions.append({
            "Date": future_date,
            "Predicted_Sales": round(float(predicted_sales), 2)
        })

        # Add predicted sales to history
        sales_history.append(float(predicted_sales))

    # Create forecast dataframe
    forecast_df = pd.DataFrame(predictions)

    # Display predictions
    print("\n======= FUTURE SALES FORECAST =======")

    print(forecast_df.to_string(index=False))

    # Save forecast report
    reports_path = BASE_DIR / "reports"
    reports_path.mkdir(exist_ok=True)

    forecast_df.to_csv(
        reports_path / "forecast_results.csv",
        index=False
    )

    print("\nForecast saved successfully!")
    print("File: reports/forecast_results.csv")


if __name__ == "__main__":
    predict_future_sales()