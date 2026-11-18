import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

def train_sales_model():

    # Load dataset
    file_path = BASE_DIR / "data" / "sales_data.csv"
    df = pd.read_csv(file_path)

    # Prepare data
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Create date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Create previous sales features
    df["Previous_Month"] = df["Sales"].shift(1)
    df["Two_Months_Ago"] = df["Sales"].shift(2)
    df["Three_Months_Ago"] = df["Sales"].shift(3)

    # Remove rows without previous sales
    df = df.dropna().reset_index(drop=True)

    # Select input features
    features = [
        "Year",
        "Month",
        "Previous_Month",
        "Two_Months_Ago",
        "Three_Months_Ago"
    ]

    X = df[features]
    y = df["Sales"]

    # Use earlier months for training
    # Use the latest six months for testing
    split_index = len(df) - 6

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n====== MODEL EVALUATION ======")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("Mean Absolute Error:", round(mae, 2))
    print("Root Mean Squared Error:", round(rmse, 2))

    # Save trained model
    model_path = BASE_DIR / "models"
    model_path.mkdir(exist_ok=True)

    joblib.dump(model, model_path / "sales_model.pkl")

    # Save feature names for prediction module
    joblib.dump(features, model_path / "feature_names.pkl")

    # Save evaluation results
    reports_path = BASE_DIR / "reports"
    reports_path.mkdir(exist_ok=True)

    results = pd.DataFrame({
        "Date": df["Date"].iloc[split_index:],
        "Actual_Sales": y_test.values,
        "Predicted_Sales": predictions
    })

    results.to_csv(
        reports_path / "model_evaluation.csv",
        index=False
    )

    print("\nModel saved successfully!")
    print("Evaluation report saved successfully!")

if __name__ == "__main__":
    train_sales_model()