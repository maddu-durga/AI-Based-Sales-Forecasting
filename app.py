import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 AI-Based Sales Forecasting System")

st.markdown(
    "Analyze historical sales and predict future sales "
    "using Machine Learning."
)

# Sidebar
st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard",
        "Upload Dataset",
        "Sales Analysis",
        "Sales Prediction",
        "About Project"
    ]
)

# Load historical dataset
data_path = BASE_DIR / "data" / "sales_data.csv"

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Load forecast results if available
forecast_path = BASE_DIR / "reports" / "forecast_results.csv"

forecast_df = None

if forecast_path.exists():
    forecast_df = pd.read_csv(forecast_path)
    forecast_df["Date"] = pd.to_datetime(
        forecast_df["Date"]
    )
# Load model evaluation results
evaluation_path = (
    BASE_DIR / "reports" / "model_evaluation.csv"
)

evaluation_df = None

if evaluation_path.exists():

    evaluation_df = pd.read_csv(evaluation_path)

    evaluation_df["Date"] = pd.to_datetime(
        evaluation_df["Date"]
    )
# Dashboard page
if menu == "Dashboard":

    st.header("Sales Overview")

    total_sales = df["Sales"].sum()
    average_sales = df["Sales"].mean()
    highest_sales = df["Sales"].max()
    lowest_sales = df["Sales"].min()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sales",
        f"₹{total_sales:,.0f}"
    )

    col2.metric(
        "Average Sales",
        f"₹{average_sales:,.0f}"
    )

    col3.metric(
        "Highest Sales",
        f"₹{highest_sales:,.0f}"
    )

    col4.metric(
        "Lowest Sales",
        f"₹{lowest_sales:,.0f}"
    )

    st.subheader("Historical Sales Trend")

    fig = px.line(
        df,
        x="Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Performance"
    )

    st.plotly_chart(fig, use_container_width=True)

    if forecast_df is not None:

        st.subheader("Future Sales Forecast")

        forecast_chart = px.line(
            forecast_df,
            x="Date",
            y="Predicted_Sales",
            markers=True,
            title="Next Three Months Sales Prediction"
        )

        st.plotly_chart(
            forecast_chart,
            use_container_width=True
        )

    else:
        st.info(
            "Forecast not available yet. "
            "Train the model and generate predictions first."
        )
# Actual vs Predicted Sales Comparison

    if evaluation_df is not None:

        st.subheader(
            "Actual vs Predicted Sales Comparison"
        )

        comparison_df = evaluation_df.melt(
            id_vars=["Date"],
            value_vars=[
                "Actual_Sales",
                "Predicted_Sales"
            ],
            var_name="Sales_Type",
            value_name="Sales"
        )

        comparison_chart = px.line(
            comparison_df,
            x="Date",
            y="Sales",
            color="Sales_Type",
            markers=True,
            title="Actual vs Predicted Sales"
        )

        st.plotly_chart(
            comparison_chart,
            use_container_width=True
        )

    else:

        st.info(
            "Run model training to view the comparison."
        )
# Sales analysis page
elif menu == "Sales Analysis":

    st.header("Sales Data Analysis")

    st.write("Historical sales records")

    st.dataframe(df, use_container_width=True)

    st.subheader("Monthly Sales Bar Chart")

    bar_chart = px.bar(
        df,
        x="Date",
        y="Sales",
        title="Monthly Sales Comparison"
    )

    st.plotly_chart(
        bar_chart,
        use_container_width=True
    )

    highest_month = df.loc[
        df["Sales"].idxmax()
    ]

    lowest_month = df.loc[
        df["Sales"].idxmin()
    ]

    st.write(
        "Highest Sales Month:",
        highest_month["Date"].strftime("%B %Y")
    )

    st.write(
        "Lowest Sales Month:",
        lowest_month["Date"].strftime("%B %Y")
    )

# Sales prediction page
# Dataset Upload Page
elif menu == "Upload Dataset":

    st.header("Upload Your Sales Dataset")

    st.write(
        "Upload your historical sales data "
        "in CSV format."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            # Check required columns
            required_columns = ["Date", "Sales"]

            if not all(
                col in uploaded_df.columns
                for col in required_columns
            ):

                st.error(
                    "CSV must contain Date and Sales columns."
                )

            else:

                # Convert columns
                uploaded_df["Date"] = pd.to_datetime(
                    uploaded_df["Date"],
                    errors="coerce"
                )

                uploaded_df["Sales"] = pd.to_numeric(
                    uploaded_df["Sales"],
                    errors="coerce"
                )

                # Remove invalid rows
                uploaded_df = uploaded_df.dropna(
                    subset=["Date", "Sales"]
                )

                uploaded_df = uploaded_df.drop_duplicates(
                    subset=["Date"]
                )

                uploaded_df = uploaded_df.sort_values(
                    "Date"
                )

                st.success("Dataset uploaded successfully!")

                st.subheader("Dataset Preview")

                st.dataframe(
                    uploaded_df,
                    use_container_width=True
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Records",
                    len(uploaded_df)
                )

                col2.metric(
                    "Total Sales",
                    f"₹{uploaded_df['Sales'].sum():,.2f}"
                )

                col3.metric(
                    "Average Sales",
                    f"₹{uploaded_df['Sales'].mean():,.2f}"
                )

                st.subheader("Missing Values")

                st.write(
                    uploaded_df.isnull().sum()
                )

                st.download_button(
                    "Download Cleaned Dataset",
                    uploaded_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="cleaned_sales_data.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"Unable to read CSV: {e}")
elif menu == "Sales Prediction":

    st.header("Future Sales Prediction")

    if forecast_df is not None:

        st.success("Forecast loaded successfully!")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        total_forecast = forecast_df[
            "Predicted_Sales"
        ].sum()

        st.metric(
            "Total Predicted Sales",
            f"₹{total_forecast:,.2f}"
        )

        st.download_button(
            label="Download Forecast Report",
            data=forecast_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No forecast found. "
            "Please train the model and run prediction.py."
        )

# About page
elif menu == "About Project":

    st.header("About Our Project")

    st.write("""
    AI-Based Sales Forecasting System is a Machine
    Learning application designed to analyze historical
    sales data and predict future sales.

    Technologies Used:

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest Regression
    - Streamlit
    - Plotly

    The system helps businesses understand sales
    patterns and make better planning decisions.
    """)

    st.success(
        "AI-Based Sales Forecasting System"
    )