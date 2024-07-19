import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from io import BytesIO

# Title of the app
st.title('Simple Forecast App')

st.write("## Load Data")
# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file for train", type="csv")
uploaded_file2 = st.file_uploader("Choose a CSV file  for test", type="csv")
if (uploaded_file is not None) and (uploaded_file2 is not None):
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    df_test = pd.read_csv(uploaded_file2)
    df_test0 = df_test.copy()
    df0 = df.copy()
    bool_cols = [x for x in df.columns if df[x].dtype==bool]
    df[bool_cols] = df[bool_cols].astype(int)
    st.write("### View Data")
    # Input widget to specify the number of rows to display
    num_rows = st.text_input("Enter the number of rows to display:", "10",key="number")

    # Selectbox to choose the column to sort by
    sort_column = st.selectbox("Select column to sort by:", df.columns)

    # Buttons to sort the DataFrame
    sort_asc = st.button("Sort Ascending")
    sort_desc = st.button("Sort Descending")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Train Data")
        # Sorting logic
        if sort_asc:
            df = df.sort_values(by=sort_column, ascending=True)
        elif sort_desc:
            df = df.sort_values(by=sort_column, ascending=False)

        try:
            num_rows = int(num_rows)
            if num_rows > len(df):
                num_rows = len(df)
            # Display the specified number of rows of the DataFrame
            st.write(f"Displaying the first {num_rows} rows of the CSV file:")
            st.dataframe(df.head(num_rows))
        except ValueError:
            st.write("Please enter a valid number.")
    with col2:
        st.write("#### Test Data")

        # Sorting logic
        if sort_asc:
            df_test0 = df_test0.sort_values(by=sort_column, ascending=True)
        elif sort_desc:
            df_test0 = df_test0.sort_values(by=sort_column, ascending=False)

        try:
            num_rows = int(num_rows)
            if num_rows > len(df_test0):
                num_rows = len(df_test0)
            # Display the specified number of rows of the DataFrame
            st.write(f"Displaying the first {num_rows} rows of the CSV file:")
            st.dataframe(df_test0.head(num_rows))
        except ValueError:
            st.write("Please enter a valid number.")

    # Scatter plot section
    st.write("### Train Line Plot")
    X = st.selectbox("Select X axis:",df.columns)
    Y = st.selectbox("Select Y axis:",df.columns)
    num_rows2 = st.text_input("Enter the number of rows to display:", "250",key="number2")
    st.line_chart(data=df.iloc[:int(num_rows2)],x=X,y=Y)
    st.write("## Setup Training Data")
    st.write("### Select id and timestamp")

    try:
        id_column = st.selectbox("Select id column:", df.columns, index= "item_id")
        
    except:
        id_column = st.selectbox("Select id column:", df.columns)
    try:
        timestamp_column = st.selectbox("Select timestamp column:", df.columns, index= "timestamp")
        
    except:
        timestamp_column = st.selectbox("Select timestamp column:", df.columns)

    train_data = TimeSeriesDataFrame.from_data_frame(
    df0,
    id_column=id_column,
    timestamp_column=timestamp_column
)   
    st.write("### Created Train data")
    num_rows3 = st.text_input("Enter the number of rows to display:", "3",key="number3")
    st.dataframe(train_data.head(int(num_rows3)))

    st.write("## AutoGluon Parameters")
    st.write("### Set Parameters:")
    prediction_length = st.number_input("Prediction Length:", min_value=1, value=48)
    try:
        target_column = st.selectbox("Select target column:", df.columns,index="target")
    except:
        target_column = st.selectbox("Select target column:", df.columns)
    try:
        presets = st.selectbox("Select presets:", ["medium_quality", "high_quality", "best_quality"],index="medium_quality")
    except:
        presets = st.selectbox("Select presets:", ["medium_quality", "high_quality", "best_quality"])

    time_limit = st.number_input("Time Limit (seconds):", min_value=1, value=10)

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path="autogluon-m4-hourly",
        target=target_column,
        eval_metric="MASE",)
    with st.spinner("Fitting the model..."):
        predictor.fit(
        train_data,
        presets=presets,
        time_limit=time_limit,
         )
    st.write("### AutoGluon Results:")
    st.write("AutoGluon LB:")
    predictions = predictor.predict(train_data)
    LB = predictor.leaderboard(df_test)
    st.dataframe(LB)

    # Plot chart with Streamlit
    st.write("AutoGluon Chart:")
    fig = predictor.plot(df_test, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)
    st.pyplot(fig)

    st.write("### Predictions Data:")
    # Convert the pandas DataFrame to a CSV
    @st.cache_data
    def convert_df(_df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return _df.to_csv().encode("utf-8")

    csv = convert_df(predictions)
    try:
        num_rows4 = st.text_input("Enter the number of rows to display:", "25",key="number4")
        st.dataframe(predictions.head(int(num_rows4)))
    except:pass
    st.write("### Download predictions data as CSV:")
    st.download_button(
        label="Download",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
    st.write("### Download predictions chart as PNG:")
    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Add a download button
    st.download_button(
        label="Download plot as PNG",
        data=buf,
        file_name="plot.png",
        mime="image/png"
    )

    
else:
    st.write("Please upload a CSV file to see its contents.")






