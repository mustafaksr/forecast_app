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

# Title of the app
st.title('CSV File Loader')
# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file for train", type="csv")
uploaded_file2 = st.file_uploader("Choose a CSV file  for test", type="csv")
if (uploaded_file is not None) and (uploaded_file2 is not None):
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    df_test = pd.read_csv(uploaded_file2)
    df0 = df.copy()
    bool_cols = [x for x in df.columns if df[x].dtype==bool]
    df[bool_cols] = df[bool_cols].astype(int)

    # Input widget to specify the number of rows to display
    num_rows = st.text_input("Enter the number of rows to display:", "10",key="number")

    # Selectbox to choose the column to sort by
    sort_column = st.selectbox("Select column to sort by:", df.columns)

    # Buttons to sort the DataFrame
    sort_asc = st.button("Sort Ascending")
    sort_desc = st.button("Sort Descending")

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

    # Scatter plot section
    st.write("## Line Plot")
    X = st.selectbox("Select X axis:",df.columns)
    Y = st.selectbox("Select Y axis:",df.columns)
    num_rows2 = st.text_input("Enter the number of rows to display:", "10",key="number2")
    st.line_chart(data=df.iloc[:int(num_rows2)],x=X,y=Y)


    st.write("## Select id and timestamp")
    id_column = st.selectbox("Select id column:", df.columns)
    timestamp_column = st.selectbox("Select timestamp column:", df.columns)

    train_data = TimeSeriesDataFrame.from_data_frame(
    df0,
    id_column=id_column,
    timestamp_column=timestamp_column
)   
    st.write("### Created Train data")
    num_rows3 = st.text_input("Enter the number of rows to display:", "3",key="number3")
    st.dataframe(train_data.head(int(num_rows3)))

    predictor = TimeSeriesPredictor(
        prediction_length=48,
        path="autogluon-m4-hourly",
        target="target",
        eval_metric="MASE",
    )

    predictor.fit(
        train_data,
        presets="medium_quality",
        time_limit=10,
    )
    st.write("AutoGluon LB:")
    predictions = predictor.predict(train_data)
    LB = predictor.leaderboard(df_test)
    st.dataframe(LB)

    # Plot chart with Streamlit
    st.write("AutoGluon Chart:")
    predictor.plot(df_test, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)

    try:
        num_rows4 = st.text_input("Enter the number of rows to display:", "25",key="number4")
        st.dataframe(predictions.head(int(num_rows4)))
    except:pass
    
    

else:
    st.write("Please upload a CSV file to see its contents.")






