import streamlit as st
import pandas as pd
from src.helper_function import summary_statistic, missing_plot, pie_plot

st.header('2. Analysis')
DATA_DIR = '/Users/andishetavakkoli/Documents/notebook/github_project/machine-learning-projects-data/anomaly_detection/'
df = pd.read_csv(DATA_DIR +'creditcard.csv')
st.markdown('### Credit Card Data')
st.dataframe(df.head())

st.markdown('#### Summary Statistic')
st.dataframe(summary_statistic(df))

st.markdown('#### Missing Values')

# Display the plot using Streamlit
st.pyplot(missing_plot(df))

# Analysis Target
st.markdown('#### Target Column Analysis')

st.plotly_chart(pie_plot(df, 'Class'))

st.write('Mean values according to the "class" column')
st.dataframe(df.groupby('Class').mean().style.background_gradient())




