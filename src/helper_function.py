
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px

def summary_statistic(df) -> pd.DataFrame:
    """
    Calculate summary statistics for the given pandas DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame, assumed to be in CSV format.

    Returns:
        pandas.DataFrame: DataFrame containing information about the columns.

    """
    feature_describe = df.describe().T.reset_index().rename(
                           columns={'index':'feature'}).drop(columns='count')

    feature_info = pd.concat([df.dtypes,
                               df.nunique(),
                               df.isna().sum(),
                               df.count()], axis=1,
                keys=['type', 'count_unique', 'count_nan', 'count']).reset_index().rename(columns={'index':'feature'})

    summary_statistic_result = feature_info.merge(feature_describe, how='left', on='feature')

    return summary_statistic_result.style.background_gradient()


def missing_plot(df):

    # Generate the missingno plot
    fig, ax = plt.subplots()
    msno.bar(df, ax=ax, figsize=(14,6), fontsize=5, color="seagreen", sort="descending")

    return fig

def pie_plot(df, column):
    fig = px.pie(df, values = df[column].value_counts().values,
                names = (df[column].value_counts()).index,
                title = f'{column} Column Distribution')
    return fig