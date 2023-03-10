# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:13:03 2023

@author: Vicente
"""
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing,Holt,ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load Data
df=pd.read_csv('Data/Predict/DataPredict_3114.csv',parse_dates=['IdPeriodo'])
df['Y_month']=df['IdPeriodo'].apply(lambda x:pd.Timestamp(x).strftime('%Y-%m'))
df=df.groupby(by=['Y_month']).sum().reset_index()

# Choose only necessary columns
df = df[['Y_month', 'CtdadUMBase']]

# Normalize Metric
df['CtdadUMBase'] = \
    df['CtdadUMBase'] \
        / pd.to_datetime(df['Y_month']).dt.day
        
def set_date_index(input_df, col_name='Y_month'):
    """Given a pandas df, parse and set date column to index.
        col_name will be removed and set as datetime index.

    Args:
        input_df (pandas dataframe): Original pandas dataframe
        col_name (string): Name of date column

    Returns:
        pandas dataframe: modified and sorted dataframe
    """
    # Copy df to prevent changing original
    modified_df = input_df.copy()

    # Infer datetime from col
    modified_df[col_name] = pd.to_datetime(modified_df[col_name])

    # Sort and set index
    modified_df.sort_values(col_name, inplace=True)
    modified_df.set_index(col_name, inplace=True)

    return modified_df

# Set date index
df = set_date_index(df, 'Y_month') # custom helper function


def combine_seasonal_cols(input_df, seasonal_model_results):
    """Adds inplace new seasonal cols to df given seasonal results

    Args:
        input_df (pandas dataframe)
        seasonal_model_results (statsmodels DecomposeResult object)
    """
    # Add results to original df
    input_df['observed'] = seasonal_model_results.observed
    input_df['residual'] = seasonal_model_results.resid
    input_df['seasonal'] = seasonal_model_results.seasonal
    input_df['trend'] = seasonal_model_results.trend
    
    plt.figure(figsize=(16,8))
    plt.plot(input_df['observed'],'--',color='green',label='Data')
    plt.plot(input_df['seasonal'],'--',color='red',label='Estacionalidad')
    plt.legend(loc='best')
    plt.show()
    plt.plot(input_df['residual'],'--',color='blue',label='Residuo')
    plt.legend(loc='best')
    plt.show()
    plt.plot(input_df['trend'],'--',color='black',label='Tendencia')
    plt.legend(loc='best')
    plt.show()
# Seasonal decompose 
sd = seasonal_decompose(df, period=12)
combine_seasonal_cols(df, sd) # custom helper function


float_format='%g'
df.to_csv('../GPI/Results.csv', float_format='%.3f')
print(df)


    
    