# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:46:04 2023

@author: Vicente
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from openpyxl import load_workbook

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams
from kats.models.sarima import SARIMAModel, SARIMAParams

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import RandomizedSearchCV
import time


import sys 

print(sys.argv)


def runHoltWinter(parte):
    t0 = time.time()
    #%% Dataframe GPI Ventas--------------------------------
    df0 = pd.read_csv('../Data/DataPredict_%s.csv'%parte, sep=",", decimal=',' )
    
    # Transformar datos a númerico
    df0['IdPeriodo'] = df0['IdPeriodo']
    df0["IdPeriodo"] = pd.to_datetime(df0['IdPeriodo'], format="%Y/%m")
    
    end_date = df0['IdPeriodo'].max()
    start_date = end_date - datetime.timedelta(days=365)
    
    # # Filtro el un año
    df2 = df0[(df0['IdPeriodo'] > start_date) & (df0['IdPeriodo'] <= end_date)]
    
    list_idMaterial = list(df2['IdMaterial'].unique())
    boolean_series = df0.IdMaterial.isin(list_idMaterial)
    df0 = df0[boolean_series]
    
    df0['CtdadUMBase'] = pd.to_numeric(df0['CtdadUMBase'])
    # df0['MntNeto'] = pd.to_numeric(df0['MntNeto'])
    # df0["MntCosto"] = pd.to_numeric(df0["MntCosto"], downcast="float")
    # df0['PrecioUnitario'] = pd.to_numeric(df0["PrecioUnitario"], downcast="float")
    # pd.to_numeric(df0["TransfUMB"], downcast="float")
    
    # df0["TransfUMB"] = pd.to_numeric(df0["TransfUMB"], downcast="float")
    # df0['Prc_IdCeSum'] = pd.to_numeric(df0["Prc_IdCeSum"], downcast="float")
    
    df0.rename(columns={"CtdadUMBase": "Demanda"}, inplace=True)
    
    df0['Demanda'] = df0['Demanda'].map(float)
    # df1 = df0.groupby(['IdPeriodo','IdMaterial','IdCeSum']).agg({'Demanda':'sum',
                                                                              # 'PrecioUnitario':'sum'}).reset_index()
    
    # df0 = df0.dropna()
    
    IdCeSumes = list(df0['IdCeSum'].unique())
    
    
    df0 = df0[(df0.IdMaterial == 10000000)]

    
    IdMaterial = list(df0['IdMaterial'].unique())
    
    df0['Demanda'] = df0['Demanda'].map(float)
    df0['MesPeriodo'] = df0['MesPeriodo'].map(int) 
    df0['Dolar'] = df0['Dolar'].map(float)
    
    # df0['Precio_carne_3trim_mean'] = df0['Precio_carne_3trim_mean'].map(float)
    # df0['Precio_carne_2trim_mean'] = df0['Precio_carne_2trim_mean'].map(float)
    df0['Precio_carne_1trim_mean'] = df0['Precio_carne_1trim_mean'].map(float)
    df0['Precio_carne_0trim_mean'] = df0['Precio_carne_0trim_mean'].map(float)
    
    # df0['Precio_leche_3trim_mean'] = df0['Precio_leche_3trim_mean'].map(float)
    # df0['Precio_leche_2trim_mean'] = df0['Precio_leche_2trim_mean'].map(float)
    df0['Precio_leche_1trim_mean'] = df0['Precio_leche_1trim_mean'].map(float)
    df0['Precio_leche_0trim_mean'] = df0['Precio_leche_0trim_mean'].map(float)
    
    df0['CtdadUMBase_3trim_mean'] = df0['CtdadUMBase_3trim_mean'].map(float)
    df0['CtdadUMBase_2trim_mean'] = df0['CtdadUMBase_2trim_mean'].map(float)
    df0['CtdadUMBase_1trim_mean'] = df0['CtdadUMBase_1trim_mean'].map(float)
    df0['CtdadUMBase_0trim_mean'] = df0['CtdadUMBase_0trim_mean'].map(float)
    
    df0['precip_0trim_mean'] = df0['precip_0trim_mean'].map(float)
    df0['precip_1trim_mean'] = df0['precip_1trim_mean'].map(float)
    df0['dia_caluroso_0trim_mean'] = df0['dia_caluroso_0trim_mean'].map(float)
    df0['dia_caluroso_1trim_mean'] = df0['dia_caluroso_1trim_mean'].map(float)
    df0['dia_lluvia_0trim_mean'] = df0['dia_lluvia_0trim_mean'].map(float)
    df0['dia_lluvia_1trim_mean'] = df0['dia_lluvia_1trim_mean'].map(float)
    df0['tmean_0trim_mean'] = df0['tmean_0trim_mean'].map(float)
    df0['tmean_1trim_mean'] = df0['tmean_1trim_mean'].map(float)
    df0['dia_helada_0trim_mean'] = df0['dia_helada_0trim_mean'].map(float)
    df0['dia_helada_1trim_mean'] = df0['dia_helada_1trim_mean'].map(float)
    
    df0['Dolar']     = df0['Dolar'].map(float)
    df0['IPC']       = df0['IPC'].map(float)
    df0['Desempleo'] = df0['Desempleo'].map(float)
    df0['IMACEC']    = df0['IMACEC'].map(float)
    
    
    
    # df0['PrecioUnitario'] = df0['PrecioUnitario'].map(float)
    
    #%% ------------------------------------------------------------------


    IdMaterialFilter = list(df0['IdMaterial'].unique())
    
    # df0 = df0[df0['Demanda'] != 0]
    # df0 = df0[df0['UMB'] != 'UN']
    
    # df1 = df0[df0['CentroSuministrador']=='Osorno']
    
    # df1 = df1.drop(['IdPeriodo','MesPeriodo','IdMaterial','Periodo',
    #                 'Unnamed: 0','Demanda'], axis=1)
    
    # print(df1.columns.tolist())
    
    # df1 = df1.drop_duplicates()
    
    df2 = df0[['IdMaterial','UMBase']].drop_duplicates()
    
    df0 = df0.sort_values(by='IdPeriodo')
    
    df0 = df0.groupby(['IdPeriodo','MesPeriodo','IdMaterial','Periodo']).agg({'Demanda':'sum'}).reset_index()
    df0['CentroSuministrador'] = 'Osorno'
    
    # df0 = df0.merge(df1, on='CentroSuministrador', how = 'left')
    df0 = df0.merge(df2, on='IdMaterial', how = 'left')
    
    df0 = df0.dropna()
        
    def get_models(ts):
        models = 
        return models
    
    def HoltWinter_Profile(df0):
    
        results = {}
        
        n_, m_ = 0, 0
    
        t1 = time.time()
        n_ = 0
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        m_ = 0
        t1 = time.time()
        for sku in IdMaterial:
            m_ += 1
            print('\n \n \n \n Porcentaje:',round(100*m_/len(IdMaterial),2), round(time.time()-t1,2),'\n \n \n \n')
            # if m_ == 10:
            #     break
        
                
            score_min = np.inf
            name_best = ['Sin Informacion']
            
            df = df0[(df0['IdMaterial']==sku)][['IdPeriodo','Demanda']]
            
            df = df.rename(columns={"IdPeriodo": "time", "Demanda": "value"})
            
            if len(df) > 2:
                split = int(0.8*len(df))
                ts = TimeSeriesData(df)
                
                train_ts = ts[0:split]
                test_ts = ts[split:]
                                
                # get the models to evaluate
                models = get_models(train_ts)
                
                for name, model in models.items():
                    
                    try:
                    
                        ys_real, ys_pred = [], []
                        model.fit(maxiter=50, low_memory = True, full_output = False)
                                                    
                        ys_real = test_ts.value.values
                      
                        model_pred = model.predict(steps=len(test_ts))
                        
                        ys_pred = model_pred['fcst'].values
                        
                        # evaluate the model
                        ys_real, ys_pred = np.array(ys_real), np.array(ys_pred)
                        scores = np.sqrt(np.mean((ys_pred-ys_real)**2)/len(ys_real))
                        
                    except:
                        
                        scores = np.inf
                    # store the results
                    if scores < score_min:
                        score_min = scores
                        name_best = name
                        print(sku,name_best,score_min)
            # except:
                # continue
            print(sku, name_best,score_min)           
            results[n_] = [sku,name_best,score_min]
            n_ += 1
        return results
    
    tuning = HoltWinter_Profile(df0)
    dict_table = pd.DataFrame.from_dict(tuning, orient='index',
                                        columns=['SKU','Params','Score']) 
    dict_table.to_csv('Parameters/HoltWinter%s.csv'%parte)
    
    print(time.time() - t0)