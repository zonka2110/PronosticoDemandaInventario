# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:30:44 2021

@author: sebastian
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

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

def runsvm(parte):
    t0 = time.time()
    #%% Dataframe GPI Ventas--------------------------------
    df0 = pd.read_csv('../Data/Predict_B/DataPredictB_%s.csv'%parte, sep=",", decimal=',' )
    
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
    
    # df0['CtdadUMBase_3trim_mean'] = df0['CtdadUMBase_3trim_mean'].map(float)
    # df0['CtdadUMBase_2trim_mean'] = df0['CtdadUMBase_2trim_mean'].map(float)
    # df0['CtdadUMBase_1trim_mean'] = df0['CtdadUMBase_1trim_mean'].map(float)
    # df0['CtdadUMBase_0trim_mean'] = df0['CtdadUMBase_0trim_mean'].map(float)
    
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
    IdMaterialFilter = list(df0['IdMaterial'].unique())
    
    # df0 = df0[df0['Demanda'] != 0]
    # df0 = df0[df0['UMB'] != 'UN']
    
    df1 = df0[df0['CentroSuministrador']=='Osorno']
    
    print(df1.columns.tolist())
    
    df1 = df1.drop(['IdMaterial','MesPeriodo','Periodo',
                    'Unnamed: 0',
                    # 'Unnamed: 0.1',
                    'index', 'Demanda',
                    # 'CtdadUMBase_3trim_mean',
                    # 'CtdadUMBase_2trim_mean',
                    # 'CtdadUMBase_1trim_mean',
                    # 'CtdadUMBase_0trim_mean'
                    ], axis=1)
        

    df1 = df1.drop_duplicates()
    
    
    # df2 = df0[['IdMaterial','UMBase']].drop_duplicates()
    
    df0 = df0.sort_values(by='IdPeriodo')
    
    df0 = df0.groupby(['IdPeriodo','MesPeriodo','IdMaterial','Periodo']).agg({'Demanda':'sum'}).reset_index()
    df0['CentroSuministrador'] = 'Osorno'
    
    df0 = df0.merge(df1, on=['CentroSuministrador','IdPeriodo'], how = 'left')
    
    # df0 = df0.merge(df2, on='IdMaterial', how = 'left')   

    
    df0 = df0[['IdMaterial', 'MesPeriodo', 'IdPeriodo', 'Periodo',
                  'Demanda', 'precip', 'tmin',
                  'tmax', 'tmean', 'dia_lluvia', 'dia_helada', 'dia_caluroso',
                  'Dolar', 'IPC', 'Desempleo', 'IMACEC', 'Precio_carne', 'Precio_leche',
                  'Precio_carne_1trim_mean', 'Precio_carne_0trim_mean',
                  'Precio_leche_1trim_mean', 'Precio_leche_0trim_mean',
                  'tmean_1trim_mean', 'tmean_0trim_mean', 'precip_1trim_mean',
                  'precip_0trim_mean', 'dia_lluvia_1trim_mean', 'dia_lluvia_0trim_mean',
                  'dia_helada_1trim_mean', 'dia_helada_0trim_mean', 'tmin_1trim_mean',
                  'tmin_0trim_mean', 'tmax_1trim_mean', 'tmax_0trim_mean', 
                  'dia_caluroso_1trim_mean', 'dia_caluroso_0trim_mean']]
    
    df0 = df0.dropna()
    
    
    def get_models():
        models = dict()
        preproc = [StandardScaler(),
                   MaxAbsScaler(), 
                   MinMaxScaler(),
                   RobustScaler(quantile_range = (0.1,0.9))]
        for n in range(len(preproc)) :
            for c in range(1,5,1):
                epsilon_ = [0.1, 0.2, 0.3, 0.4]
                for e in epsilon_:
                    gamma_ = ['scale', 'auto']
                    for g in gamma_:
                        shrinking_ = [True, False]
                        for s in shrinking_:
                            kernel_ = ['linear', 'poly', 'rbf', 'sigmoid']
                            for k in kernel_:
                                key = (n, c,e,g,s,k)
                                models[key] = make_pipeline(preproc[n],
                                                            SVR(C         = c,
                                                                epsilon   = e,
                                                                gamma     = g,
                                                                shrinking = s,
                                                                kernel    = k))
        return models
    
    def SVM_Profile(df0):
        
        results = {}
        weather       = ['precip_0trim_mean','precip_1trim_mean',
                              'dia_caluroso_0trim_mean','dia_caluroso_1trim_mean',
                              'dia_lluvia_0trim_mean','dia_lluvia_1trim_mean',
                              'tmean_0trim_mean','tmean_1trim_mean',
                              'dia_helada_0trim_mean','dia_helada_1trim_mean']
        
        macroeconomic = ['Dolar','IPC','Desempleo','IMACEC',
                              'Precio_carne_0trim_mean','Precio_carne_1trim_mean',
                              'Precio_leche_0trim_mean','Precio_leche_1trim_mean',
                              ]
        
        # entreprise    = ['CtdadUMBase_3trim_mean','CtdadUMBase_2trim_mean',
                              # 'CtdadUMBase_1trim_mean','CtdadUMBase_0trim_mean']
        
        temporal      = ['MesPeriodo']
        
        predictors = {
                        # 'all'    : self.temporal + self.weather+ self.macroeconomic,#, + self.entreprise
                        'exogenous': weather+ macroeconomic,
                        # 'weather'     :  weather, #+ self.entreprise
                        # 'macroeconomic': macroeconomic,# + self.entreprise
                        # 'timedate' : []
                        # 'entreprise'   : self.temporal + self.entreprise
                    }
        n_, m_ = 0, 0
        
        df0.MesPeriodo = df0.MesPeriodo.map(str)
        
        df1 = df0.MesPeriodo.str.get_dummies()
        df1.columns = ['is_' + col for col in df1.columns]
        df0 = pd.concat([df0, df1], axis=1)
        temporal = df1.columns.tolist()
            
        for key in predictors.keys():
            predictors[key] += temporal 
            
        t1 = time.time()
        print(df0.columns)
        IdMaterial = df0.IdMaterial.unique().tolist()
        n_, m_ = 0, 0
        for sku in IdMaterial:
            m_ += 1
            print('\n \n \n \n Porcentaje:',round(100*m_/len(IdMaterial),2), round(time.time()-t1,2),'\n \n \n \n')
            # IdCeSum = df0['IdCeSum'].unique()
            # for suc in IdCeSum:
            df = df0[(df0['IdMaterial']==sku)]
            score_min = np.inf
            name_best = ['Insuficiente Data']
            
            # define dataset
            # train = df[:int(len(df)*.8)]
            # test  = df[int(len(df)*.8):]
            if len(df) > 2 and len(df) > 0:
                for keys_ in predictors.keys():
                    
                    predictor = predictors[keys_]
                    X, y = df[predictor], df['Demanda']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
                    # X, y = train[predictor], train['Demanda']
                                    
                    # get the models to evaluate
                    models = get_models()
                    # X2, y2 = test[predictor], test['Demanda']
                    # evaluate the models and store results
    
                    for name, model in models.items():
                        
                        ys_real, ys_pred = [], []
                        model.fit(X_train, y_train)
                        
                        for value in y_test:
                            ys_real.append(value)
                            
                        for value in model.predict(X_test):
                            ys_pred.append(value)
                        
                        # evaluate the model
                        ys_real, ys_pred = np.array(ys_real), np.array(ys_pred)
                        
                        scores = np.mean(np.abs(ys_real-ys_pred)/(1e-3+np.abs(ys_pred)+np.abs(ys_real)))
                        #np.sqrt(np.mean((ys_pred-ys_real)**2))
                        # store the results
                        

                        if scores < score_min:
                            score_min = scores
                            name_best = name
    
                results[n_] = [sku,name_best,score_min]
                print(sku,name_best,score_min)
                n_ += 1
        return results
    
    tuning = SVM_Profile(df0)
    dict_table = pd.DataFrame.from_dict(tuning, orient='index',
                                        columns=['SKU','Params','Score']) 
    dict_table.to_csv('Parameters/Categoria_B/svm/svmtest%s.csv'%parte)
    
    print(time.time() - t0)
    
runsvm(parte=sys.argv[1])