#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:25:46 2021

@author: sdavila
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
# from kats.models.prophet import ProphetModel, ProphetParams
from kats.models.sarima import SARIMAModel, SARIMAParams

from lightgbm import LGBMRegressor
import os
import ast

import Parameters  # for path finding
p = str(Parameters.__path__)
l = p.find("'")
r = p.find("'", l+1)
data_path = p[l+1:r]
print("Parameters",data_path)

path = os.path.join(data_path,'svm13v2.csv')
psvm = pd.read_csv(path,  sep=",", decimal=',' )

path = os.path.join(data_path,'lgbm3.csv')
plgbm = pd.read_csv(path,  sep=",", decimal=',' )

path = os.path.join(data_path,'RandomForestv3.csv')
prf = pd.read_csv(path,  sep=",", decimal=',' )

path = os.path.join(data_path,'Sarima.csv')
sarima = pd.read_csv(path,  sep=",", decimal=',' )

#%% Dataframe GPI Ventas--------------------------------

df0 = pd.read_csv('../Data/DataPredictd10.csv', sep=",", decimal=',' )
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
df0.rename(columns={"CtdadUMBase": "Demanda"}, inplace=True)

df0['Demanda'] = df0['Demanda'].map(float)

df0 = df0.dropna()

IdCeSumes = list(df0['IdCeSum'].unique())
IdMaterial = list(df0['IdMaterial'].unique())
IdMaterial = IdMaterial[round(0.75*len(IdMaterial)):]

df0['Demanda'] = df0['Demanda'].map(float)
df0['MesPeriodo'] = df0['MesPeriodo'].map(int) 
df0['Dolar'] = df0['Dolar'].map(float)

df0['Precio_carne_1trim_mean'] = df0['Precio_carne_1trim_mean'].map(float)
df0['Precio_carne_0trim_mean'] = df0['Precio_carne_0trim_mean'].map(float)

df0['Precio_leche_1trim_mean'] = df0['Precio_leche_1trim_mean'].map(float)
df0['Precio_leche_0trim_mean'] = df0['Precio_leche_0trim_mean'].map(float)

df0['CtdadUMBase_3trim_mean'] = df0['CtdadUMBase_1trim_mean'].map(float)
df0['CtdadUMBase_2trim_mean'] = df0['CtdadUMBase_0trim_mean'].map(float)
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

#%% ------------------------------------------------------------------

df0 = df0.sort_values(by='IdPeriodo')

def get_svm(df0, params):
    model = make_pipeline(StandardScaler(), SVR(C         = 1.0,
                                                epsilon   = (None
                                                             if params[0] == 'None' 
                                                             else float(params[0])),
                                                gamma     = params[1],
                                                shrinking = (params[2] == 'True'),
                                                kernel    = params[3]))
    return model

def get_lgbm(df0, params):
    model = LGBMRegressor(
        verbose = -1,
        # device = 'gpu',
        n_estimators            = int(params[0]), # equivalent num_iterations
        max_depth               = 8,
        num_leaves              = 20,
        learning_rate           = float(params[2]),
        boosting_type           = params[3],
        n_jobs                  = 8,
        # min_gain_to_split       = 0.5,
        # tree_learner            ='feature',
        # max_bin                 = 255
        min_data_in_leaf        = 50,
        min_sum_hessian_in_leaf = 100,
        # lambda_l1               = 1e-3,
        # lambda_l2               = 1e-3,
        # min_gain_to_split       = 1e-3,
        # path_smooth                 = 1e-3
        # bagging_freq = 0 if types == 'goss' else 5,
        # bagging_fraction = 1.0 if types == 'goss' else 0.75
        )
    return model

def get_rforest(df0, params):
        
    try:                                    
        model = RandomForestRegressor(max_samples       =(None
                                                      if params[0] == 'None' 
                                                      else float(params[0])),
                                                        max_features      = 'auto' if params[1] == 'auto' 
                                                                            else int(params[1]),
                                                        n_estimators      = int(params[2]),
                                                        max_depth         = (None
                                                                             if params[3] == 'None' 
                                                                             else float(params[3])),
                                                        min_samples_split = int(params[5]),
                                                        min_samples_leaf  = int(params[6]),
                                                        bootstrap         = True if params[4] == 'True' 
                                                                            else False,
                                                        n_jobs            = 8,
                                                        random_state      = 123)
    except:
        params = ['None', 'auto',100, 'None', 'True', 2, 1]
        model = RandomForestRegressor(max_samples       =(None
                                                      if params[0] == 'None' 
                                                      else float(params[0])),
                                                        max_features      = 'auto' if params[1] == 'auto' 
                                                                            else int(params[1]),
                                                        n_estimators      = int(params[2]),
                                                        max_depth         = (None
                                                                             if params[3] == 'None' 
                                                                             else float(params[3])),
                                                        min_samples_split = int(params[5]),
                                                        min_samples_leaf  = int(params[6]),
                                                        bootstrap         = True if params[4] == 'True' 
                                                                            else False,
                                                        n_jobs            = 8,
                                                        random_state      = 123)
    
    return model

def get_sarima(df0, params):
    # so = params['seasonal_order'].replace("'","")
    # params['seasonal_order'] = tuple(int(i) for i in so if i != "," and i != '(' and i != ')' and i != ' ')
    model = SARIMAParams(
                           p              = int(params[0]),
                           d              = int(params[1]),
                           q              = int(params[2]),
                           # seasonal_order = params['seasonal_order'],
                           # tren           = params['trend']
                                  )
    return model

def get_models(df0, params):
    models = {'lgbm'    : get_lgbm(df0, params['lgbm']),
              # 'svm'     : get_svm(df0, params['svm']),
              'rforest' : get_rforest(df0, params['rforest']),
              # 'sarima'  : get_sarima(df0, params['sarima'])
              }
    return models

def Profile(df0, psvm, plgbm, prf):
    
    results = {}
    weather       = ['precip_0trim_mean','precip_1trim_mean',
                     'dia_caluroso_0trim_mean','dia_caluroso_1trim_mean',
                     'dia_lluvia_0trim_mean','dia_lluvia_1trim_mean',
                     'tmean_0trim_mean','tmean_1trim_mean',
                     'dia_helada_0trim_mean','dia_helada_1trim_mean']
    
    macroeconomic = ['Dolar','IPC','Desempleo','IMACEC',
                     'Precio_carne_0trim_mean','Precio_carne_1trim_mean',
                     'Precio_leche_0trim_mean','Precio_leche_1trim_mean',]
    
    entreprise    = ['CtdadUMBase_3trim_mean','CtdadUMBase_2trim_mean',
                     'CtdadUMBase_1trim_mean','CtdadUMBase_0trim_mean']
    
    temporal      = ['MesPeriodo']
    
    predictors    = {#'weather'     : temporal + entreprise + weather,
                     #'macroeconomic': temporal + entreprise + macroeconomic,
                      'all'          : temporal + entreprise + weather+ macroeconomic
                     }
    n_, m_, p_ = 0, 0, 0
    for sku in IdMaterial:
        m_ += 1
        print('==============================================================')
        print('\n\n\n',sku,'Porcentaje:%s'%(100*m_/len(IdMaterial)),'\n','\n','\n')
        print('==============================================================')
        IdCeSum = df0['IdCeSum'].unique()
        params_ = {}
        for suc in IdCeSum:
            df = df0[(df0['IdMaterial']==sku) & (df0['IdCeSum']==suc)]
            
            # param = psvm[(psvm['SKU'] == sku) & (psvm['IdCeSum'] == suc)]['Params'].unique()
            # print(sku,suc,param)
            # if param.size > 0 and type(param[0]) == str:
            #     param = param[0].replace('(','').replace(')','').replace("'",'')
            #     params_['svm'] = [x.strip() for x in param.split(',')]
            #     if params_['svm'][0] == '[Insuficiente Data]':
            #         params_['svm'] = [0.1, 'scale', 'True', 'rbf']
            # else:
            #     params_['svm'] = [0.1, 'scale', 'True', 'rbf']
            
            # param = plgbm[(plgbm['SKU'] == sku) & (plgbm['IdCeSum'] == suc)]['Params'].unique()
            # if param.size > 0:
            #     param = param[0].replace('(','').replace(')','').replace("'",'')
            #     params_['lgbm'] = [x.strip() for x in param.split(',')]
            #     if params_['lgbm'][0] == '[Insuficiente Data]':
            #         params_['lgbm'] = [100,  31, 0.1, 'gbdt']
            # else:
            params_['lgbm'] = [5000,-1, 0.001, 'gbdt']
            
            param = prf[(prf['SKU'] == sku)]['Params'].unique()
            if param.size > 0:
                param = param[0].replace('(','').replace(')','').replace("'",'')
                params_['rforest'] = [x.strip() for x in param.split(',')]
                if params_['rforest'][0] == '[Insuficiente Data]':
                    params_['rforest'] = ['None', 'auto', 100, 'None',2,1]
            else:
                params_['rforest'] = ['None', 'auto',100, 'None', 'True', 2, 1]

            # param = sarima[(sarima['SKU'] == sku) & (sarima['IdCeSum'] == suc)]['Params'].unique()
            # print(param,sku,suc)
            # params_['sarima'] = [3,2,1]#ast.literal_eval(param[0])

            score_min = np.inf
            name_best = 'Falta Información'
            param_best = []

            # define dataset
            train = df[:int(len(df)*.8)]
            test  = df[int(len(df)*.8):]
            if len(train) >= 2 and len(test) > 0:
                models = get_models(df0, params_)
               
                for name, model in models.items():
                    p_ += 1

                    try:
                        predictor = 'all'
                        predictor_  = predictors[predictor].copy()
                        
                        if name == 'svm' or name == 'rforest':
                            train_ = train
                            test_ = test
                            
                            df1 = train_.IdCeSum.str.get_dummies()
                            df1.columns = ['is_' + col for col in df1.columns]
                            train_ = pd.concat([train_, df1], axis=1)
                            branches = df1.columns.tolist()    
                    
                            df1 = test_.IdCeSum.str.get_dummies()
                            df1.columns = ['is_' + col for col in df1.columns]
                            test_ = pd.concat([test_, df1], axis=1)
                            branches = df1.columns.tolist()
                                        
                            predictor_ += branches
                        else:
                            train_ = train
                            test_ = test
                                
                        
                        X, y = train_[predictor_], train_['Demanda']
                        # get the models to evaluate
    
                        X2, y2 = test_[predictor_], test_['Demanda']
                        # # evaluate the models and store results
                        # fig = plt.figure()
                        # ax = fig.add_subplot(1, 1, 1)
                        # ax.set_xlabel('Date')
                        # ax.set_ylabel('Demand')
                    
                    # for name, model in models.items():
                        # print(name,params_[name], predictor_)
                        
                        ys_real, ys_pred = [], []
                        
                        for value in y2:
                            ys_real.append(value)
                                                                    
                        if name == 'svm' or name == 'lgbm' or name == 'rforest':
                                                        
                            data_predict = X2
                            model.fit(X, y)
                            model_pred = model.predict(data_predict)                       
                            
                        elif name == 'sarima':
                            train_ts = train[['IdPeriodo','Demanda']]
                            train_ts = train_ts.rename(columns={"IdPeriodo": "time", "Demanda": "value"})
                            train_ts = TimeSeriesData(train_ts)
                            
                            data_predict = len(ys_real)
                            model = SARIMAModel(train_ts, model)
                            model.fit()
                            model_pred = model.predict(data_predict)['fcst']
                        
                        for value in model_pred:
                            ys_pred.append(value)
    
                        # evaluate the model
                        ys_real, ys_pred = np.array(ys_real), np.array(ys_pred)
                        # print(ys_real, ys_pred, name)
                        scores = np.mean(np.abs(ys_real-ys_pred))
                            
                             # store the results
                        if scores < score_min:
                            score_min = scores
                            name_best = name
                            param_best = params_[name]
                            
                            # ax.plot([i for i in range(len(ys_pred))], ys_pred, label=name)
                        
                            # ax.plot([i for i in range(len(ys_real))], ys_real, '--', label = 'real')
                    except:
                        print('Error')
                        print(name,params_[name])
                # plt.show()
            
            results[n_] = [sku,suc,name_best,param_best]
            
            # print(results[n_])
            
            n_ += 1
    return results

tuning = Profile(df0, psvm, plgbm, prf)
dict_table = pd.DataFrame.from_dict(tuning, orient='index',
                                    columns=['SKU','IdCeSum','NameAlg','Params']) 
dict_table.to_csv('Profile/ModelParam1.csv')

df_aux = df0.groupby(['IdMaterial','IdCeSum']).agg({'IdPeriodo':'count'}).reset_index()

df_aux = df_aux[df_aux['IdPeriodo']>=8]