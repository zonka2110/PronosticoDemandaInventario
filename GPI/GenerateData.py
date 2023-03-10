#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:43:39 2021
Genera la data que se utiliza para hacer el pronóstico
Consideremos los SKU que han tenido venta durante el último año

@author: sdavila
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
from openpyxl import load_workbook
import pandas as pd      
import datetime
import os
import time

from dateutil.relativedelta import relativedelta
from ConnectionSQL import*

import psycopg2


# Recuperar camino de la carpeta Data
path = os.getcwd()
data_path = 'Data'



# import Data 
# p = str(Data.__path__)
# l = p.find("'")
# r = p.find("'", l+1)
# data_path = p[l+1:r]
# print("Data",data_path)

class Cooprinsem:
    def __init__(self):
        self.df = ReadSQL('coo_consumos')
        self.path_varexog = 'Data/PredictVarsExogenasv4.csv'
        self.t0 = time.time()
            
    def getData(self):
        
        # Obtiene data historica
        print('Se ha cargado la Data historica con exito')
        
        # self.df = pd.read_csv(self.path,  sep=",", decimal=',' )
        self.df = self.df.dropna()
 
        self.df['IdPeriodo']      = self.df['Periodo'].map(str) + '-' + self.df['MesPeriodo'].map(str)
        self.df["IdPeriodo"]      = pd.to_datetime(self.df['IdPeriodo'], format="%Y/%m")       
        
        # Recupera la fecha del último año
        self.date_current  = self.df['IdPeriodo'].max()
        self.date_lastyear = self.date_current - relativedelta(months=12)
        
        # Filtra los SKU del último año
        IdMaterial = self.df[(self.df['IdPeriodo'] <= self.date_current) & 
                    (self.df['IdPeriodo'] >= self.date_lastyear)]['IdMaterial'].unique()
        
        self.df = self.df[self.df['IdMaterial'].isin(IdMaterial)]
                    
        self.df = self.df.sort_values(by='IdPeriodo')

        # print('Mes getData',self.df['MesPeriodo'].unique())
        print('Fecha de la data utilizada',self.df.IdPeriodo.max())
                    
    def updateFormat(self):
        # Actualiza el formato de las columnas utilizadas.
        self.df['CtdadUMBase']       = pd.to_numeric(self.df['CtdadUMBase'])
        # self.df['MntNeto']        = pd.to_numeric(self.df['MntNeto'])
        self.df["MntCosto"]       = pd.to_numeric(self.df["MntCosto"], downcast="float")
        self.df['IdMaterial']     = pd.to_numeric(self.df['IdMaterial'], downcast = "integer")
        
        self.dfumb = self.df[['IdMaterial','UMBase']]
        self.dfumb = self.dfumb.drop_duplicates()
             
        self.IdMaterial = self.df.IdMaterial.unique()
        
        DateMin                 = self.df.groupby(['IdMaterial',
                                                   'IdCeSum']).agg({'IdPeriodo' : 'min'}).reset_index()

        DateMin = DateMin.rename(columns={'IdPeriodo':'MinPeriodo'})
        
        self.df2 = self.df.groupby(['IdMaterial','IdCeSum',
                                    'Periodo','MesPeriodo',
                                    'CentroSuministrador',
                                    'IdPeriodo']).agg({'CtdadUMBase'       : 'sum',
                                                           # 'MntNeto'        : 'sum',
                                                           'MntCosto'       : 'sum'}).reset_index()
                                                                                                                        
        dfaddnull = pd.pivot_table(self.df2,
                                   values    = 'CtdadUMBase',
                                   columns   = 'IdMaterial',
                                   index     = ['IdCeSum','CentroSuministrador',
                                                'Periodo','MesPeriodo'],
                                   fill_value = 0).reset_index()

        dfaddnull = dfaddnull.melt(id_vars   = ['IdCeSum','CentroSuministrador',
                                                'Periodo','MesPeriodo'], 
                                  var_name   = ['IdMaterial'],
                                  value_name = 'CtdadUMBase')


        dfaddnull = dfaddnull.merge(DateMin, on=['IdMaterial',
                                                  'IdCeSum'], 
                                    )

        dfaddnull['IdPeriodo'] = (dfaddnull['Periodo'].map(str) + '-' +
                                  dfaddnull['MesPeriodo'].map(str))
        dfaddnull["IdPeriodo"] = pd.to_datetime(dfaddnull['IdPeriodo'], format="%Y/%m")
          
        dfaddnull['MinPeriodo'] = (pd.DatetimeIndex(dfaddnull['MinPeriodo']).year.map(str) +
                                   '-'+pd.DatetimeIndex(dfaddnull['MinPeriodo']).month.map(str))

        dfaddnull["MinPeriodo"] = pd.to_datetime(dfaddnull['MinPeriodo'], format="%Y/%m")

        self.df = dfaddnull[(dfaddnull['IdPeriodo'] >=dfaddnull['MinPeriodo']) ]
        
        
        print('Termino updateFormat',self.df.IdPeriodo.max())

    # Si es necesario se filtran los datos
    def filterData(self, filter):

        self.df['Linea'] = self.df['IdMaterial'].map(str)
        self.df['Linea'] = self.df['Linea'].str.slice(0, 2)
        
        # filterlinea = ['10','11','12','13','14',
        #                '20','21','23','25','41']
        # filterlinea = ['14']
        
        # self.df         = self.df[(self.df['Linea'].isin(filterlinea))]
        filtersku= filter
        
        self.df         = self.df[(self.df['IdMaterial'].isin(filtersku))]
        
    # Para utilizar Series de Tiempo se agregan los meses sin ventas
    def addNonSell(self):
        self.Dates      = sorted(self.df.IdPeriodo.unique())
        n_ = 0
        for sku in self.IdMaterial:
            n_ += 1
            IdCeSum_sku = self.df[self.df['IdMaterial']==sku].IdCeSum.unique()
            
            for IdCeSum in IdCeSum_sku:
                CentroSuministrador = list(self.df[(self.df['IdMaterial']==sku) &
                                    (self.df['IdCeSum']==IdCeSum)]['CentroSuministrador'].unique())[0]
        
                date_min = min(self.df[(self.df['IdMaterial']==sku) &
                                          (self.df['IdCeSum']==IdCeSum)].IdPeriodo.unique())
                
                date_max = max(self.df[(self.df['IdMaterial']==sku) &
                                          (self.df['IdCeSum']==IdCeSum)].IdPeriodo.unique())
                
                Dates_aux = [date_ for date_ in self.Dates if date_ >= date_min and date_ <= date_max]
        
                Dates_IdCeSum = self.df[(self.df['IdMaterial']==sku) &
                                          (self.df['IdCeSum']==IdCeSum)].IdPeriodo.unique()
                
                Dates_add = set(Dates_aux) - set(Dates_IdCeSum) 
                Dates_add = sorted(list(Dates_add), reverse=False)
                
                for date_ in Dates_add:
                    month = date_.astype('datetime64[M]').astype(int) % 12 + 1
                    year  = date_.astype('datetime64[Y]').astype(int) + 1970
                    if month == 1:
                        month2 = 12
                        year2  = year-1
                    else:
                        month2 = month-1
                        year2 = year
                    
                    costo = list(self.df[(self.df['Periodo']==year2) & 
                                      (self.df['MesPeriodo']==month2) &
                                      (self.df['IdMaterial']==sku) &
                                      (self.df['IdCeSum'] == IdCeSum)]['MntCosto'].unique())[0]
        
                    d = {'MesPeriodo': [month],
                        'Periodo': [year],
                        'IdMaterial': [sku],
                        'MntNeto': [0],
                        'MntCosto': [costo],
                        'CtdadUMBase': [0],
                        'IdCeSum': [IdCeSum],
                        'IdPeriodo': date_,
                        'CentroSuministrador': [CentroSuministrador],
                        # 'PrecioUnitario': [precio]
                        }
                    
                    DF_aux = pd.DataFrame(data=d)

                    self.df = self.df.append(DF_aux, ignore_index=True)

    '''
    Preprocesa la data
    '''
    def preprocesing(self, filter):
        t0 = time.time()

        
        self.getData()
        self.updateFormat()

        # self.filterData(filter)
        # self.addNonSell()
                
        # Guarda la base de datos
        path = os.path.join(data_path,'DataCooprinsem.csv')
        self.df.to_csv(path)
        
    '''
    Agrega la data externa
    '''
    def addDataExternal(self):
        
        self.dfreg = self.df[['IdMaterial',
                              'MesPeriodo',
                              'IdPeriodo',
                              'Periodo',
                              'IdCeSum',
                              'CentroSuministrador',
                              'CtdadUMBase',
                              'UMBase',
                              'MntCosto'
                               # 'MntNeto',
                               # 'PrecioUnitario'
                              ]]
                
        
        print('Ingresa a addDataExternal:',self.dfreg.IdPeriodo.max())
        
        print('N° de filas: %s'%len(self.dfreg))
        
        dfMacro       = ReadSQL('coo_macroeconomia')
        dfMeteoro     = ReadSQL('coo_metorologia')
        dfPrecioCarne = ReadSQL('coo_preciocarne')
        dfPrecioleche = ReadSQL('coo_precioleche')
        
        dfMeteoro['precip'] = dfMeteoro.precip.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['tmin']   = dfMeteoro.tmin.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['tmax']   = dfMeteoro.tmax.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['tmean']  = dfMeteoro.tmean.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['dia_lluvia']   = dfMeteoro.dia_lluvia.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['dia_helada']   = dfMeteoro.dia_helada.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['dia_caluroso'] = dfMeteoro.dia_caluroso.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['Periodo'] = dfMeteoro.Periodo.replace('None',np.nan).replace('',np.nan)
        dfMeteoro['MesPeriodo'] = dfMeteoro.MesPeriodo.replace('None',np.nan).replace('',np.nan)

        
        dfMeteoro = dfMeteoro.astype({'precip'        :'float',
                                      'tmin'          :'float',
                                      'tmax'          :'float', 
                                      'tmean'         :'float',  
                                      'dia_lluvia'    :'float',  
                                      'dia_helada'    :'float',  
                                      'dia_caluroso'  :'float',
                                      'Periodo'       :'int',
                                      'MesPeriodo'    :'int'})
        
        dfPrecioleche = dfPrecioleche[dfPrecioleche['Producto']=='Leche Fluida Entera']
        dfPrecioleche = dfPrecioleche[['IdPeriodo','Precio_promedio']]
        dfPrecioleche.rename(columns={"Precio_promedio": "Precio_leche"}, inplace=True)

        dfPrecioCarne = dfPrecioCarne.groupby(['IdPeriodo']).agg({'Precio_promedio':'mean'}).reset_index()
        dfPrecioCarne.rename(columns={"Precio_promedio": "Precio_carne"}, inplace=True)

        dfPrecioCarne['Precio_carne'] = pd.to_numeric(dfPrecioCarne['Precio_carne'])
        dfPrecioleche['Precio_leche'] = pd.to_numeric(dfPrecioleche['Precio_leche'])

        # Crea DataFrame con datos meteorologico por IdCeSum-Periodo-MesPeriodo (promedio o suma según corresponda)
        dfClima = dfMeteoro.groupby(['CentroSuministrador','Periodo','MesPeriodo']).agg({'precip'        :'mean',
                                                                                         'tmin'          :'mean',
                                                                                         'tmax'          :'mean', 
                                                                                         'tmean'         :'mean',  
                                                                                         'dia_lluvia'    :'sum',  
                                                                                         'dia_helada'    :'sum',  
                                                                                         'dia_caluroso'  :'sum' }).reset_index()

        # Paso 0: Eliminar filas con IdCeSum = nan
        self.dfreg.dropna(subset=['IdCeSum'], inplace=True)
        print('Paso 0 :N° de filas: %s'%len(self.dfreg))
        
        dfClima['IdPeriodo'] = dfClima['Periodo'].map(str) + '-' + dfClima['MesPeriodo'].map(str)

        dfMacro["IdPeriodo"]       = pd.to_datetime(dfMacro['IdPeriodo'], format="%Y/%m")
        dfClima["IdPeriodo"]       = pd.to_datetime(dfClima['IdPeriodo'], format="%Y/%m")
        dfPrecioCarne["IdPeriodo"] = pd.to_datetime(dfPrecioCarne['IdPeriodo'], format="%Y/%m")
        dfPrecioleche["IdPeriodo"] = pd.to_datetime(dfPrecioleche['IdPeriodo'], format="%Y/%m")
                
        self.dfreg = pd.merge(self.dfreg, dfMacro,  how='left',    on=['IdPeriodo','MesPeriodo','Periodo']) #left_on, right_on
        print('Paso 1 :N° de filas: %s'%len(self.dfreg),self.dfreg.IdPeriodo.max())

        self.dfreg['CentroSuministrador'] = self.dfreg['CentroSuministrador'].str.strip()
        self.dfreg['CentroSuministrador'] = np.where((self.dfreg.CentroSuministrador == 'Santiago'),'Chimbarongo-Santiago',self.dfreg.CentroSuministrador)
        self.dfreg['CentroSuministrador'] = np.where((self.dfreg.CentroSuministrador == 'Chimbarongo'),'Chimbarongo-Santiago',self.dfreg.CentroSuministrador)
                
        self.dfreg = pd.merge(self.dfreg, dfClima, how='left', on=['CentroSuministrador','IdPeriodo','MesPeriodo','Periodo']) #left_on, right_on
        print('Paso 2 :N° de filas: %s'%len(self.dfreg),self.dfreg.IdPeriodo.max())
        self.dfreg = pd.merge(self.dfreg, dfPrecioCarne, how='left',  on=['IdPeriodo']) #left_on, right_on
        print('Paso 3 :N° de filas: %s'%len(self.dfreg),self.dfreg.IdPeriodo.max())
        
        self.dfreg = pd.merge(self.dfreg, dfPrecioleche, how='left', on=['IdPeriodo']) #left_on, right_on
        print('Paso 4 :N° de filas: %s'%len(self.dfreg),self.dfreg.IdPeriodo.max())
                
        # Guarda la base de datos
        print('Guardar base de dato con data externa',self.dfreg.IdPeriodo.max())
        path = os.path.join(data_path,'DataCooprinsem14.csv')
        self.dfreg.to_csv(path)
        
        self.dfexog = pd.merge(dfClima, dfMacro, how='left', on=['Periodo','MesPeriodo','IdPeriodo'])
        self.dfexog = pd.merge(self.dfexog, dfPrecioCarne, how='left', on=['IdPeriodo'])
        self.dfexog = pd.merge(self.dfexog , dfPrecioleche, how='left',  on=['IdPeriodo'])
                
        return self.dfreg
    
    # Genera columnas dependiente de data pasada
    def columnRegressiveMean(self,data,start_date,end_date,column_agg, name_column):
        # -> Cantidad un mes luego año anterior
        data[name_column] = (data[column_agg]
                           .rolling(window=start_date)
                           .sum() - 
                           data[column_agg]
                           .rolling(window=end_date)
                           .sum())/3
        # print(data.head())
        return data   
    
    #============================================================================
    def getDataPredict(self):
        
        self.dfreg = self.addDataExternal()
        print('Entra a getDataPredict:',self.dfreg.IdPeriodo.max())
        print('getDataPredict:N° de filas inicial: %s'%len(self.dfreg))
        self.IdMaterial = self.dfreg['IdMaterial'].unique()
                
        # -> Cantidad promedio de los últimos tres meseses
        group_data      = ['IdMaterial','IdPeriodo','IdCeSum']
        column_agg_list = ['CtdadUMBase']

        date = {'3trim_mean': (13,10),'2trim_mean':(10,7),
                '1trim_mean': (7,4)  ,'0trim_mean':(4,1)}
        
        filter_column = [group for group in group_data] 
        for column_agg in column_agg_list:
            for time_ in date.keys():
                if time_ == '1trim_mean' or time_ == '0trim_mean' or column_agg == 'CtdadUMBase':
                    name_column = column_agg+'_'+time_
                    filter_column.append(name_column)
                
        df_aux = pd.DataFrame()
        n_, n_total  = 0, 0
        for sku in self.IdMaterial:
            n_total += len(list(self.dfreg[(self.dfreg['IdMaterial']==sku)]['IdCeSum'].unique())) 
        
        for sku in self.IdMaterial:
            IdCeSum = self.dfreg[(self.dfreg['IdMaterial']==sku)]['IdCeSum'].unique() 
            for IdCeSum in IdCeSum:

                dfaux = self.dfreg[(self.dfreg['IdMaterial'] == sku) & 
                                    (self.dfreg['IdCeSum']   == IdCeSum)]
                n_ += 1#len(dfaux)
                print('getDataPredict',round(100*n_/n_total,2))
                dfaux = dfaux.sort_values(by='IdPeriodo')
                for column_agg in column_agg_list:
                    for time_ in date.keys():
                        if time_ == '1trim_mean' or time_ == '0trim_mean' or column_agg == 'CtdadUMBase':
                            # -> Hace tres meseses
                            (start_date, end_date) = date[time_]
                            name_column = column_agg+'_'+time_
                            dfaux = self.columnRegressiveMean(dfaux,start_date,end_date,column_agg,name_column)
                
                df_aux = df_aux.append(dfaux, ignore_index=True)

        df_aux     = df_aux[filter_column]
        self.dfreg = self.dfreg.merge(df_aux, how='left', on=group_data)
        self.dfreg = self.dfreg.merge(self.dfumb, how='left', on='IdMaterial')

        self.dfreg = self.dfreg.fillna(0)
            
                
        return self.dfreg

    def getDataPredict2(self):
        
        self.dfreg = self.df[['IdMaterial',
                              'MesPeriodo',
                              'IdPeriodo',
                              'Periodo',
                              'IdCeSum',
                              'CentroSuministrador',
                              'CtdadUMBase',
                              #'MntCosto'
                               # 'MntNeto',
                               # 'PrecioUnitario'
                              ]]
                
        self.dfreg = self.dfreg.merge(self.dfumb,
                                      on=['IdMaterial'],
                                      how='left')
        
        self.dfpredictors = pd.read_csv(self.path_varexog,
                                        sep = ',',
                                        decimal = ',',
                                        index_col=0)
        
        self.dfreg['Periodo'] = self.dfreg['Periodo'].map(int)
        self.dfreg['MesPeriodo'] = self.dfreg['MesPeriodo'].map(int)
        self.dfreg['CentroSuministrador'] = self.dfreg['CentroSuministrador'].str.strip()
        # self.dfreg['IdPeriodo'] = pd.to_datetime(self.dfreg['IdPeriodo'],
        #                                          format="%Y/%m") 
        
        # self.dfreg.to_csv('Data/DataPredict0.csv')
        
        self.dfpredictors['Periodo'] = self.dfpredictors['Periodo'].map(int)
        self.dfpredictors['MesPeriodo'] = self.dfpredictors['MesPeriodo'].map(int)
        
        
        self.dfpredictors = self.dfpredictors.drop(['IdPeriodo'], axis=1)
        
        self.dfreg = self.dfreg.merge(self.dfpredictors,
                                      on=['Periodo',
                                          'MesPeriodo',
                                          'CentroSuministrador'],
                                      how='left')

        # self.getDataPredict()
        
        # print('Entra a getDataPredict:',self.dfexog.IdPeriodo.max())
        # print('Entra a getDataPredict:',self.dfexog.IdPeriodo.min())
        # print('getDataPredict:N° de filas inicial: %s'%len(self.dfexog))
        # # self.IdMaterial = self.dfexog['IdMaterial'].unique()
                
        # # -> Cantidad promedio de los últimos tres meseses
        # group_data      = ['IdPeriodo','CentroSuministrador']
        # column_agg_list = ['Precio_carne','Precio_leche',
        #                    'tmean','precip','dia_lluvia','dia_helada',
        #                    'tmin','tmax','dia_caluroso']
        # date = {'3trim_mean': (13,10),'2trim_mean':(10,7),
        #         '1trim_mean': (7,4)  ,'0trim_mean':(4,1)}
        
        # filter_column = [group for group in group_data] 
        # for column_agg in column_agg_list:
        #     for time_ in date.keys():
        #         if time_ == '1trim_mean' or time_ == '0trim_mean' or column_agg == 'CtdadUMBase':
        #             name_column = column_agg+'_'+time_
        #             filter_column.append(name_column)
                
        # df_aux = pd.DataFrame()
        # IdCeSum = self.dfexog['CentroSuministrador'].unique() 
        # for idcesum in IdCeSum:

        #     dfaux = self.dfexog[(self.dfexog['CentroSuministrador']   == idcesum)]

        #     dfaux = dfaux.sort_values(by='IdPeriodo')

        #     for column_agg in column_agg_list:

        #         for time_ in date.keys():
        #             if time_ == '1trim_mean' or time_ == '0trim_mean' or column_agg == 'CtdadUMBase':
        #                 # -> Hace tres meseses
        #                 (start_date, end_date) = date[time_]
        #                 name_column = column_agg+'_'+time_
        #                 dfaux = self.columnRegressiveMean(dfaux,start_date,end_date,column_agg,name_column)
            
        #     df_aux = df_aux.append(dfaux, ignore_index=True)

        # df_aux     = df_aux[filter_column]
        # self.dfreg = self.dfreg.merge(df_aux, how='left', on=group_data)

        path = os.path.join(data_path,'DataPredict.csv')
        self.dfreg.to_csv(path)
                        
        return 0#self.dfreg
    
 