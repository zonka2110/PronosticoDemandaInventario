#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:16:25 2022

@author: sdavila
"""

import pandas as pd

name_files = [
                ('Directo','directo'),
                ('Central_Compra','central_compra'),
                ('PorSucursal','central_compra'),
                ('Central_Abastecimiento','central_abastecimiento'),
              ('Agregado','agregado')
               ]
with open('Status.txt', 'w') as f:
    f.write('') 
f.close()

dfabc = pd.read_csv('Data/ABCxSucursal.csv', sep=',', decimal=',', index_col=0)

lines = []
for idx in range(len(name_files)):
    df = pd.DataFrame()
    (folder, file) = name_files[idx]
    for n in range(6000 ):
        try:
            
            dfaux = pd.read_csv('Results/%s/%s_%s.csv'%(folder,file,n), sep=',', decimal=',')
            print(n)
            # if 'ABC' in dfaux.columns.tolist() and 'IdCeSum' in dfaux.columns.tolist():
            #     dfaux = dfaux.drop(['ABC', 'ABC_Sucursal'], axis=1)
            #     dfaux = dfaux.merge(dfabc[
            #                               ['IdMaterial','IdCeSum',
            #                                 'ABC_Sucursal','ABC'
            #                               ]
            #                               ], on=['IdMaterial','IdCeSum'], how='left')
                
            #     dfaux['ABC'] = dfaux['ABC'].fillna('B')
            #     dfaux['ABC_Sucursal'] = dfaux['ABC_Sucursal'].fillna('B')
                        
            # elif 'IdCeSum' in dfaux.columns.tolist():
            #     dfaux = dfaux.merge(dfabc[
            #                               ['IdMaterial','IdCeSum',
            #                                 'ABC_Sucursal',
            #                                 'Margen_Suc',
            #                                 'Margen_Total']
            #                               ], on=['IdMaterial','IdCeSum'], how='left')
            #     dfaux['ABC'] = dfaux['ABC'].fillna('B')
            #     dfaux['ABC_Sucursal'] = dfaux['ABC_Sucursal'].fillna('B')

            # elif 'ABC' not in dfaux.columns.tolist():
            #     dfaux = dfaux.merge(dfabc[
            #                               ['IdMaterial',
            #                                'ABC',
            #                                 'Margen_Suc',
            #                                 'Margen_Total']
            #                               ], on=['IdMaterial'], how='left')
            #     dfaux['ABC'] = dfaux['ABC'].fillna('B')
            
            df = df.append(dfaux)
            
            with open('Status.txt', 'a') as f:
                f.write('%s \n'%n) 
            f.close()
            
        except:
            continue
            # print(n)

    if folder == 'Agregado':
        df.to_excel('Results/%s.xlsx'%folder)
    else:
        df.to_csv('Results/%s.csv'%folder)
    