#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:03:00 2022

@author: sdavila
"""

import pandas as pd

datafile ='Data/Data_DataPredict.csv'
df             = pd.read_csv(datafile, 
                        sep=',',
                        decimal='.',
                        index_col=0)



# consult2 = df[df['IdMaterial'] == 12000501]

IdMaterial = list(df.IdMaterial.unique())
longIdMaterial = len(IdMaterial)

n = longIdMaterial
m = int(longIdMaterial / n)

for i in range(n-1):
    status = 'w' if i == 0 else 'a'
    with open('instances.txt', status) as f:
        f.write('%s-%s\n'%(i,IdMaterial[i]))
    f.close()
    
    IdMaterialAux = IdMaterial[i*m:(i+1)*m]
    df2 = df[df['IdMaterial'].isin(IdMaterialAux)]
    df2.to_csv('Data/Predict/DataPredict_%s.csv'%(i))
    
    
with open('instances.txt', status) as f:
    f.write('%s-%s \n'%((n-1),IdMaterial[n-1]))
f.close()
    
IdMaterialAux = IdMaterial[(n-1)*m:]
df2 = df[df['IdMaterial'].isin(IdMaterialAux)]
df2.to_csv('Data/Predict/DataPredict_%s.csv'%(n-1))

# consult = pd.read_csv('Data/Predict_A/DataPredictA_69.csv',sep=',',
#                 decimal='.',
#                 index_col=0)



