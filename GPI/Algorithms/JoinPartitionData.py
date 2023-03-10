#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:16:25 2022

@author: sdavila
"""

import pandas as pd

paths = [('rforest','RandomForest'), ('svm','svmtest'), ('lgbm','lgbmtest'), ('sarima','sarima')]

for idx in range(len(paths)):
    
    (folder, file) = paths[idx]
    
    df = pd.read_csv('Parameters/Categoria_B/%s/%s0.csv'%(folder,file), sep=',', decimal=',')
    
    for n in range(1,616):
        try:
            dfaux = pd.read_csv('Parameters/Categoria_B/%s/%s%s.csv'%(folder,file,n), sep=',', decimal=',')
            df = df.append(dfaux, ignore_index=True)
        except:
            continue
        
    df.to_csv('Parameters/Categoria_B/%s_complete.csv'%file)
        