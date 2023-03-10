# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 19:00:44 2023

@author: Golden Gamers
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import string as str
import time 

contenido = os.listdir('Results/Agregado')
df = pd.DataFrame()
i = 0
for file in contenido:
    print(file)
    ddaFull = pd.read_csv('Results/Agregado/'+file)
    print(ddaFull.head())
    df.iloc[i] = ddaFull
    i+=1



    