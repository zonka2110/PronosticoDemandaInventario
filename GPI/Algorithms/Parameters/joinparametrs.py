#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 22:44:33 2022

@author: sdavila
"""

import pandas as pd

rf0 = pd.read_csv('RandomForest_complete.csv', sep=',', decimal='.', index_col=0)

idmaterial0 = rf0.SKU.unique().tolist()

rf0 = rf0.set_index('SKU')

rf = pd.read_csv('RandomForestcomplete.csv', sep=',', decimal='.', index_col=0)

idmaterial = rf.SKU.unique().tolist()

# rf = rf.set_index('SKU')

for sku in idmaterial0:
    # rf.loc[sku]['Params'] = 
    # rf.loc[rf.SKU == sku, "Params"]  = rf0.loc[sku]['Params']
    print(sku, rf.loc[rf.SKU == sku, "Params"], rf0.loc[sku]['Params'])