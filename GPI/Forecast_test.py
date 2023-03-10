# -*- coding: utf-8 -*-
__version__ = "2.1"
__author__ = "GPI"

# Inputs and environment generator


from Predict_paralelo_local import *

import time
import os
import sys
        
def predict(sku_consult):
    
    for i in range(40):
        try:
            p = Forescast(name=i)#sys.argv[1])
            
            # Generar sugerencia de abastecimiento
            p.read(sku_consult)
            p.updateFormat()
            p.Predict()
            p.SaveSheet
            print('Se ha generado el Pronósitco')
        except:
            continue


if __name__ == "__main__":

    t0 = time.time()
    # Directorio de Base de Datos Data Interna
    # datafile ='Data/Predict/DataPredict_3236.csv' 
    sku_consult = []
    df = predict(sku_consult)# 
    
    # with open('ReportScript/forecast.txt', 'a') as f:
        # f.write('Ha finalizado el pronóstico del bloque %s: %s (s) \n'%(sys.argv[1],time.time() - t0))
    # f.close()
    
    print ('\n El Tiempo de Ejecución es: %s'%(time.time() - t0))

