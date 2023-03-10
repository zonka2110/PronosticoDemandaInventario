# -*- coding: utf-8 -*-
__version__ = "1.0"
__author__ = "GPI"

# Inputs and environment generator
from GenerateData import *

import time
import os
import sys

def generatedata(sku_consult,t0):

    # Create enviorment 
    env = Cooprinsem()
    
    # Run Predict External
    with open('Estado_ejecución_Generar_Data.txt', 'w') as f:
        f.write('Ha comenzado a cargar data externa \n')
        
    t1 = time.time()
    import PredictDataExternal
    
    with open('Estado_ejecución_Generar_Data.txt', 'a') as f:
        f.write('Ha finalizado la carga de data externa: %s(s), duración actual: %s(s) \n'%(time.time()-t1,time.time()-t0))
        f.write('Ha comenzado a cargar data interna \n')
    f.close()
    
    print('Se ha cargado la Data Externa con éxito')
     
    # Preprocesar la data historica
    env.preprocesing(sku_consult)
    print('Se ha preprocesado la data historica')
    
    # Generar data para algoritmo de pronóstico
    env.getDataPredict2()
    print('Se ha generado la data del pronósitico')
    
    with open('Estado_ejecución_Generar_Data.txt', 'a') as f:
        f.write('Ha finalizado la carga de interna: %s(s), duración actual: %s(s) \n'%(time.time()-t0,time.time()-t0))
    f.close()
            
if __name__ == "__main__":
    t0 = time.time()

    sku_consult = []
    generatedata(sku_consult,t0)

    print ('\n El Tiempo de Ejecución es: %s'%(time.time() - t0))

