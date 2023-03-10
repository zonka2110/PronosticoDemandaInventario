import os
import pandas as pd
import matplotlib.pyplot as plt
import string as str
import time 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Funciones primer caso
# Este caso contempla salidas por sku considerando todos los centros en los que se encuentra el producto

def generateFirstPlot(ddaResumen, nameSKU):

    plt.figure(figsize=(16,9))
    plt.plot(ddaResumen['ddaFecha'], ddaResumen['Valor_Inv'], label = 'Valor_Inv')
    plt.xticks(rotation = 90)
    plt.xticks(range(0, len(ddaResumen['ddaFecha']), 3))
    plt.xlabel('ddaFecha')
    plt.ylabel('Valor Inventario')
    plt.tight_layout()
    plt.title(nameSKU, fontsize = 15)
    plt.legend(loc = 'best')

    plt.savefig('Results1/Plots_SKU_Cooprinsem/'+nameSKU+'.jpg')



def getFirstSummary(idSearch, op):

    if (op == 1):
        
        for file in os.listdir('ResultsCDyAC'):
            idExcel = file 
        
            if(idExcel == idSearch):
                ddaResumen = pd.DataFrame({'ddaFecha': [], 'Valor_Inv': []})
                ddaFull = pd.ExcelFile('ResultsCDyAC/'+file)
                sheets = ddaFull.sheet_names
            
                temp = 0
                for sheet in sheets:
                    ddaCentro = pd.read_excel('ResultsCDyAC/'+file, sheet_name = sheet, usecols = [0,1,4])
                
                    if(temp == 0):
                        for i in range(len(ddaCentro)):
                            ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                            ddaResumen.loc[i,'Valor_Inv'] = ddaCentro.iloc[i]['Valor_Inv']
                
                    else:
                        for i in range(len(ddaCentro)):
                            ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                            # generar caso para evitar suma con valores negativos
                            if ( (ddaResumen.loc[i,'Valor_Inv'] >= 0) and  (ddaCentro.iloc[i]['Valor_Inv'] >= 0) ):
                                ddaResumen.loc[i,'Valor_Inv'] = ddaResumen.iloc[i]['Valor_Inv'] + ddaCentro.iloc[i]['Valor_Inv']
                            elif (ddaResumen.loc[i,'Valor_Inv'] >= 0) and (ddaCentro.iloc[i]['Valor_Inv'] < 0):
                                ddaResumen.loc[i,'Valor_Inv'] = ddaResumen.loc[i,'Valor_Inv'] + 0
                            elif (ddaResumen.loc[i,'Valor_Inv'] < 0) and (ddaCentro.iloc[i]['Valor_Inv'] >= 0):
                                ddaResumen.loc[i,'Valor_Inv'] = ddaCentro.iloc[i]['Valor_Inv']
                            else:
                                ddaResumen.loc[i,'Valor_Inv'] = 0
                    temp+=1

        nameSKU = 'sku_Glob_'+idSearch[7:].replace('.xlsx','')
        generateFirstPlot(ddaResumen, nameSKU)
        ddaResumen.to_excel('Results1/'+nameSKU+'.xlsx')

    elif(op == 2):
    
        ddaResumenFinal = pd.DataFrame({'ddaFecha': [], 'Valor_Inv': []})

        for x in range(len(idSearch)):
            idSearchAux = idSearch[x]
            tempAux = 0

            for file in os.listdir('ResultsCDyAC'):
                idExcel = file 

                if(idExcel == idSearchAux):
                    ddaResumen = pd.DataFrame({'ddaFecha': [], 'Valor_Inv': []})
                    ddaFull = pd.ExcelFile('ResultsCDyAC/'+file)
                    sheets = ddaFull.sheet_names

                    temp = 0
                    for sheet in sheets:
                        ddaCentro = pd.read_excel('ResultsCDyAC/'+file, sheet_name = sheet, usecols = [0,1,4])

                    if(temp == 0):
                        for i in range(len(ddaCentro)):
                            ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                            ddaResumen.loc[i,'Valor_Inv'] = ddaCentro.iloc[i]['Valor_Inv']
                    
                    else:
                        for i in range(len(ddaCentro)):
                            ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                            ddaResumen.loc[i,'Valor_Inv'] = ddaResumen.iloc[i]['Valor_Inv'] + ddaCentro.iloc[i]['Valor_Inv']

                    temp+=1
            
            
            if(tempAux == 0):
                for i in range(len(ddaResumen)):
                    ddaResumenFinal.loc[i,'ddaFecha'] = ddaResumen.iloc[i]['ddaFecha']
                    ddaResumenFinal.loc[i,'Valor_Inv'] = ddaResumen.iloc[i]['Valor_Inv']

            else:
                for i in range(len(ddaResumen)):
                    ddaResumenFinal.loc[i,'ddaFecha'] = ddaResumen.iloc[i]['ddaFecha']
                    ddaResumenFinal.loc[i,'Valor_Inv'] = ddaResumenFinal.iloc[i]['Valor_Inv'] + ddaResumen.iloc[i]['Valor_Inv']

            tempAux+=1

        nameSKU = 'sku_Glob_'+idSearch[0][7:].replace('.xlsx','')
        generateFirstPlot(ddaResumen, nameSKU)
        ddaResumen.to_excel('Results1/'+nameSKU+'.xlsx')

    else:
        print('error con op')

   
            
def remove(lst):
    res = []
    check = set()

    for x in lst:
        hsh = tuple(sorted(x))
        if hsh not in check:
            res.append(x)
            check.add(hsh)
    return res



def firstCase():

    listIDS = pd.DataFrame({'ID': []})
    i = 0
    for file in os.listdir('ResultsCDyAC'):
        listIDS.loc[i,'ID'] = file
        i+=1
    idMatch = []
    for i in range(len(listIDS)):
        idAux = []
        idAux.append(i)
        for j in range(len(listIDS)):
            if( ( (int(listIDS.iloc[i]['ID'][7:].replace('.xlsx',''))) == (int(listIDS.iloc[j]['ID'][7:].replace('.xlsx',''))) ) and (i != j) ):
                idAux.append(j)
        idMatch.append(idAux)
    idMatch = remove(idMatch)
    for i in range(len(idMatch)):
        if(len(idMatch[i]) == 1):
            for j in range(len(listIDS)):       
                if(listIDS.index[j] == idMatch[i][0]):
                    idSearch = listIDS.iloc[idMatch[i][0]]['ID']
                    getFirstSummary(idSearch, 1)
                    
        if(len(idMatch[i]) != 1):
            idSearch = []
            for x in range(len(idMatch[i])):
                for j in range(len(listIDS)):
                    if(listIDS.index[j] == idMatch[i][x]):
                        idSearch.append(listIDS.iloc[idMatch[i][x]]['ID'])
          
            getFirstSummary(idSearch, 2)



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Funciones segundo caso

# Este caso contempla salidas por centro, considerando todos los skus correspondientes por cada sucursal

def generateSecondPlot(ddaResumen, namePlot):

    plt.figure(figsize=(16,9))
    plt.plot(ddaResumen['ddaFecha'], ddaResumen['Valor_Inv'], label = 'Valor_Inv')
    plt.xticks(rotation = 90)
    plt.xticks(range(0, len(ddaResumen['ddaFecha']), 3))
    plt.xlabel('ddaFecha')
    plt.ylabel('Valor Inventario')
    plt.tight_layout()
    plt.title(namePlot, fontsize = 15)
    plt.legend(loc = 'best')

    plt.savefig('Results2/Plots_Centros_Cooprinsem/'+namePlot+'.jpg')



def getSecondSummary(centers):

    for x in range (len(centers)):
        
        ddaResumen = pd.DataFrame({'ddaFecha': [], 'Valor_Inv': []})

        for file in os.listdir('ResultsCDyAC'):

            ddaFull = pd.ExcelFile('ResultsCDyAC/'+file)
            sheets = ddaFull.sheet_names
            
            for sheet in sheets:
                
                if(sheet.split('_')[1] == centers[x]):
                    ddaCentro = pd.read_excel('ResultsCDyAC/'+file, sheet_name = sheet, usecols = [0,1,4])
                    
                    if(ddaResumen.empty):
                        for i in range(len(ddaCentro)):
                            ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                            ddaResumen.loc[i,'Valor_Inv'] = ddaCentro.iloc[i]['Valor_Inv']


                    else:
                        for i in range(len(ddaCentro)):
                            ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                            ddaResumen.loc[i,'Valor_Inv'] = ddaResumen.iloc[i]['Valor_Inv'] + ddaCentro.iloc[i]['Valor_Inv']


        name = 'centro_Glob_'+centers[x]
        generateSecondPlot(ddaResumen, name)
        ddaResumen.to_excel('Results2/'+name+'.xlsx') 
        


def secondCase():

    listIDS = pd.DataFrame({'ID': []})

    i = 0
    for file in os.listdir('ResultsCDyAC'):
        listIDS.loc[i,'ID'] = file
        i+=1

    centersAux = []
    for file in os.listdir('ResultsCDyAC'):
        ddaCenters = pd.ExcelFile('ResultsCDyAC/'+file)
        sheets = ddaCenters.sheet_names
        
        for i in range(len(sheets)):
            centersAux.append(sheets[i].split('_')[1])
    
    seen = set()
    centers = []
    for item in centersAux:
        if item not in seen:
            seen.add(item)
            centers.append(item)


    getSecondSummary(centers)



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Funciones tercer caso
# Este caso contempla el nivel mas alto, el nivel Cooprinsem
# Considera todos los skus y todos los centros
# Su entrada es el resumen obtenido por centros (entrada ubicada en carpeta Results 2; salida ubicada en carpeta Results 3)


def generateThirdPlot(ddaResumen, namePlot):

    plt.figure(figsize=(16,9))
    plt.plot(ddaResumen['ddaFecha'], ddaResumen['Valor_Inv'], label = 'Valor_Inv')
    plt.xticks(rotation = 90)
    plt.xticks(range(0, len(ddaResumen['ddaFecha']), 3))
    plt.xlabel('ddaFecha')
    plt.ylabel('Valor Inventario')
    plt.tight_layout()
    plt.title(namePlot, fontsize = 15)
    plt.legend(loc = 'best')

    plt.savefig('Results3/Plot_Cooprinsem/'+namePlot+'.jpg')



def thirdCase():

    ddaResumen = pd.DataFrame({'ddaFecha': [], 'Valor_Inv': []})

    for file in os.listdir('Results2'):
        if('xlsx' in file):
            ddaFull = pd.ExcelFile('Results2/'+file)
            sheets = ddaFull.sheet_names

            for sheet in sheets:
                ddaCentro = pd.read_excel('Results2/'+file, sheet_name = sheet, usecols = [0,1,2])

                if(ddaResumen.empty):
                    for i in range(len(ddaCentro)):
                        ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                        ddaResumen.loc[i,'Valor_Inv'] = ddaCentro.iloc[i]['Valor_Inv']

                else:
                    for i in range(len(ddaCentro)):
                        ddaResumen.loc[i,'ddaFecha'] = ddaCentro.iloc[i]['ddaFecha']
                        ddaResumen.loc[i,'Valor_Inv'] = ddaResumen.iloc[i]['Valor_Inv'] + ddaCentro.iloc[i]['Valor_Inv']
    

    name = 'valorInv_Cooprinsem'
    generateThirdPlot(ddaResumen, name)
    ddaResumen.to_excel('Results3/'+name+'.xlsx') 




# Ejecuta secuencialmente los 3 resumenes descritos anteriormente

def main():
    inicio = time.time()
    
    firstCase()
    secondCase()
    thirdCase()
    
    fin = time.time()
    
    print('tiempo ejecucion: ',fin-inicio)


main()