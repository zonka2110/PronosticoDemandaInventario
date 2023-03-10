#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:16:15 2022

@author: sdavila
"""

import psycopg2
import pandas as pd
import os

# Recuperar camino de la carpeta Data
import Data  
p = str(Data.__path__)
l = p.find("'")
r = p.find("'", l+1)
data_path = p[l+1:r]

def ReadSQL(name_table):

    print('Connecting to the PostgreSQL database...')
    
    conn = psycopg2.connect(
    
        host="104.196.196.191",
    
        database="cooprinsem",
    
        user="cooprinsem_read",
    
        password="coo_gpi2022")
    
    cursor = conn.cursor()
    
    postgreSQL_select_Query = "select * from public."+name_table#+" order by id_material desc limit 10"
    
    dat = pd.read_sql_query(postgreSQL_select_Query, conn)
    
    # print(dat.columns.tolist())
    conn = None
            
    if name_table == 'coo_preciocarne':
        
        dat = dat.rename(columns = {'id_precio_carne':'',
                                      'id_periodo':'IdPeriodo',
                                      'producto':'Producto',
                                      'unidad':'Unidad',
                                      'precio_minimo':'Precio_mínimo',
                                      'precio_maximo':'Precio_máximo',
                                      'precio_promedio':'Precio_promedio'})
        
    elif name_table == 'coo_precioleche':
        
        dat = dat.rename(columns = {'id_precio_leche':'',
                                      'id_periodo':'IdPeriodo',
                                      'producto':'Producto',
                                      'unidad':'Unidad',
                                      'precio_minimo':'Precio_mínimo',
                                      'precio_maximo':'Precio_máximo',
                                      'precio_promedio':'Precio_promedio'})
                     
    elif name_table == 'coo_centraldirecto':
        
        dat = dat.rename(columns = {'id_proveedor':'IdProveedor',
                                      'txt_proveedor':'TxTProveedor',
                                      'imp_nac':'Imp/Nac',
                                      'sucursal':'IdCeSum',
                                      'l_negocio':'Lnegocio',
                                      'provision':'Descripcion',
                                      'txt_id_material':'Provision',
                                      'leadtime':'LeadTime',
                                      'id_central_directo':'id_central_directo',
                                      'id_material':'IdMaterial'})
        
                
    elif name_table == 'coo_macroeconomia':
        dat = dat.rename(columns = {'id_periodo':'IdPeriodo',
                                    'periodo':'Periodo',
                                    'mes_periodo':'MesPeriodo',
                                    'dolar':'Dolar',
                                    'ipc':'IPC',
                                    'desempleo':'Desempleo',
                                    'imacec':'IMACEC'})
        
    elif name_table == 'coo_metorologia':
        dat = dat.rename(columns = {'centro_suministrador':'CentroSuministrador',
                                    'latitud':'Latitud', 
                                    'longitud':'Longitud',
                                    'periodo':'Periodo',
                                    'mes_periodo':'MesPeriodo',
                                    'dia_periodo':'DiaPeriodo',
                                    'id_periodo':'IdPeriodo',
                                    'precip':'precip',
                                    'tmin':'tmin',
                                    'tmax':'tmax',
                                    'tmean':'tmean',
                                    'dia_lluvia':'dia_lluvia',
                                    'dia_helada':'dia_helada',
                                    'dia_caluroso':'dia_caluroso',
                                    'ola_calor':'ola_calor'})

    elif name_table == 'coo_entransito':
        dat = dat.rename(columns = {'id_transito':'',
                                     'nro_pedido':'NroPedido',
                                     'pos_pedido':'PosPedido',
                                     'id_clase_compra':'IdClaseCompra',
                                     'clase_compra':'ClaseCompra',
                                     'fecha_pedido':'FechaPedido',
                                     'id_material':'IdMaterial',
                                     'id_centro':'IdCentro',
                                     'ctdad_pedido_umb':'CtdadPedidoUMB',
                                     'ctdad_entregada_umb':'CtdadEntregadaUMB',
                                     'ctdad_pendiente_umb':'CtdadPendienteUMB'})
        
    elif name_table == 'coo_ventas':
        dat = dat.rename(columns = {'mes_periodo':'MesPeriodo',
                                    'periodo':'Periodo',
                                    'id_material':'IdMaterial',
                                    'umb':'UMB',
                                    'familia1':'Familia1',
                                    'familia2':'Familia2',
                                    'familia3':'Familia3',
                                    'mnt_venta':'MntVenta',
                                    'mnt_costo':'MntCosto',
                                    'ctc_umb':'CtdUMB',
                                    'sucursal':'Sucursal'})
    
    elif name_table == 'coo_consumos':
        
        dat = dat[(dat['id_almacen']  != 'B099')  & 
                  ((dat['id_clase_mov'] == 201) |
                  (dat['id_clase_mov'] == 202) |
                  (dat['id_clase_mov'] == 261) |
                  (dat['id_clase_mov'] == 262) |
                  (dat['id_clase_mov'] == 281) |
                  (dat['id_clase_mov'] == 282) |
                  (dat['id_clase_mov'] == 601) |
                  (dat['id_clase_mov'] == 602) |
                  (dat['id_clase_mov'] == 633) |
                  (dat['id_clase_mov'] == 634) |
                  (dat['id_clase_mov'] == 653) |
                  (dat['id_clase_mov'] == 654))
                ]
        
        dat = dat.rename(columns = {'mes_periodo':'MesPeriodo',
                                     'periodo':'Periodo',
                                     'id_material':'IdMaterial',
                                     'um_base':'UMBase',
                                     'ctdad_um_base':'CtdadUMBase',
                                     'mnt_costo':'MntCosto',
                                     'id_ce_sum':'IdCeSum',
                                     'centro_suministrador':'CentroSuministrador'})
        
    elif name_table == 'coo_stock':
        
        dat = dat[(dat['almacen']  == 'B000')  |
                  (dat['almacen']  == 'B001')  |
                  (dat['almacen']  == 'B002')  |
                  (dat['almacen']  == 'B012')  
                ]
        
        dat = dat.rename(columns = {'id_stock':'',
                                    'id_material':'IdMaterial',
                                    'nombre_material':'NombreMaterial',
                                    'familia1':'Familia1',
                                    'familia2':'Familia2',
                                    'familia3':'Familia3',
                                    'sucursal':'Sucursal',
                                    'almacen':'Almacen',
                                    'ctdad_stock_umb':'CtdadStockUMB'})
                                    
    elif name_table == 'coo_zmb5tq':

        dat = dat.rename(columns = {'id_material':'IdMaterial',
                                    'id_centro':'IdCentro',
                                    'ctdad_en_transito':'CtdadInternoUMB',
                                    'FechaProceso':'fechaproceso'
                                    })
        dat['IdCentro'] = dat['IdCentro'].str.strip()
        
                                 
        
    return dat

df = ReadSQL('coo_zmb5tq')

df.to_excel('Lead_Time_Linea14.xlsx')
