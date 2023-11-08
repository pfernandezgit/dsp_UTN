#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:17:44 2023

@author: alumno
"""

# Importo librerías
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Limpieza
plt.close()
#%% Parametros Senoidal
a1 = np.sqrt(2) #Amplitud normalizada de la señal
nn = 1000       # Cantidad de muestras que tiene mi señal
rr = 30         # La cantidad de realizaciones que se hacen
fs=nn           # Igualo fs a nn para que deltaf =1 
K = 10      

Ts =  1 / fs    #   Me detengo en la mitad de la frecuencia de trabajo
W0 = fs / 4     #

# fr = np.random.uniform(-1/2, 1/2, size = (1,rr))
fr = np.random.uniform(-1/2, 1/2, size = (1,rr))  # Genero la dispersión de frecuencias
W1 = W0 + fr    # cantidad de W1 igual a la cantidad de realizaciónes 
#W1 = W0 + 1/2


tt = np.arange(0, nn*Ts, Ts).reshape(nn,1)  # Defino el eje de tiempo para cada realización
tt2 = np.arange(0, 10*nn*Ts, Ts).reshape(10*nn,1) 



ttr =  tt * np.ones((nn,rr)) 
#Senoidal pura
xx = a1 * np.sin(W1*2*np.pi * ttr)
#%%
####Ventana
#####Boxcar - Ventana implicita####
wnb = sig.windows.boxcar(nn).reshape(nn,1) ##inicialmente decial 1 en vez de rr
xxb = xx * wnb
#Normalizo los valores de las señales con su ventana
XX_boxcar = xxb / np.sqrt(np.var(xxb))  

#####Flattop#####
wnf = sig.windows.flattop(nn).reshape(nn,1) ##inicialmente decial 1 en vez de rr

#Normalizo los valores de las señales con su ventana
xxf = xx * wnf
XX_flattop = xxf / np.sqrt(np.var(xxf))
#%%
#####Redundo con ceros para analizar mejor las señales
xx = np.vstack([XX_boxcar,np.zeros([(K-1)*nn,rr])])

#Grafico el espectro de las repeticiones con los ceros y diesmado. Debería coincidir en la frec central
plt.figure(1)
df = fs/K/nn
ff = np.linspace(0, (K*nn-1)*df, K*nn)
fft_boxcar   = np.fft.fft( xx,    axis = 0 )/nn
bfrec = ff <= fs/2
dp_sig   = np.abs(fft_boxcar[bfrec])**2
plt.plot( ff[bfrec], 10* np.log10(2*dp_sig),   'o:g', linewidth=1)

df = fs/nn
ff = np.linspace(0, (nn-1)*df, nn)
fft_boxcar   = np.fft.fft( XX_boxcar,    axis = 0 )/nn
bfrec = ff <= fs/2
dp_sig  = np.abs(fft_boxcar[bfrec])**2
plt.plot( ff[bfrec], 10* np.log10(2*dp_sig),   'o:r', linewidth=1)

#Redundo con ceros para analizar mejor las señales
xx = np.vstack([XX_flattop,np.zeros([9*nn,rr])])  # Zero padding

plt.figure(2)
df = fs/10/nn
ff = np.linspace(0, (10*nn-1)*df, 10*nn)
fft_flattop   = np.fft.fft( xx,    axis = 0 )/nn

bfrec = ff <= fs/2
dp_sig   = np.abs(fft_flattop[bfrec])**2
plt.plot( ff[bfrec], 10* np.log10(2*dp_sig),   'o:b', linewidth=1)

df = fs/nn
ff = np.linspace(0, (nn-1)*df, nn)
fft_flattop   = np.fft.fft( XX_flattop,    axis = 0 )/nn
bfrec = ff <= fs/2
dp_sig  = np.abs(fft_flattop[bfrec])**2
plt.plot( ff[bfrec], 10* np.log10(2*dp_sig),   'o:r', linewidth=1)

#%% Estimador de amplitud

a1_boxcar = np.abs(fft_boxcar[int(W0),:])*2
a1_flattop = np.abs(fft_flattop[int(W0),:])*2
plt.figure(3)
plt.hist(a1_boxcar,label= 'boxcar');
plt.hist(a1_flattop,label= 'flattop');
plt.legend()