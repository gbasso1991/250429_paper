#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

"""Calculo de FFT de ciclos RF.

Se obtiene amplitud y fase.

Luego se hace un calculo de tau del sistemas suponiendo que satisface esta ecuación:

La ecuación a satisfacer es: 

d M(H,t) / dt = (1/tau) ( Meq(H) - M(H,t) )

M. Shliomis, Sov. Phys. Uspekhi (Engl. transl.) 17 (2) (1974) 153.


"""


import numpy as np
import matplotlib.pyplot as plt
from   scipy.integrate import odeint
import pandas as pd
import os
from uncertainties import ufloat
from uncertainties import unumpy as unp
#%%

"""
FORMATO DE LOS ARCHIVOS LEIDOS:

# Temperatura_=_2.2981932773109244
# Concentracion g/m^3_=_19900.0
# C_Vs_to_Am_M_A/Vsm_=_116720831517.84952
# pendiente_HvsI_1/m_=_3716.3
# ordenada_HvsI_A/m_=_1297.0
# frecuencia_Hz_=_268343.75863349007
# Promedio de 238 ciclos

Tiempo_(s)	Campo_(V.s)	Magnetizacion_(V.s)	Campo_(kA/m)	Magnetizacion_(A/m)
0.000000e+00	-9.097153e-10	-8.166168e-09	-5.692573e+01	-9.510513e+02
1.000000e-08	-9.094748e-10	-8.162679e-09	-5.691069e+01	-9.509030e+02
2.000000e-08	-9.089848e-10	-8.163510e-09	-5.688002e+01	-9.506806e+02
3.000000e-08	-9.082635e-10	-8.159695e-09	-5.683489e+01	-9.503853e+02
4.000000e-08	-9.073033e-10	-8.152945e-09	-5.677481e+01	-9.500199e+02
...
...
"""

def leer_archivo_texto(ruta):
    metadatos = {}
    datos_inicio = 0  # Para saber dónde empiezan los datos numéricos

    with open(ruta, 'r') as f:
        lineas = f.readlines()

    for i, linea in enumerate(lineas):
        linea = linea.strip()
        if linea.startswith("#"):
            partes = linea[1:].split("_=_")
            if len(partes) == 2:
                clave = partes[0].strip()
                valor = partes[1].strip()
                # Intentamos convertir a número si es posible
                try:
                    valor = float(valor)
                except ValueError:
                    pass
                metadatos[clave] = valor
        elif linea:
            datos_inicio = i  # La primera línea que no empieza con # es la cabecera
            break

    # Leer los datos a partir de la línea encontrada
    df = pd.read_csv(ruta, delimiter="\t", skiprows=datos_inicio, comment="#")

    return metadatos, df

########  ARCHIVO ANALIZADO

# ruta_archivo = "/media/pedro/Datos/ciclos_Rf/Solvotermal/265kHz_150dA_100Mss_bobN1NF00_ciclo_promedio_H_M.txt" 
# #ruta_archivo = "/media/pedro/Datos/ciclos_Rf/Solvotermal/265kHz_150dA_100Mss_bobN1NF00_ciclo_promedio_H_M(1).txt"
# #ruta_archivo = "/media/pedro/Datos/ciclos_Rf/Solvotermal/265kHz_150dA_100Mss_bobN1NE5X0_ciclo_promedio_H_M.txt" 
# ruta_archivo = "/media/pedro/Datos/ciclos_Rf/135kHz_100dA_100Mss_NEdd070_ciclo_H_M.txt"
# ruta_archivo = "/media/pedro/Datos/ciclos_Rf/135kHz_100dA_100Mss_NEdd017_ciclo_H_M.txt"
# ruta_archivo = "/media/pedro/Datos/ciclos_Rf/135kHz_100dA_100Mss_NEdd007_ciclo_H_M.txt"
# #ruta_archivo = "/media/pedro/Datos/ciclos_Rf/135kHz_100dA_100Mss_NEdd093_ciclo_H_M.txt"
# ruta_archivo="/media/pedro/Datos/ciclos_Rf/NE_citrico_7,4gL/pruebas/135kHz_100dA_100Mss_NEdd093_ciclo_H_M.txt"
# #ruta_archivo="/media/pedro/Datos/ciclos_Rf/NE_citrico_7,4gL/pruebas/265kHz_150dA_100Mss_bobN1Pmag00_ciclo_promedio_H_M.txt"

ruta_archivo='135kHz_100dA_100Mss_NEdd0097_ciclo_H_M.txt'
metadatos, df = leer_archivo_texto(ruta_archivo)
#%%
##################################################################################################################3

#Constantes
pi  = np.pi
mu0 = 4*pi*1e-7                 # permeabilidad del vacio [Tm/A].
mub = 9.2740097e-24             # Magneton de Bhor [J/T].
kB  = 1.3806488e-23             # Constante de Boltzman [J/T].


#=======================
tiempo          =df['Tiempo_(s)'].values    
campo           =df['Campo_(kA/m)'].values
magnetizacion   =df['Magnetizacion_(A/m)'].values

t=tiempo
c=campo
y=magnetizacion
ruido = np.random.uniform(low=-0.1, high=0.1, size=len(y))
#y=y+0.1*ruido   
#======================= 
      
#Gráficas


"""
Ciclo M vs H y curva M vs t
"""  
plt.figure(888)
plt.clf()

f1 = plt.subplot(2,1,1)
plt.plot(campo, magnetizacion,     'k-',label='Ciclo medido')
plt.legend(loc='upper left')
f1.set_ylabel('Magnetización (A/m)')
f1.set_xlabel('Campo (A/m)')

f2 = plt.subplot(2,1,2)
plt.plot(tiempo, magnetizacion/max(magnetizacion),  'b-',label='Magnetizacion')
plt.plot(tiempo, campo/max(campo),                  'm-',label='Campo')
plt.legend(loc='upper left')
f2.set_ylabel('Magnetizacion y Campo Normalizados')
f2.set_xlabel('Tiempo (s)')



"""
Ciclo M vs H y curva M vs t
Figura 890 A
"""  
plt.figure(890)
plt.clf()

#Plot del ultimo ciclo que sera usado para la FFT
plt.subplot(3,1,1)
plt.plot(t,y/max(y),'b.',label='mag')
plt.plot(t,c/max(c),'r.',label='campo')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Ciclo usado para FFT')
plt.legend()


"""
····················   Implementamos la FFT
"""
Fs=t[-1]/(len(t)-1)
n = len(y)                       # length of the signal
k = np.arange(n)
TT = t[-1]                       # n/Fs
frq = k/TT                       # two sides frequency range
freq = frq[range(n//2)]          # one side frequency range
ff=freq[1]                       # frecuencia del experimento

#Calculamos FFT de Magnetizacion
Y = np.fft.fft(y)/n              
Y = Y[range(n//2)]
indices = np.where(abs(Y) > max(abs(Y))*0.01)[0]

#Calculamos FFT de Campo
Yc = np.fft.fft(c)/n             
Yc = Yc[range(n//2)]
indicesc = np.where(abs(Yc) > 0.2)[0]

#Reconstruimos Magnetizacion y Campo a partir de coeficientes de FFT para verificar
m_recon=0.*t

for j in range(len(indices)):
    m_recon += 2*np.real(Y[indices[j]]) * np.cos(2 * np.pi * freq[indices[j]] * t) - 2*np.imag(Y[indices[j]]) * np.sin(2 * np.pi * freq[indices[j]]* t)

c_recon = 2*np.real(Yc[indicesc]) * np.cos(2 * np.pi * ff * t) - 2*np.imag(Yc[indicesc]) * np.sin(2 * np.pi * ff* t)



"""
Ciclos RECONSTRUIDOS M vs H y curva M vs t
Figura 890 A
""" 
plt.plot(t,m_recon/max(m_recon),'k-',label='Inv_fft_M')
plt.plot(t,c_recon/max(c_recon),'k-',label='Inv_fft_C')
plt.legend()



"""
Amplitud de los armonicos de la FFT
Figura 890 B
"""
#Plot de la amplitud de los armónicos
plt.subplot(3,1,2)
plt.plot(freq, abs(Y), 'r-')
plt.xlim([0,ff*30])
plt.xlabel('freq (Hz)')
plt.ylabel('|Y(freq)|')

#==============================================================================

# Trabajamos la fase

fase = np.angle(Y[indices])%(np.pi)   #Fase en modulo Pi
#fase_unw=np.unwrap(fase)
amp=abs(Y[indices])
fasec=np.angle(Yc[indicesc])%(np.pi)  #Fase en modulo Pi
#fasec= (np.ceil(fasec * 10**4) / 10**4)#%(2 * np.pi) #Redondeo

#Para control
tanfic=np.imag(Yc[indicesc])/np.real(Yc[indicesc])
tanfim=np.imag(Y[indices])/np.real(Y[indices])

#Calculamos la Fase usada en el modelo
fase_modelo = (indices*np.angle(Yc[indicesc[0]])-fase)%pi

#Asignamos error a la Fase usada en el modelo
val_con_inc_fm1 = np.array([ufloat(v, e) for v,e in zip(fase_modelo,[0.05,0.05,0.07,0.15])])
valores_error = [0.05, 0.05, 0.07, 0.15]
error = valores_error[:len(fase_modelo)]

#Calculojamos Tau a partir de fft para cada armónico
tau_fft=np.tan(fase_modelo)/(2*pi*freq[indices])

#Calculo de la fase 
#fase_t=(indices*np.angle(Yc[indicesc[0]])-np.arctan(2*np.pi*freq[indices]*tau_fft))%(1*np.pi)
fase_m=np.arctan(2*np.pi*freq[indices]*tau_fft)%(1*np.pi)  #fase de modelo obtenido con el tau de FFT
fase_fft_mod= (indices*fasec-fase_m)%pi

#Calculo del error en la tangente de la fase del modelo
yerr_lower = abs(np.tan(fase_modelo-error)-np.tan(fase_modelo))
yerr_upper = abs(np.tan(fase_modelo+error)-np.tan(fase_modelo))


#Calculo de Tau a partir de fases del modelo y un ajuste lineal
w=1
pend=np.sum(amp**w*np.tan(fase_modelo)*freq[indices])/np.sum(amp**w*freq[indices]**2)

peso_error=1/(yerr_upper-yerr_lower)**2
pend=np.sum(peso_error**w*np.tan(fase_modelo)*freq[indices])/np.sum(peso_error**w*freq[indices]**2)


tau_FFT_fit_lineal=pend/(2*pi)

"""==============================================================================
       GRAFICOS
"""

"""
Fase de la FFT
Figura 890 C
"""
#Plot de la fase
plt.subplot(3,1,3)
plt.xlim([0,ff*30])
plt.xlabel('freq (Hz)')
plt.ylabel('fase')
plt.plot(freq[indices], fase     , 'ob',label='Fase de la FFT')
#plt.plot(freq[indices], fase_fft_mod   ,'r-' ,label='Calculada con modelo a partir de fundamental')
plt.legend()


"""
Tangente de fase del modelo
Figura 1200 
"""
#%%
plt.figure(1200)
plt.clf()
plt.plot(freq[indices], np.tan(fase_modelo)/(2*np.pi*freq[indices]) , 'ob',label=r"$\frac{\tan(n \varphi_C - \varphi_{M_n})}{2\pi f}$")
plt.title('Tangente de la fase usada en las cuentas/(2 Pi f) = TAU')
plt.plot(0,0)
plt.xlabel('freq (Hz)')
plt.legend()
plt.show()
"""
Tangente de fase del modelo = f * TAU
Figura 1200 
"""
#%%
plt.figure(1203)
#plt.clf()
plt.plot(freq[indices], np.tan(fase_modelo) , 'ob',label=r"$\tan(n \varphi_C - \varphi_{M_n})$")
plt.title('Tangente de la fase usada en las cuentas = 2 Pi f TAU')
#plt.plot(freq[indices], tau_fft                       , '-',label='f*tau')

plt.plot(np.insert(freq[indices],0,0), (np.insert(freq[indices],0,0))*pend ,'r-',label='Pesado por Amp')   
#plt.plot(freq[indices], freq[indices]*pend1 ,'b-',label='Sin peso')   

y_aux1 = unp.nominal_values(unp.tan(val_con_inc_fm1))

plt.errorbar(freq[indices], y_aux1, yerr=[yerr_lower, yerr_upper], fmt='o', capsize=5, label='Datos')
#plt.errorbar(freq[indices], y, yerr=y_err, fmt='o', capsize=5, label='Datos con incertidumbre')

plt.plot(0,0)
plt.xlabel('freq (Hz)')
plt.legend()
plt.show()



#%%
fig,ax=plt.subplots(constrained_layout=True)
ax.plot(freq[indices], np.tan(fase_modelo) , 'og',label=r"$\tan(n \varphi_C - \varphi_{M_n})$")
ax.plot(np.insert(freq[indices],0,0), (np.insert(freq[indices],0,0))*pend ,'r-',label='Pesado por Amp')
ax.errorbar(freq[indices], y_aux1, yerr=[yerr_lower, yerr_upper], fmt='.', capsize=5, label='Datos')
ax.grid()
ax.legend()
ax.set_xlabel('Frecuency (Hz)')
plt.savefig('grafico_G.png',dpi=300)
#%%
"""
Fase del modelo
Figura 1201 
"""
plt.figure(1201)
plt.clf()
plt.title('Fase usada en las cuentas')
#plt.plot(freq[indices], np.arctan(np.tan((fase_modelo))) , 'ob',label=r"$n \varphi_C - \varphi_{M_n}}$")
plt.plot(0,0)

y_aux2 = unp.nominal_values(val_con_inc_fm1)
y_err_aux2 = unp.std_devs(val_con_inc_fm1)
plt.errorbar(freq[indices], y_aux2, yerr=y_err_aux2, fmt='o', capsize=5, label=r"$n \varphi_C - \varphi_{M_n}}$")

plt.xlabel('freq (Hz)')
plt.legend()
plt.show()


"""
IMPRIMIMOS RESUMEN DE LOS RESULTADOS
"""

print('Valores obtenidos de la FFT:')
print('tau:',tau_fft)
print('Fase del campo:',np.angle(Yc[indicesc[0]]))
print('Fase fundamental:',np.angle(Y[indices[0]]))
print('Diferencia de Fase:',np.angle(Yc[indicesc[0]])-np.angle(Y[indicesc[0]]))
print('Valores obtenidos de la FFT:')
print('Tau de ajuste lineal:',tau_FFT_fit_lineal)

plt.figure(1220)
plt.plot(np.imag(Y[indices])/np.real(Y[indices]),'o')


# %%
