# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:51:37 2017

@author: Ievgen Voloshenko
"""

import os
import h5py
import math as m
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import numpy as np

def DO(lc,AOI,n,N_m):
    lc=2*m.pi/lc
    a=1
    b=-(2*m.sin(m.radians(AOI))*lc)/(N_m**2-m.sin(m.radians(AOI))*m.sin(m.radians(AOI)))
    c=-lc**2/(N_m**2-m.sin(m.radians(AOI))*m.sin(m.radians(AOI)))
    
    d=b**2-4*a*c
    
    x=(-b+m.sqrt(d))/(2*a)
    return x

def getReighleyWoods(gratingPeriod, order, AOI, azimuth, n1, n2):
	gratingPeriod = float(gratingPeriod)
	order = float(order)
	AOI = np.radians(AOI)
	azimuth = np.radians(azimuth)
	n1 = float(n1)
	n2 = float(n2)
	
	if order < 0:
		lambdaM = -(gratingPeriod/order)*(n1*np.sin(AOI)*np.cos(azimuth) + np.sqrt(n2**2-n1**2*np.sin(AOI)**2*np.sin(azimuth)**2))
	else:
		lambdaM = -(gratingPeriod/order)*(n1*np.sin(AOI)*np.cos(azimuth) - np.sqrt(n2**2-n1**2*np.sin(AOI)**2*np.sin(azimuth)**2))
	
	return lambdaM    
    


period=[180]

plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['font.size']=10
for per in period:
    wave=[]
    Q=[]
    k=[]
    energy=[]
    #AOI=[50,5,40,60,25,35,10,65,15,70,0,75,20,55,45,30]
    AOI=[]
    os.chdir('Examples\\Dispersion')
    files=os.listdir()
    for file in files:
        print(file)
        f=h5py.File(file,'r+')
        wave.append(list(f['wave']))
        Q.append(list(f['q_ext']))
        aoi=float(''.join(list(re.split(r'AOI-',file)[1])[0:4]))
        AOI.append(aoi)
   
        f.close()
    
    for it,aoi in zip(wave,AOI):
        k.append([2*m.pi/x*m.sin(m.radians(aoi))/1e7 for x in it])
        energy.append([1240/x/1e9 for x in it])
    
      
    gp=600
    en=[1240/getReighleyWoods(gp,2,aoi,0,1,1.0) for aoi in sorted(AOI)]
    kdo=[2*np.pi/gp*np.cos(np.radians(aoi))*1e2 for aoi in sorted(AOI)]
    
    
    fig=plt.figure()
    cp = plt.contourf(k, energy, Q,500,vmin=0,vmax=4,cmap='rainbow')
    plt.plot(kdo,en,'w--')
    plt.xlim([0,2])
    plt.ylim([1.2,3])


    plt.xlabel(r'$K_{\parallel} (m^{-7})$',size=15)
    plt.ylabel('Energy (eV)',size=15)
    plt.suptitle('Extinction efficiency dispersion, period={}'.format(str(per)+'nm'), fontsize=15)
    cax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    cNorm = mpl.colors.Normalize(vmin=0,vmax=4)
    cb1 = mpl.colorbar.ColorbarBase(cax, norm=cNorm,cmap='rainbow')
    axes = plt.gca()
    plt.show()
    
    #plt.savefig('{}nm.png'.format(str(per)),dpi=600)
    #plt.close()

   
    




