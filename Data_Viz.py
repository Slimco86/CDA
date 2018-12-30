# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:34:08 2018

@author: slimc
"""

import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import re

plt.rcParams['axes.facecolor'] = 'white'
Data=[]
AOI=[]
d=os.chdir('Examples\\Data_viz\\')
files=os.listdir(d)
for file in files:
    print(file)
    if True:
    
       f = h5py.File(file, 'r')
       wave=f['wave']
       Q=f['q_ext']
       r_eff=round(list(f['r_eff'])[0][0]*1e9,2)
       Q_sca=f['c_ext']
       aoi=float(''.join(list(re.split(r'AOI-',file)[1])[0:4]))
       AOI.append(aoi)
       
       
       wave=[i*1e9 for i in list(wave)]
       
       #plt.plot(wave,Q,label='')
       plt.plot(wave,Q)
       plt.xlabel('Wavelength (nm)',fontsize=20)
       plt.xlim([min(wave),max(wave)])
       plt.ylim([0,7])
       Data.append(Q)
       f.close()  
    
    
    

#X,Y=np.meshgrid(wave,R)
#Data=np.asarray(Data).reshape(len(R),-1)
#plt.contourf(Y,X,Data,500,cmap='jet')
plt.legend(['0','5','10','15','20','25','30','35','40','45','50','55'])
plt.show()
