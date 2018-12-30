# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:05:18 2018

@author: Ievgen Voloshenko
"""
import matplotlib.pyplot as plt
import numpy as np
import os 
import re
import h5py
from math import ceil
from mpl_toolkits.mplot3d import Axes3D
from mayavi.mlab import quiver3d,points3d,flow
from mayavi import mlab

# Read hd5py file
os.chdir('Examples\\Dipole_viz\\')
file=h5py.File('Square Au AOI-0.0 Azi-45.0 15x15 R=96nm pol-p','r')

# Read the dataframe
dipole=np.array(file['p_calc'])
pos=np.array(file['pos'])
wvl=list(np.array(file['wave']))
wvl=[round(i*1e9,1) for i in wvl]
print(wvl)

wave=676  # define the wavelength at which to show dipoles
element=ceil(wvl.index(wave))

dipole_vec=np.array([[(dipole[i][element]*1e23),(dipole[i+1][element]*1e23),(dipole[i+2][element]*1e23)] for i in range(0,dipole.shape[0]-1,3)])
#print(dipole_vec[1])
X,Y,Z,U,V,W=[],[],[],[],[],[]

for l,p in zip(dipole_vec,pos):

   X.append(p[0]*1e9)
   Y.append(p[1]*1e9)
   Z.append(p[2]*1e9)
   U.append(l[0])
   V.append(l[1])
   W.append(l[2])

X=np.asarray(X)
Y=np.asarray(Y)
Z=np.asarray(Z)
U=np.asarray(U)
V=np.asarray(V)
W=np.asarray(W)

Phi=np.linspace(-5*np.pi,5*np.pi,3600)

s=mlab.quiver3d(X,Y,Z,U.real,V.real,W.real,line_width=10, scale_factor=5,mode='arrow')
ms = s.mlab_source    
p=points3d(X,Y,Z,scale_factor=90,color=(1,0,0),mode='sphere')
@mlab.animate(delay=10)
def anim():
   for phi in Phi:
      
     
      Un = U.real*np.cos(phi+np.angle(U))
      Vn = V.real*np.cos(phi+np.angle(V))
      Wn = W.real*np.cos(phi+np.angle(W))
      
      ms.set(u=Un,v=Vn,w=Wn)
      
      #mlab.savefig('{}'.format(phi), size=None, figure=None, magnification='auto')
      yield

#mlab.start_recording(ui=True)
anim()

#s.scene.movie_maker.record = True
mlab.show()

#mlab.stop_recording(file='Au Betta_Graphyne_ext AOI-0.0 Azi-45.0 10x10 R=45nm pol-s.mp4')
   

