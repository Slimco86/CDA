# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:16:11 2018

@author: Ievgen Voloshenko
"""
import matplotlib.pyplot as plt
import numpy as np
import os 
import re
import h5py
import scipy.spatial.distance as dist
from math import ceil
import matplotlib as mpl
from tqdm import tqdm
import matplotlib.colors as colors


def Grid(norm,distance,nx,ny,resolution,pos):
   
   cnter_x=(max([i[0] for i in pos])+min([i[0] for i in pos]))/2
   cnter_y=(max([i[1] for i in pos])+min([i[1] for i in pos]))/2
   cnter_z=(max([i[2] for i in pos])+min([i[2] for i in pos]))/2
   
   if norm =='Z':
      grid=[[cnter_x+i,cnter_y+j,distance] for i in np.linspace(-nx/2,nx/2,ceil(nx/resolution))
            for j in np.linspace(-ny/2,ny/2,ceil(ny/resolution))]
      
   if norm =='X':
      grid=[[distance,cnter_y+i,cnter_z+j] for i in np.linspace(-nx/2,nx/2,ceil(nx/resolution))
            for j in np.linspace(-ny/2,ny/2,ceil(ny/resolution))]
      
   if norm =='Y':
      grid=[[cnter_x+i,distance,cnter_y+j] for i in np.linspace(-nx/2,nx/2,ceil(nx/resolution))
            for j in np.linspace(-ny/2,ny/2,ceil(ny/resolution))]
   
   
   grid=np.asarray(grid).reshape(-1,3) 
   
   return grid


def calc_Dipole_Field(grid,pos,dipole,wave):
   eps_0=8.854187817*10e-12
   Field_on_Grid=[]
   
   for gp in tqdm(grid):
      E=np.array([0,0,0],dtype='complex128')
      for p,dp in zip(dipole,pos):
         R=dist.euclidean(gp,dp)
         r=(np.asarray(gp)-np.asarray(dp))/R         
         E+=1/(4*np.pi*eps_0*R**3)*(3*np.dot(np.dot(p,r),r)-p)
      
      Field_on_Grid.append(E)      
         
   return Field_on_Grid

def Plot(proj,Type,grid,Field_on_Grid,pixels,animate=False):
   if Type=='Real':
      if proj=='X':
         data=[i[0].real for i in Field_on_Grid]
      if proj=='Y':
         data=[i[1].real for i in Field_on_Grid]
      if proj=='Z':
         data=[i[2].real for i in Field_on_Grid]
   
   elif Type=='Imag':
      if proj=='X':
         data=[i[0].imag for i in Field_on_Grid]
      if proj=='Y':
         data=[i[1].imag for i in Field_on_Grid]
      if proj=='Z':
         data=[i[2].imag for i in Field_on_Grid]
   
   
   elif Type=='Phase':
      if proj=='X':
         data=[np.angle(i[0],deg=True) for i in Field_on_Grid]
      if proj=='Y':
         data=[np.angle(i[1],deg=True) for i in Field_on_Grid]
      if proj=='Z':
         data=[np.angle(i[2],deg=True) for i in Field_on_Grid]
    
   try:   
       data=np.array(data).reshape(pixels,pixels)
   except ValueError:
       data=np.array(data).reshape(pixels-1,pixels-1)
       
        
   fig=plt.figure()
   plt.imshow(data.T,interpolation='gaussian',cmap='jet')                                           
   plt.axis('off')
   plt.colorbar()
   plt.show()
   
 

def calc_Dipole_Field_t(grid,pos,dipole,wave):
   eps_0=8.854187817*10e-12
   p=dipole
   Field_on_Grid=[]
   R=dist.cdist(grid,pos)
   r=np.array([g-p for g in grid for p in pos]).reshape(R.shape[0],-1,3)
   print('R-{}  , r-{}  ,   dipols-{}'.format(R.shape,r.shape,dipole.shape))
   print(dipole[0])
   factor=1/(4*np.pi*eps_0*R**3)
   #E=factor*(3*np.dot(np.dot(p,r),r)-p)

   #Field_on_Grid=list(E)
         
   return Field_on_Grid 
      

if __name__=='__main__':
   
   file=h5py.File('Examples\\Dipole_Field\\Square Au AOI-0.0 Azi-45.0 15x15 R=96nm pol-p')
   #E_inc=np.array(list(file['E_inc'])).reshape(-1,3)
   wvl=list(np.array(file['wave']))
   wvl=[round(i*1e9,1) for i in wvl]
   print(wvl)
   dipole=np.array(list(file['p_calc']))
   wave=678  # define the wavelength at which to show dipoles
   element=ceil(wvl.index(wave))

   dipole=np.array([[(dipole[i][element]),(dipole[i+1][element]),(dipole[i+2][element])] for i in range(0,dipole.shape[0]-1,3)])
     
   pos=np.array(list(file['pos']))
   file.close()
   
   distance=50e-9
   nx=800e-9
   ny=800e-9
   resolution=10e-9
   pixels=int(np.sqrt(nx*ny/resolution**2))+1
   
   grid=Grid('Z',distance,nx,ny,resolution,pos)
   
   Eg=calc_Dipole_Field(grid,pos,dipole,wave)
   Plot('Z','Real',grid,Eg,pixels,animate=False)
   