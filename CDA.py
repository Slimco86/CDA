# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:11:25 2018

@author: slimc
"""
import matplotlib
matplotlib.use('Agg')
from Lattice import Lattice
from LD import LD
from Core import *
import argparse
from Aij_matrix import *
from calc_A_matrix import *
import numpy as np
import time
from numpy import dot, conj
from numpy.linalg import  norm
from scipy.sparse.linalg import isolve
import scipy.integrate as integrate


if __name__=='__main__':
    #Parse the arguments given throught the Terminal
    
    parse=argparse.ArgumentParser(description='Parser for CDA arguments')
    parse.add_argument('wave_min',help='Starting wavelength in meters',type=float)
    parse.add_argument('wave_max',help='End wavelength in meters',type=float)
    parse.add_argument('wave_step',help='Wavelength step for calculations',type=float)
    parse.add_argument('tol',help='Solver tolerance',type=float)
    parse.add_argument('a',help='X-radius of the particles in meters',type=float)
    parse.add_argument('b',help='Y-radius of the particles in meters',type=float)
    parse.add_argument('c',help='Z-radius of the particles in meters',type=float)
    parse.add_argument('AOI',help='Angle of incidence. ',type=float)
    parse.add_argument('Azi',help='Azimuthal orientation of the wave',type=float)
    parse.add_argument('mat',help='Material for the nanoparticle')
    parse.add_argument('N_m',help='Refractive index of the surrounding medium.',type=float)
    parse.add_argument('nx',help='Number of unit cells in x direction',type=int)
    parse.add_argument('ny',help='Number of unit cells in y direction',type=int)
    parse.add_argument('lc',help='Lattice constant (distance between nearest neighbours)',type=float)
    parse.add_argument('Lat',help='type of the lattice')
    parse.add_argument('save_dir',help='Directory to save data')
    parse.add_argument('pol', help='Initial polarization state')
    
    args = parse.parse_args() #all arguments are now stored here
    print(args)
    # Setiing up the variables for calculation
    
    #calculate depolarization factor of the given particles based on shape 
    Nd=depFactor(args.a,args.b,args.c) 
    #create array of wavelength for calculation
    wave = np.arange(args.wave_min, args.wave_max, args.wave_step)
    # Generate the dielectric function based on the given material
    if args.mat=='Air' or args.mat=='Polysterene' or args.mat=='TiO2':
        eps =LD(wave,material=args.mat,model='Cauchy')       #For dielectric particles
    elif args.mat=='VO2':
        eps=LD(wave,material=args.mat,model='RF')
    else:
        eps = LD(wave, material=args.mat, model='LD')    #For metallic nano particles
    
    AOI=np.radians(args.AOI)
    phi=np.radians(args.Azi)
    
    # incident polarization of the light, [1,0,0] means light is polarized in x, [0,1,0] mean light is polarized in y    
    k_vec,E0_vec_p,E0_vec_s=new_Basis([0,0,1],[1,0,0],[0,1,0],args.AOI,args.Azi)
    
    if args.pol=='p':
        E0_vec = E0_vec_p
    elif args.pol=='s':
        E0_vec = E0_vec_s
    
    print('k  :  {},  E0_vec  :  {}'.format(k_vec,E0_vec))
    # inciddent k vector [0,0,1] means light is travelling in +z axis
    #k_vec = np.array([np.sin(AOI)*np.cos(phi), np.sin(AOI)*np.sin(phi), np.cos(AOI)]) 
    
    
    # Dielectric function of surrounding medium
    eps_surround = args.N_m ** 2 
    # Create lattice object
    Lat=Lattice(args.nx,args.ny,args.lc,args.a,args.b,args.c)
    N,pos,r_eff=getattr(Lat,args.Lat)() # Call a class method by a string
    print('Total number of particles : {}'.format(N))
    Lat.Grid_Viz() # Visualize the grid
    
    # Create empy containers
    A = np.zeros([3 * N, 3 * N], dtype='complex') # A matrix of 3N x 3N filled with zeros
    p = np.zeros(3 * N, dtype='complex')  # A vector that stores polarization Px, Py, Pz of the each particles, we will use this for initial guess for solver
    E0 = np.tile(E0_vec, N) # A vector that has the Ex, Ey, Ez for each particles
    E_inc = np.zeros(3 * N, dtype='complex') # A vector storing the Incident Electric field , Einc_x, Einc_y, Einc_z for each particle


    n_wave = wave.size
    p_calc = np.zeros([3 * N, n_wave], dtype='complex')  # This stores the dipoles moments for each particle at different wavelengths
    c_ext = np.zeros(n_wave) # stores the extinction crosssection
    c_abs = np.zeros(n_wave) # stores the absorption crossection
    c_scat = c_ext-c_abs # stores the scattering crossection

    # Start of the computation
    start = time.clock()
    
    for w_index, w in enumerate(wave):
        start_at_this_wavelength = time.clock()

        print ('*'*100)
        print ('Running wavelength: ', w * 1E9)

        k = (2 * np.pi / w)  # wave momentum

        epsilon = eps.epsilon[w_index] # Get the dielectric constant
        
        
        calc_A_matrix(A,args.a,args.b,args.c,w, N, args.N_m, epsilon, eps_surround,Nd, pos) # Calculate the inverse polarizability matrix
        calc_E_inc(k_vec*k, N, pos, E0, E_inc) # Calculate field of each dipole
        
        # Calback function for progress trace
        iter=1
        def mycallback(xk):
            # here xk is current solution of the iteration
            global iter
            # residual is defined as 
            residual = norm(E_inc - dot(A, xk)) / norm(E_inc)
            print("{} : {}".format(iter, residual))
            iter=+1
        
        
        # Solve main equation AP = E, where A = N x N complex matrix, P and E are N-dim vector 
        #using biconjugate gradient method

        print("Iteration : Residual")
        p_calc[:,w_index], info = isolve.bicgstab(A, E_inc, callback=mycallback, x0=p, tol=args.tol, maxiter=None)
        
        if info == 0:
            print ('Successful Exit')
            # calculate the extinction crossection
            c_ext[w_index] = (4 * np.pi * k / norm(E0) ** 2) * np.sum(np.imag(dot(conj(E_inc), p_calc[:, w_index])))

            # calculate the absorption crossection
            for i in range(N):
                c_abs[w_index] += ( np.imag(dot(p_calc[3*i : 3*(i+1), w_index],
                                                    dot(conj(A[3 * i: 3 * (i + 1), 3 * i: 3 * (i + 1)]),
                                                        conj(p_calc[3*i: 3*(i+1), w_index]))
                                               )
                                          )
                                  - (2.0/3)*k**3*norm(p_calc[3*i : 3*(i+1), w_index])**2
                                  )
            c_abs[w_index] *= (4 * np.pi * k / norm(E0) ** 2)
            # calculate scattering cross section
            c_scat[w_index] = c_ext[w_index] - c_abs[w_index]

        elif info > 0:
            print ('Convergence not achieved, may be increase the number of maxiter')

        elif info < 0:
            print (info)
            print ('Illegal input')

        end_at_this_wavelength = time.clock()

        print("Elapsed Time @ this wavelenth %3.2f sec" % (end_at_this_wavelength - start_at_this_wavelength))
        
    def efficiency_calc_p(cross_section,AOI):
        r1=args.a
        r2=args.c
        r=(r1*r2)/np.sqrt(np.sin(AOI)*np.sin(AOI)*r2**2+np.cos(AOI)*np.cos(AOI)*r1**2)
        return cross_section /(np.pi * r*args.b)
        
    def efficiency_calc_s(cross_section,AOI):
        r1=args.a
        r2=args.b
        return (cross_section/(np.pi*r1*r2))
    
    if args.pol == 's':
        q_ext = efficiency_calc_s(c_ext,AOI)
        q_abs = efficiency_calc_s(c_abs,AOI)
        q_scat = efficiency_calc_s(c_scat,AOI)
    elif args.pol == 'p':
        q_ext = efficiency_calc_p(c_ext,AOI)
        q_abs = efficiency_calc_p(c_abs,AOI)
        q_scat = efficiency_calc_p(c_scat,AOI)
    
    end = time.clock()
    print("Elapsed Time %3.2f sec" % (end - start))
    
    #Saving data
    
    import h5py
    import os
    #d =  os.chdir('{}'.format(args.save_dir))
    
    f = h5py.File("{} {} AOI-{} Azi-{} {}x{} R={}nm pol-{}".format(args.mat,args.Lat,args.AOI,args.Azi,args.nx,args.ny,round(args.a*1e9),args.pol), "w")
    f.create_dataset('wave', data=wave)
    f.create_dataset('c_abs', data=c_abs)
    f.create_dataset('c_ext', data=c_ext)
    f.create_dataset('c_scat', data=c_scat)
    f.create_dataset('q_abs', data=q_abs)
    f.create_dataset('q_ext', data=q_ext)
    f.create_dataset('q_scat', data=q_scat)
    f.create_dataset('p_calc', data = p_calc)
    f.create_dataset('pos', data = pos)
    f.create_dataset('r_eff', data = r_eff)
    f.create_dataset('AOI',data=args.AOI)
    f.create_dataset('AZI', data= args.Azi)
    f.create_dataset('E_inc', data=E0_vec)
    f.close()
    
       
        
