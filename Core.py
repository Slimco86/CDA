# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:22:28 2018

@author: slimc
"""
from numpy import pi, exp, dot
import numpy as np
from numpy.linalg import inv
import scipy.integrate as integrate
import Aij_matrix




def new_Basis(k,es,ep,phi,theta):
    
    phi=np.radians(phi)/2
    theta=np.radians(theta)
    #rx=np.array([1,0,0,0,np.cos(phi),np.sin(phi),0,-np.sin(phi),np.cos(phi)]).reshape(3,3)
    Rx=np.array([np.cos(phi),0,np.sin(phi),0,1,0,-np.sin(phi),0,np.cos(phi)]).reshape(3,3)
    Ry=np.array([np.cos(phi),0,np.sin(phi),0,1,0,-np.sin(phi),0,np.cos(phi)]).reshape(3,3)
    Rz=np.array([np.cos(theta),-np.sin(theta),0,np.sin(theta),np.cos(theta),0,0,0,1]).reshape(3,3)
    
    k_n=np.dot(np.dot(np.dot(k,Rx),Ry),Rz)
    
    es_n=np.dot(np.dot(np.dot(es,Rx),Rx),Rz)
    ep_n=np.dot(np.dot(np.dot(ep,Ry),Ry),Rz)
    return k_n,es_n,ep_n


def depFactor(a,b,c):
    ac=a*1e9
    bc=b*1e9
    cc=c*1e9
    
    
    def integrand(q,ac,bc,cc):
        return(1/((q+ac**2)*np.sqrt((q+ac**2)*(q+bc**2)*(q+cc**2))))
    Int=integrate.quad(lambda q: integrand(q,ac,bc,cc),0,np.inf)
    Nx=Ny=ac*bc*cc/2*Int[0]
    Nz=1-Nx-Ny
    return np.diag([Nx,Ny,Nz])

def calc_alpha_i(a,b,c, eps, eps_surround,wvl,N_m,Nd):
    """
    :param r_eff: effective radius of the i th particle
    :param eps: Dielectric function at a certain wavelength
    :param eps_surround: Dielectric of surrounding at certain wavelength
    :return: calculate the polarlization tensor for sphere
    """
    
    
    
    def statPolarizability(a,b,c,eps,eps_surround,Nd):
        alpha=a*b*c/3*((eps-eps_surround)/(Nd*(eps-eps_surround)+eps_surround))
        return alpha
    
    al=statPolarizability(a,b,c,eps,eps_surround,Nd)
    #print(al)
    def correctedPolarizability(alpha,wvl,a,b,c,N_m):    
        k=2*N_m*pi/wvl
        alphaCorr=alpha/(1-((2j/3)*k**3)*alpha-(k**2/np.asarray([a,b,c]))*alpha)
        return alphaCorr*np.identity(3)
    
    alpha_i = correctedPolarizability(al,wvl,a,b,c,N_m)
    #print(alpha_i)
    return alpha_i

"""
def Aij_matrix(arguments):
    
    Calculates the interaction matrix between two particles
    :param arguments[0]: Aij
    :param arguments[1]: wave vector
    :param arguments[2]: vector with x,y,z location of particle i
    :param arguments[3]: vector with x,y,z location of particle j
    :return: Aij, 3x3 interaction matrix between two particles
   
    

    Aij = arguments[0]
    k = arguments[1]
    pos_i=arguments[2]
    pos_j =arguments[3]
    
    temp = pos_j - pos_i
    r_ij = np.sqrt(temp[0]**2+temp[1]**2+temp[2]**2)
    # calculate the unit vectors between two particles
    nx, ny, nz = temp / r_ij
    eikr = exp(1j * k * r_ij)

    A = (k ** 2) * eikr / r_ij
    B = (1 / r_ij ** 3 - 1j * k / r_ij ** 2) * eikr
    C = 3*B-A

    Aij[0][0] = A * (ny ** 2 + nz ** 2) + B * (3 * nx ** 2 - 1)
    Aij[1][1] = A * (nx ** 2 + nz ** 2) + B * (3 * ny ** 2 - 1)
    Aij[2][2] = A * (ny ** 2 + nx ** 2) + B * (3 * nz ** 2 - 1)

    Aij[0][1] = Aij[1][0] = nx * ny * C
    Aij[0][2] = Aij[2][0] = nx * nz * C
    Aij[1][2] = Aij[2][1] = ny * nz * C

    return Aij





    

def calc_A_matrix(A,a,b,c, wvl, N, N_m, eps, eps_surround,Nd,pos):
    k=N_m*2*pi/wvl
    
    This function calcualtes the A matrix for N particles
    :param A: 3N x 3N matrix filled with zeros - input
    :param k: wavevector
    :param N: number of spheres
    :param r_eff: vector containing the effective radius of the spheres
    :param epsilon: dielectric function of the metal
    :param eps_surround: dielectric function of the surrounding
    :return: Returns A filled with interaction terms between particles
    

    Aij = np.zeros([3, 3], dtype=complex) # interaction matrix

    for i in range(N):
        for j in range(N):
            if i == j:
                # diagonal of the matrix
                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = inv(calc_alpha_i(a,b,c, eps, eps_surround,wvl,N_m,Nd))
            elif j>i:
                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = A[3 * j: 3 * (j + 1), 3 * i: 3 * (i + 1)]               
            else:
                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = -1 * Aij_matrix([Aij, k, pos[i], pos[j]])


"""
def calc_E_inc(k_vec_k, N, pos, E0, E_inc):
    """
    Calculates the incident electric field at the centers of each particles
    :param k_vec_k: k_unit vector * k
    :param N: number of the particles
    :param pos: vector containing the position of the particles
    :param E0: Incident electric field vector
    :param E_inc: prepopulate vector of 3N size filled with zeros
    :return: E_inc: vector of 3N filled with incident electric field
    """
    for i in range(N):
        E_inc[3 * i: 3 * (i + 1)] = E0[3 * i: 3 * (i + 1)] * exp(1j * dot(k_vec_k, pos[i]))
