cimport numpy as np
import numpy as np
from Aij_matrix import *
from Core import calc_alpha_i

def calc_A_matrix(np.ndarray[np.complex128_t, ndim=2] A,double a,double b,double c,double wvl,int N,double N_m,complex eps,complex eps_surround,
                                                           np.ndarray[np.float64_t, ndim=2] Nd, np.ndarray[np.float64_t, ndim=2] pos):
    cdef double k = N_m*2*np.pi/wvl
    """
    This function calcualtes the A matrix for N particles
    :param A: 3N x 3N matrix filled with zeros - input
    :param k: wavevector
    :param N: number of spheres
    :param r_eff: vector containing the effective radius of the spheres
    :param epsilon: dielectric function of the metal
    :param eps_surround: dielectric function of the surrounding
    :return: Returns A filled with interaction terms between particles
    """

    cdef np.ndarray[np.complex128_t, ndim=2] Aij = np.zeros([3, 3], dtype=complex) # interaction matrix
    cdef Py_ssize_t i, j 
    for i in range(N):
        for j in range(N):
            if i == j:
                # diagonal of the matrix
                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = np.linalg.inv(calc_alpha_i(a,b,c, eps, eps_surround,wvl,N_m,Nd))
            elif j>i:
                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = A[3 * j: 3 * (j + 1), 3 * i: 3 * (i + 1)]               
            else:
                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = -1 * Aij_matrix([Aij, k, pos[i], pos[j]])