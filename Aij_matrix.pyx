cimport numpy as np
import numpy as np

def Aij_matrix(arguments):
    """
    Calculates the interaction matrix between two particles
    :param arguments[0]: Aij
    :param arguments[1]: wave vector
    :param arguments[2]: vector with x,y,z location of particle i
    :param arguments[3]: vector with x,y,z location of particle j
    :return: Aij, 3x3 interaction matrix between two particles
    """
    

    cdef np.ndarray[np.complex128_t, ndim=2] Aij = arguments[0]
    cdef Py_ssize_t i, j 
    cdef double k = arguments[1]
    cdef np.ndarray[np.float64_t,ndim=1] pos_i= arguments[2]
    cdef np.ndarray[np.float64_t,ndim=1] pos_j = arguments[3]
    
    cdef np.ndarray[np.float64_t,ndim=1] temp = pos_j - pos_i
    cdef double r_ij = np.sqrt(temp[0]**2+temp[1]**2+temp[2]**2)
    # calculate the unit vectors between two particles
    cdef double nx = temp[0] / r_ij
    cdef double ny = temp[1] / r_ij
    cdef double nz = temp[2] / r_ij
    cdef complex eikr = np.exp(1j * k * r_ij)
     

    cdef complex A = (k ** 2) * eikr / r_ij
    cdef complex B = (1 / r_ij ** 3 - 1j * k / r_ij ** 2) * eikr
    cdef complex C = 3*B-A

    Aij[0][0] = A * (ny ** 2 + nz ** 2) + B * (3 * nx ** 2 - 1)
    Aij[1][1] = A * (nx ** 2 + nz ** 2) + B * (3 * ny ** 2 - 1)
    Aij[2][2] = A * (ny ** 2 + nx ** 2) + B * (3 * nz ** 2 - 1)

    Aij[0][1] = Aij[1][0] = nx * ny * C
    Aij[0][2] = Aij[2][0] = nx * nz * C
    Aij[1][2] = Aij[2][1] = ny * nz * C

    return Aij





    

