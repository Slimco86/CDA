# Coupled Dipole Approximation (CDA)
Coupled Dipole Approximation with Linux parallel compatibility.

Based on previouse achievement of Bala  http://juluribk.com/2016/07/20/coupled-dipole-approximation-in-python/, 
I continued to extend the functionality and generality of the CDA. Main changes to the core calculations are 
the orientation of the excitation, now the electromagnetic field can be incident at different incidence and azimuthal angles.
Secondly, now the nanoparticles can have non spherical shapes, approximated by the ellipsoid with the 3-axis given by the use.
The third thing, the core calculations were rewritten in Cython and the 4-fold increase in computation speed was achieved.
Finally, the code was completly rewritten to achieve parallelization in terms of computation wavelength. The most simple way
to use it is to run it over the linux parallel utility as a list of arguments (see-below).

# A bit introduction to the principle.

In simple case the objective of the program is to calculate the responce of an electric dipole, its scattering,absorption,extinction, 
orientation of the dipole and its field. This is done by assuming the polarizability tensors of the medium and the nanoparticles, for each particle, resulting in the 3Nx3N inverse polarizability matrix α, where N is the total number of the dipoles.
The dipole moment matrix is determined as P=αE, where E is a self-consistent field E=E_inc+E_ind, where E_inc is an incident field and E_ind is the induced field of the dipole array. The polarizability α is a tensor, which is scalar for a sphere, but also
can be tensor and non-diagonal if the particle is assumed to have some elliptical shape. In this case the depolarization factor of the dipole is calculated and the tensor is modified accordingly. This allows user to calculate the response of the non spherical particles which can be approximated by a dipole (for gold up to 100 nm in radius, based on Mie expansion coefficients).

# Use cases
 As an exaple here is extinction efficiency for Au nanoparticle in n=1.5 medium of in plane radius 60 nm
and different aspect ratio in respect to an out-of-plane axis.

![alt text](https://github.com/Slimco86/CDA/blob/master/pictures/Aspect-ratio.png)


The LD class (LD.py) provides a material library with a variaty of metals and few dielectrics, which are generated based on Lorentz-Drude equation in case of metals, or Cauchy equation in case of dielectrics.

![alt text](https://github.com/Slimco86/CDA/blob/master/pictures/Mat_dep.png)

In more complex cases, one whants to simmulate the response of an arrangements of the dipoles. In cases with metallic nanoparticles
the dipole approximation works up to around 100nm in radius. So if we are dealing with particles of such sizes, we can mimic the response 
of the lattice. In the Lattice class (Lattice.py) a variety of the lattices is avaliable : Square, Honeycomb, Hexagonal and a couple of 
Graphyne-based structures. The structures can be easily extended. 

![alt text](https://github.com/Slimco86/CDA/blob/master/pictures/lattice.jpg)


One can be interested in the dispersion relation of such arrangements. In such a case it is easy to calculate the responce at several angles of incidence. As the CDA is a self consistent method, the contributions from the lattice (i.e. the diffraction and interference) is also calculated as an interference of the dipoles. Using avaliable visualization scripts one can plot the dispersion relation. As shown here, a particular example for the particles of 60 nm in-plane radius and 24 nm height, arranged in the square lattice with period of 600 nm and embeded in the medium with refractive index n=1.5. One can clearly observe a couple of diffraction orders coupled to the LSPR and leading to anomalous dispersion.

![alt text](https://github.com/Slimco86/CDA/blob/master/pictures/disp1.png)


If interested in dipolar lattice orientation and modes propagation, the user can use the Dipole_Viz.py , providing the calculated file and the wavelength in nm to obtain the animation of the oscillations.
![alt text](https://github.com/Slimco86/CDA/blob/master/pictures/Webp.net-gifmaker%20(7).gif)


Finally, it is possible to visualize the field of the dipolar arrangements projected on planes in certesian bases. As shown on the following figure, the 3-compnents of the dipole field are projected on the Z-plane which is situated 50 nm above the square lattice. 
![alt text](https://github.com/Slimco86/CDA/blob/master/pictures/Dipole%20Y.png)

With the Mueller matrix routine (MM_sub.py), one can calculate a Mueller-Jones matrix from dipolar orientation calculated for p- and s-polarizations, following the approach provided in the DDSCAT software. The Mueller matrix is visualized as slices or with azimuthal dependence dependent on the requirements.





In the plan:
1. To increase the computation speed, maybe to transfer the core code into Julia
2. Extend the material class
3. Extende the Lattice class
4. Creaste the Visualization class
5. Write a GUI, maybe web browser JS GUI
6. Write a proper documentation
