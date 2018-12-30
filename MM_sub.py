from numpy import pi, exp, dot
import numpy as np
from numpy.linalg import inv
import scipy.integrate as integrate
from scipy.spatial.distance import cdist as dist
import h5py
import matplotlib.pyplot as plt

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


def PlotBasis(phi,theta):
   k=np.array([0,0,1])
   e1=np.array([1,0,0])
   e2=np.array([0,1,0])
   kn,e1n,e2n=new_Basis(k,e1,e2,phi,theta)
   from mpl_toolkits.mplot3d import Axes3D
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   plt.quiver(0,0,0,*k)
   plt.quiver(0,0,0,*e1)
   plt.quiver(0,0,0,*e2)
   plt.quiver(0,0,0,*kn,color='r')
   plt.quiver(0,0,0,*e1n,color='k')
   plt.quiver(0,0,0,*e2n,color='g')
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_xlim([-1,1])
   ax.set_ylim([-1,1])
   ax.set_zlim([-1,1])
   plt.show()
   return


def Jones(dipole_p,dipole_s,dipole_pos, point,wvl,K,es,ep,Phi,theta):    
    
    # Calculate required variables
    k=2*pi/wvl

    k_vec,e_s,e_p=new_Basis(K,es,ep,Phi,theta)
    k_n=k*k_vec
   
    #print(dipole_pos)
    #distance=dist(point,dipole_pos)
    #print(distance)
    r_vec=(point-dipole_pos)
    #print(r_vec)
    #print(k_n)
    
    
    # Calculate Jones elements
    #print('Dipole shape : {}\n'.format(dipole_p.shape))
    #print('k_vec : {}\n'.format(k_vec.shape))
    #print('distance : {}\n'.format(distance.shape))
    #print('Distance 0 : {} \n'.format(distance[0]))
    #print('k_vec x distance[0] : {} \n'.format(k_vec*distance[1]))
    kv_dist=[np.dot(k_n,r_vec[i]) for i in range(r_vec.shape[0])]
    
    kv_dist=np.array(kv_dist).reshape(r_vec.shape[0])
    
    
    dpp=np.dot(dipole_p,np.conj(e_p))
    dps=np.dot(dipole_p,np.conj(e_s))
    dsp=np.dot(dipole_s,np.conj(e_p))
    dss=np.dot(dipole_s,np.conj(e_s))
       
    
    
    j_11=k**3*np.sum(dss*exp(-1j*kv_dist))
    
    j_12=k**3*np.sum(dsp*exp(-1j*kv_dist))
    
    j_21=k**3*np.sum(dps*exp(-1j*kv_dist))
    
    j_22=k**3*np.sum(dpp*exp(-1j*kv_dist))
    
    J=np.array([j_22,j_12,j_21,j_11]).reshape(2,2)
    
    
    
    return J,e_s,e_p,k_vec
    
def Scat_Matrix (J,Phi,pol,pol_s):
    Phi=np.radians(Phi)
    
    a=np.conj(pol[1]).dot(np.array([1,0,0]))
    b=np.conj(pol[1]).dot(np.array([0,1,0]))
    c=np.conj(pol[0]).dot(np.array([1,0,0]))
    d=np.conj(pol[0]).dot(np.array([0,1,0]))
    
    A=np.dot(pol_s[0],np.array([1,0,0]))
    B=-np.dot(pol_s[0],np.array([0,1,0]))
    
    s1=-1j*(J[1,0]*(b*A-a*B)+J[1,1]*(d*A-c*B))
    s2=-1j*(J[0,0]*(a*A+b*B)+J[0,1]*(c*A+d*B))
    s3=1j*(J[0,0]*(b*A-a*B)+J[0,1]*(d*A-c*B))
    s4=1j*(J[1,0]*(a*A+b*B)+J[1,1]*(c*A+d*B))
    
    
    S=np.array([s1,s2,s3,s4]).reshape(2,2)
    
    return S
    
def MM_calc(S):
    
    S_11= (np.absolute(S[0,0])**2+np.absolute(S[0,1])**2+np.absolute(S[1,0])**2+np.absolute(S[1,1])**2)/2
    S_12= (np.absolute(S[0,1])**2-np.absolute(S[0,0])**2+np.absolute(S[1,1])**2-np.absolute(S[1,0])**2)/2
    S_13= (S[0,1]*np.conjugate(S[1,0])+S[0,0]*np.conjugate(S[1,1])).real
    S_14= (S[0,1]*np.conjugate(S[1,0])-S[0,0]*np.conjugate(S[1,1])).imag
    S_21= (np.absolute(S[0,1])**2-np.absolute(S[0,0])**2+np.absolute(S[1,0])**2-np.absolute(S[1,1])**2)/2
    S_22= (np.absolute(S[0,0])**2+np.absolute(S[0,1])**2-np.absolute(S[1,0])**2-np.absolute(S[1,1])**2)/2
    S_23= (S[0,1]*np.conjugate(S[1,0])-S[0,0]*np.conjugate(S[1,1])).real
    S_24= (S[0,1]*np.conjugate(S[1,0])+S[0,0]*np.conjugate(S[1,1])).imag
    S_31= (S[0,1]*np.conjugate(S[1,1])+S[0,0]*np.conjugate(S[1,0])).real
    S_32= (S[0,1]*np.conjugate(S[1,1])-S[0,0]*np.conjugate(S[1,0])).real
    S_33= (S[0,0]*np.conjugate(S[0,1])+S[1,0]*np.conjugate(S[1,1])).real
    S_34= (S[0,1]*np.conjugate(S[0,0])+S[1,1]*np.conjugate(S[1,0])).imag
    S_41= (S[1,1]*np.conjugate(S[0,1])+S[0,0]*np.conjugate(S[1,0])).imag
    S_42= (S[1,1]*np.conjugate(S[0,1])-S[0,0]*np.conjugate(S[1,0])).imag
    S_43= (S[0,0]*np.conjugate(S[0,1])-S[1,0]*np.conjugate(S[1,1])).imag
    S_44= (S[0,0]*np.conjugate(S[0,1])-S[1,0]*np.conjugate(S[1,1])).real
    
    MM=np.array([S_11,S_12,S_13,S_14,S_21,S_22,S_23,S_24,S_31,S_32,S_33,S_34,S_41,S_42,S_43,S_44]).reshape(4,4)
    
    return MM
 
def MM_calc2(S):
    
    A=np.array([1,0,0,1,1,0,0,-1,0,1,1,0,0,1j,-1j,0]).reshape(4,4)
    MM=A.dot(np.kron(S,np.conjugate(S))).dot(np.linalg.inv(A))
    return MM.reshape(4,4)

def MM_norm(mm):
    MM=mm/mm[0,0]
    for i in range(4):
        for j in range(4):
            if abs(MM[i][j])<1e-10:
                MM[i][j]=0
    return MM
    
if __name__=='__main__':
    
    MM={}
    
    
    p_path='Examples\\MM_sub\\Square Au AOI-0.0 Azi-0.0 10x10 R=60nm pol-p'
    s_path='Examples\\MM_sub\\Square Au AOI-0.0 Azi-0.0 10x10 R=60nm pol-s'
    #Read data from file for p-pol
    p_file=h5py.File(p_path,'r')
    dipole_p=np.array(p_file['p_calc'])
    
    wvl1=list(p_file['wave'])
    #dipole_p=dipole_p.reshape(3,len(wvl1),-1)
    p_file.close()
    
    #Read data from file for s-pol
    s_file=h5py.File(s_path,'r')
    dipole_s=np.array(s_file['p_calc'])
    wvl=list(s_file['wave'])
    #dipole_s=dipole_s.reshape(3,len(wvl1),-1)
    positions=np.array(s_file['pos'])
    s_file.close()
    #check if number of wvl match for p and s polarizations
    assert wvl1==wvl, 'Wavelength do not match for p and s polarizations'
    point=np.array([0,0,max(wvl1)]).reshape(1,3)
    Phi=0 #polar
    for teta in range(0,5,5):
       Theta=teta#azimuthal
       if Theta not in MM.keys():
          MM[Theta]={}
       es=np.array([1,0,0])
       ep=np.array([0,1,0])
       k=np.array([0,0,1])
       
       for i,wvl in enumerate(wvl1):
           dp=np.array([[(dipole_p[j][i]),(dipole_p[j+1][i]),(dipole_p[j+2][i])] for j in range(0,dipole_p.shape[0]-1,3)])
           ds=np.array([[(dipole_s[j][i]),(dipole_s[j+1][i]),(dipole_s[j+2][i])] for j in range(0,dipole_s.shape[0]-1,3)])   
           J,es_sc,ep_sc,kn=Jones(dp,ds,positions,point,wvl,k,es,ep,Phi,Theta)
           print('New basis vectors are : k-{}, e1-{}, e2-{}\n'.format(np.cross(es_sc,ep_sc),es_sc,ep_sc))
           #print('The calculated Jones matrix is : {}'. format(J))
           print('Jones matrix \n{}'.format(J))
           S=Scat_Matrix(J,Phi,[es,ep],[es_sc,ep_sc])
           print('S- matrix \n{}'.format(S))
           mm=MM_calc(S)
           M_norm=MM_norm(mm)
           print('Wavelength: {} nm '.format(round(wvl*1e9,2)))
           print('Mueller Matrix : \n{}'.format(M_norm))
           
           if wvl*1e9 not in MM[Theta].keys():
               MM[Theta][wvl*1e9]=M_norm
           else:
               MM[wvl*1e9][Theta]+=M_norm
           
           
      
def sliceMM(MM,Thetas):
   fig=plt.figure()
   for theta in Thetas:
      assert theta in MM.keys(),'Azimuthal angle {} is not in calculated MM'.format(theta)
      WL=list(MM[theta].keys())
      
      s=1
      for i in range(0,4):
          for j in range(0,4):
             plt.subplot(4,4,s)
             plt.plot([wl for wl in sorted(WL)],[MM[theta][wl][i][j] for wl in WL],label='M_{}{}'.format(i,j))
             plt.xlim([401,1099])
             #plt.ylim([-1,1])
             #plt.legend()
             s+=1
       
      #m12=[MM[w][1][3] for w in WL]
      #plt.plot(WL,m12)
   plt.show()
           
#sliceMM(MM,range(0,360,360))       


def PolarMM(MM):
   import matplotlib as mpl
   mpl.rcParams['axes.labelsize']	= 18
   mpl.rcParams['xtick.labelsize'] = 18
   mpl.rcParams['ytick.labelsize'] = 18
   mpl.rcParams['legend.fontsize'] = 18
   mpl.rcParams['xtick.major.pad']='10'
   mpl.rcParams['ytick.major.pad']='10'
   fig=plt.subplot(4,4,16,projection='polar')
   theta=sorted(list(MM.keys()))   
   WL=sorted(list(MM[theta[0]].keys()))
   thet=np.radians(np.array(theta))
   Wl=np.array(WL)
   X,Y=np.meshgrid(thet,Wl)
   s=1
   for i in range(0,4):
      for j in range(0,4):
          ax=plt.subplot(4,4,s,projection='polar')
          data=np.array([MM[t][wvl][i][j] for t in sorted(theta) for wvl in sorted(WL)]).reshape(len(theta),len(WL)).T
          plt.contourf(X,Y*1e9,data,50,cmap='jet')
          ax.yaxis.set_major_formatter(plt.NullFormatter())
          thetaticks = np.arange(0,360,180)
          ax.set_thetagrids(thetaticks)
          #plt.legend()
          s+=1   
   plt.show()    



#PolarMM(MM)    
Mm={}
Mm['0.000']=MM
#    
from MM import MM
mm=MM()
mm.data=Mm
#mm.depIndex('0.000',0)
# mm.Plot('data','0.000','MM',1)
mm.slicePlot('data',['0.000'],[0],[],400,1000,[5],1)
#mm.PlotMMSlice('data',['0.000'],[45])
# =============================================================================
