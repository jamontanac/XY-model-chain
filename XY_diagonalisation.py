import numpy as np
from scipy.linalg import circulant,toeplitz, hankel, expm
import pickle
import matplotlib.pylab as plt
from multiprocessing import Pool
from functools import partial
import time
try:
    import pyfftw
except ImportError:
    pass
    #warnings.warn("I couldn't find the package pyfftw, please check the location of it and re run it. Our suggestion is to install it to use the functions that have been optimized")



class Sampling_Random_State:

    #### --------- Definition of variables ------------------------
    N_size=500001
    Gamma=0.5
    Lambda=0.5
    num_data = 200
    mu = 0.0
    
    #### ------------------------------------------------------------

    @classmethod
    def Alpha(cls,theta:np.float64)-> np.float64:
        return cls.Lambda+np.cos(theta)
    @classmethod
    def Beta(cls,theta:np.float64)-> np.float64:
        return cls.Gamma*np.sin(theta)
    @classmethod
    def Omega(cls,theta:np.float64)-> np.float64:
        return np.sqrt(cls.Alpha(theta)**2 + cls.Beta(theta)**2 )
    @classmethod
    def Phi(cls,theta:np.float64)-> np.float64:
        return np.arctan2(cls.Beta(theta),cls.Alpha(theta))



    @classmethod
    def Fermi_dirac(cls,n:np.int64,beta:np.float64) -> np.float64:
        # beta is the inverse thermic energy associated in the system (beta)
        # mu corresponds to the chemical potential
        # n is the position of the particle
        # f=np.exp(T*(Omega(Gamma,Lambda,2.0*(np.pi/N)*n)-mu)) +1
        # N corresponds to the size of the system

        f=np.exp(beta*(cls.Omega(((2.*np.pi)/np.float64(cls.N_size)) * n)-cls.mu)) +1
        return 1/f
    @classmethod
    def Sample_number_sin_cos(cls,Ground:bool = False)-> list:
        x=np.arange(0,(cls.N_size-1)/2+ 1)
        beta = np.min(cls.Omega(np.linspace(-np.pi,np.pi,np.int64(1000))))
        if Ground:
            m_cos=[-0.5 for i in x]
            m_sin=[-0.5 for i in x]
        else:
            m_cos=[-0.5 if np.random.random()>cls.Fermi_dirac(n=i,beta=beta) else 0.5 for i in x]
            m_sin=[-0.5 if np.random.random()>cls.Fermi_dirac(n=i,beta=beta) else 0.5 for i in x]
        return m_sin,m_cos
    @classmethod
    def Sample_numbers(cls,number:np.int64,rank:np.int64 = 0,Ground:bool=False)->np.ndarray:
        np.random.seed(rank*cls.num_data+number)
        beta = np.min(cls.Omega(np.linspace(-np.pi,np.pi,np.int64(1000))))
        x=np.arange(0,(cls.N_size-1)/2+ 1)
        if Ground:
            m_cos=[-0.5 for i in x]
            m_sin=[-0.5 for i in x]
        else:
            m_cos=[-0.5 if np.random.random()>cls.Fermi_dirac(n=i,beta=beta) else 0.5 for i in x]
            m_sin=[-0.5 if np.random.random()>cls.Fermi_dirac(n=i,beta=beta) else 0.5 for i in x]
        
        n = np.zeros(cls.N_size)
        n[::2] =np.array(m_cos)+0.5
        n[1::2] = np.array(m_sin[:-1])+0.5
        return n
        
    @classmethod
    def Sample_State(cls,Ground:bool =False)-> np.ndarray:
        m_sin,m_cos = cls.Sample_number_sin_cos(Ground=Ground)
        x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+1)
        M_minous=[((m_cos[np.abs(int(i))]-m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
        M_plus = [((m_cos[np.abs(int(i))]+m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
        Mminousband=np.array(M_minous)
        Mplusband=np.array(M_plus)
        return Mminousband,Mplusband

    @classmethod
    def Get_Bands_Matrix(cls,Ground:bool =False,Cluster:bool = False)-> np.ndarray:
        Mminous, Mplus = cls.Sample_State(Ground=Ground)
        x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+ 1)
        if Cluster:
            M_plus=np.fft.ifftshift(Mplus)
            M_minous=np.fft.ifftshift(Mminous)
            Fourier_minous=np.fft.fft(M_minous)
            Fourier_plus=np.fft.fft(M_plus)

        else:
            M_plus=pyfftw.empty_aligned(cls.N_size, dtype='complex128')
            M_plus[:]=np.fft.ifftshift(Mplus)
            M_minous=pyfftw.empty_aligned(cls.N_size, dtype='complex128')
            M_minous[:]=np.fft.ifftshift(Mminous)
            Fourier_minous=pyfftw.interfaces.numpy_fft.fft(M_minous)
            Fourier_plus=pyfftw.interfaces.numpy_fft.fft(M_plus)
        return Fourier_minous/cls.N_size, Fourier_plus/cls.N_size


    @classmethod
    def Toeplitz_matrix(cls,Fourier_P:np.ndarray,L:np.int64)-> np.ndarray:
        First_column = Fourier_P[:L]
        First_row = np.roll(Fourier_P,-1)[::-1][:L]
        return toeplitz(First_column,First_row)

    @classmethod
    def Hankel_matrix(cls,Fourier_M:np.ndarray,L:np.int64)-> np.ndarray:
        to_use=Fourier_M[:2*L-1]
        First_column=to_use[:L]
        Last_row=np.roll(to_use,-L+1)[:L]
        return hankel(First_column,Last_row)

    @classmethod
    def Covariance_matrix(cls,L:np.int64,Ground:bool=False)-> np.ndarray:
        Fourier_minous,Fourier_plus=cls.Get_Bands_Matrix(Ground=Ground)
        return (cls.Toeplitz_matrix(Fourier_plus,L)+cls.Hankel_matrix(Fourier_minous,L))
    @classmethod
    def Covariance_matrix_from_sub_sample(cls,Fourier_plus:np.ndarray,L:np.int64,Fourier_minous:np.ndarray=None,Circulant:bool=False)-> np.ndarray:
        if Fourier_minous is None:
            if Circulant:
                Cov_Matrix=(cls.Toeplitz_matrix(Fourier_plus,L))
                M_corner=np.zeros((L,L))
                Cov_Matrix[0,L-1],Cov_Matrix[L-1,0] = 0.0,0.0
                M_corner[0,L-1],M_corner[L-1,0]=Cov_Matrix[1,0],Cov_Matrix[0,1]
                return M_corner + Cov_Matrix
            else:
                return (cls.Toeplitz_matrix(Fourier_plus,L))
        else:
            if Circulant:
                Cov_Matrix=(cls.Toeplitz_matrix(Fourier_plus,L)+cls.Hankel_matrix(Fourier_minous,L))
                M_corner=np.zeros((L,L))
                Cov_Matrix[0,L-1],Cov_Matrix[L-1,0] = 0.0,0.0
                M_corner[0,L-1],M_corner[L-1,0]=Cov_Matrix[1,0],Cov_Matrix[0,1]
                return M_corner + Cov_Matrix
            else:
                return (cls.Toeplitz_matrix(Fourier_plus,L)+cls.Hankel_matrix(Fourier_minous,L))

    @classmethod
    def get_band_of_matrix(cls,Matrix:np.ndarray,num_band:np.int64)-> np.ndarray:
        L,C=Matrix.shape
        if L!=C:
            raise ValueError("Only squared matrix can be computed")
        if num_band > 0:
            return np.array([[Matrix[i,j] for i in range(num_band,L) if i-j == num_band] for j in range(L-num_band)]).reshape(L-num_band)
        elif num_band <0:
            return np.array([[Matrix[i,j] for i in range(L) if i-j == num_band] for j in range(-num_band,L)]).reshape(L+num_band)
        else:
            return np.diagonal(Matrix)
    @classmethod
    def Binary_entropy(cls,x:np.ndarray)->np.ndarray:
        result=[0 if np.abs(i-1)<10E-12 or np.abs(i)<10E-12 else -i*np.log(i)-(1-i)*np.log(1-i) for i in x]
        return np.array(result)


class Computations_XY_model(Sampling_Random_State):

    beta = np.min(Sampling_Random_State.Omega(np.linspace(-np.pi,np.pi,int(1000))))

    @classmethod
    def Sample_Fermi_dirac(cls,n:np.int64,Size:np.int64) -> np.float64:
        # beta is the inverse thermic energy associated in the system (beta)
        # mu corresponds to the chemical potential
        # n is the position of the particle
        # f=np.exp(T*(Omega(Gamma,Lambda,2.0*(np.pi/N)*n)-mu)) +1
        # N corresponds to the size of the system
        f=np.exp(cls.beta*(cls.Omega(((2.*np.pi)/np.float64(Size)) * n)-cls.mu)) +1
        return 1/f

    @classmethod
    def Compute_svd_Cov_Matrix(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64,Circulant:bool = False,Complete:bool = True)->np.ndarray:
        Cov_matrix=cls.Covariance_matrix_from_sub_sample(Fourier_minous= Fourier_M,Fourier_plus = Fourier_P,L=L,Circulant=Circulant)
        return np.linalg.svd(Cov_matrix,compute_uv=Complete)


    @classmethod
    def Compute_Entropy_State(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,n_size:np.int64=100,step:np.int64=2,Circulant:bool=False)->np.ndarray:
        """
        This function computes the Entropy of a given state being this the reason why the Fourier plus and the minous band have
        to be passed as a parameters. by default we compute the entropy for a size of 2 up to 100 and therfore we return
        an array.
        """
        S = [np.sum(cls.Binary_entropy(0.5-cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=i,Circulant=Circulant,Complete = False))) for i in range(2,n_size,step)]
        return np.array(S)


    @classmethod
    def Compute_Density_Matrix_Random_State(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64,Circulant:bool=False)->np.ndarray:
        """
        This function returns the  density matrix from a random state, this is why we need to pass the Fourier plus and minous
        to this function, this does not compute the fourier transform, only the density matrix associated with it.
        """
        O_1, S, O_2 = cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=Circulant,Complete = True)
        S = -S +0.5
        x= np.log(1-S) - np.log(S)
        M = -(O_1@np.diag(x)@O_2)/cls.beta
        return M


    @classmethod
    def Compute_Spectrum_Random_Distribution_Associated(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64,Circulant:bool=False)->np.ndarray:
        S = cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=Circulant,Complete = False)#np.linalg.svd(cls.Covariance_matrix_from_sub_sample_Toeplitz(Fourier_plus= Fourier_P,L=L,Circulant=Circulant),compute_uv=False)
        n=np.arange(-(L-1)/2,(L-1)/2 +1)
        S=sorted(-S+0.5,reverse=True)
        Fermi = sorted(cls.Sample_Fermi_dirac(n=n,Size=L),reverse=True)
        return np.array(S),np.array(Fermi)

    @classmethod
    def Compute_Fourier_Transforms(cls,Ground:bool = False,Save:bool=False,Route:str = "./",Cluster:bool = False):
        Fourier_minous = np.zeros((cls.num_data,cls.N_size))
        Fourier_plus = np.zeros((cls.num_data,cls.N_size))
        for i in range(cls.num_data):
            a,b = cls.Get_Bands_Matrix(Ground=Ground,Cluster=Cluster)
            Fourier_minous[i,:]=a.real
            Fourier_plus[i,:]=b.real
        if Save:
            with open(Route + 'Fourier_plus.pkl','wb') as f:
                pickle.dump(Fourier_plus, f)
                f.close()
            with open(Route + 'Fourier_minous.pkl','wb') as f:
                pickle.dump(Fourier_minous, f)
                f.close()
        else:
            return Fourier_minous,Fourier_plus


    @classmethod
    def Simple_Fourier_Transform(cls,num:np.int64,rank:np.int64 = 0,Ground:bool = False,Cluster:bool = False) ->np.ndarray:
        """
        This was done specially to use the pool function to use a multiple thread programing
        """
        np.random.seed(rank*cls.num_data+num)
        Data = np.zeros((cls.N_size,2))
        a,b = cls.Get_Bands_Matrix(Ground=Ground,Cluster=Cluster)
        Data[:,0] = a.real
        Data[:,1] = b.real
        return Data
    
    @classmethod
    def Fourier_Parallel_Transform(cls,Ground = False,Threads:np.int64 = 3,rank:np.int64=0 , Cluster:bool = False) ->np.ndarray:
        with Pool(Threads) as p:
            Fourier_Function = partial(cls.Simple_Fourier_Transform,Ground=Ground,Cluster=Cluster,rank=rank)
            Fourier_Transforms = np.array(p.map(Fourier_Function,range(cls.num_data)))
        return Fourier_Transforms



    @classmethod
    def Compute_Participation_Function(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64,Circulant:bool = False)->np.ndarray:
        O_1,S,O_2 = cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=Circulant,Complete = True)
        return (O_1**2 + O_2.T**2)*0.5


class Plot_XY_Computations(Computations_XY_model):
    @classmethod
    def Plot_Modes(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64,Circulant:bool = False,Save:bool = False,title:str = "Normal modes",Both:bool=False)->np.ndarray:
        fig,axes = plt.subplots(nrows=L, ncols=2, figsize=(15,int(2*L)), tight_layout=True)
        if Both:
            O_1,S,O_2 =cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=True,Complete = True)
            O_1_toe,S_toe,O_2_toe =cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=False,Complete = True)#cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M, Fourier_P=Fourier_P, L=L,Circulant=False)
            for i in range(L):
                axes[i,0].set_title(r"Mode number {} of $O_1$".format(i+1))
                axes[i,0].plot(O_1[:,i],color="navy",label="Circulant")
                axes[i,0].plot(O_1_toe[:,i],color="forestgreen",label = "Toeplitz")
                axes[i,1].set_title(r"Mode number {} of $O_2$".format(i+1))
                axes[i,1].plot(O_2[i,:],color="firebrick",label="Circulant")
                axes[i,1].plot(O_2_toe[i,:],color="rebeccapurple",label = "Toeplitz")
                axes[i,0].legend()
                axes[i,1].legend()
            if Save:
                fig.savefig(title+".png")
                plt.close()
            else:
                return fig, axes
        else:
            O_1,S,O_2 =cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=Circulant,Complete = True)
            for i in range(L):
                axes[i,0].set_title(r"Mode number {} of $O_1$".format(i+1))
                axes[i,0].plot(O_1[:,i],color="navy")
                axes[i,1].set_title(r"Mode number {} of $O_2$".format(i+1))
                axes[i,1].plot(O_2[i,:],color="firebrick")
            if Save:
                fig.savefig(title+".png")
                plt.close()
            else:
                return fig, axes


    @classmethod
    def Plot_Participation_Function(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64,Circulant = False,Save:bool = False,title:str = "Participation Function",Both:bool=False)->np.ndarray:
        fig,axes = plt.subplots(nrows=L, ncols=2, figsize=(15,int(2*L)), tight_layout=True)
        if Both:
            P=cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=True,Complete = True)
            P_toe=cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=False,Complete = True)
            for i in range(L):
                axes[i,0].set_title(r"Participation Function Column {}".format(i+1))
                axes[i,0].plot(P[:,i],color="navy",label="Circulant")
                axes[i,0].plot(P_toe[:,i],color="forestgreen",label="Toeplitz")
                axes[i,1].set_title(r"Participation Function rows {}".format(i+1))
                axes[i,1].plot(P[i,:],color="firebrick",label = "Circulant")
                axes[i,1].plot(P_toe[:,i],color="rebeccapurple",label="Toeplitz")
                axes[i,0].legend()
                axes[i,1].legend()
            if Save:
                fig.savefig(title+".png")
                plt.close()
            else:
                return fig, axes

        else:
            P=cls.Compute_svd_Cov_Matrix(Fourier_M=Fourier_M,Fourier_P=Fourier_P,L=L,Circulant=Circulant,Complete = True)
            for i in range(L):
                axes[i,0].set_title(r"Participation Function Column {}".format(i+1))
                axes[i,0].plot(P[:,i],color="navy")
                axes[i,1].set_title(r"Participation Function rows {}".format(i+1))
                axes[i,1].plot(P[i,:],color="firebrick")
            if Save:
                fig.savefig(title+".png")
                plt.close()
            else:
                return fig, axes













# L=30
# State = Sampling_Random_State()
# F_minous,F_plus=State.Get_Bands_Matrix()
# beta = np.min(State.Omega(np.linspace(-np.pi,np.pi,int(1000))))
# New_cov_matrix=State.Covariance_matrix_from_sub_sample(F_plus,F_minous,L)
# S=np.linalg.svd(New_cov_matrix,compute_uv=False)
# n=np.arange(-(L-1)/2,(L-1)/2 +1)
# array_to_plot=sorted(-S+0.5,reverse=True)
# plt.plot(array_to_plot,label="Singular values")
# plt.plot(np.array(sorted(Fermi_dirac(n=n,Size=L,beta=beta),reverse=True)),label="Fermi distribution")
# plt.legend()
# plt.title("lenght of {}".format(L))
# plt.show()
