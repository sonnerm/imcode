import numpy as np
import numpy.linalg as la
import scipy.integrate as integrate
def cquad(f,lo,hi):
    imag=integrate.quad(lambda x:f(x).imag,lo,hi)[0]
    real=integrate.quad(lambda x:f(x).real,lo,hi)[0]
    return real+1.0j*imag
def wideband_fermiexp(beta,mu,tmax,nsteps,nsubsteps):
    def rtdens_a(beta,mu,t):
        pass
    def rtdens_b(beta,mu,t):
        pass
    return rtdens_to_fermiexp(rtdens_a,rtdens_b,tmax,nsteps,nsubsteps)

def spectral_density_to_fermiexp(spectral_density,int_min,int_max, beta, mu, tmax, nsteps,nsubsteps):
    def rtdens_a(t):
        def g_a(energy,beta,mu,t):
            return 1./(1+np.exp(beta * (energy - mu))) * np.exp(-1.j* energy *t)
        return 1./(2*np.pi)*cquad(lambda x:spectral_density(x) * g_a(x,beta,mu,t),int_min,int_max)
    def rtdens_b(t):
        def g_b(energy,beta,mu,t):
            return (1./(1+np.exp(beta * (energy - mu))) - 1) * np.exp(-1.j* energy *t)
        return 1./(2*np.pi)*cquad(lambda x:spectral_density(x) * g_b(x,beta,mu,t),int_min,int_max)
    return rtdens_to_fermiexp(rtdens_a,rtdens_b,tmax,nsteps,nsubsteps)

def rtdens_to_fermiexp(rtdens_a,rtdens_b,tmax,nsteps,nsubsteps):
    delta_t=tmax/nsteps/nsubsteps
    B = np.zeros((4*nsteps*nsubsteps,4*nsteps*nsubsteps),dtype=np.complex128)
    for j in range (nsteps*nsubsteps):
        for i in range (j+1,nsteps*nsubsteps):
            
            t = i * delta_t
            t_prime = j * delta_t
            integ_a = rtdens_a(t-t_prime) 
            integ_b = rtdens_b(t-t_prime)
            
            B[4*i,4*j+1] = - np.conj(integ_b) * delta_t**2
            B[4*i,4*j+2] = - np.conj(integ_a) * delta_t**2
            B[4*i+1,4*j] = integ_b * delta_t**2
            B[4*i+1,4*j+3] = integ_a * delta_t**2
            B[4*i+2,4*j] =  integ_b * delta_t**2
            B[4*i+2,4*j+3] = integ_a * delta_t**2
            B[4*i+3,4*j+1] = - np.conj(integ_b) * delta_t**2
            B[4*i+3,4*j+2] = - np.conj(integ_a) * delta_t**2
            

        #for equal time
        integ_a = rtdens_a(0.0)
        integ_b = rtdens_b(0.0)
        
        B[4*j+1,4*j] =  integ_b * delta_t**2
        B[4*j+2,4*j] =  integ_b * delta_t**2
        B[4*j+3,4*j+1] = - np.conj(integ_b) * delta_t**2
        B[4*j+3,4*j+2] = - np.conj(integ_a) * delta_t**2
        
        
        # the plus and minus one here come from the overlap of GMs
        B[4*j+2,4*j] += 1 
        B[4*j+3,4*j+1] -=1 

    B += - B.T#like this, one obtains 2*exponent, needed for Grassmann code.
    #here, the IF is in the folded basis: (in-fw, in-bw, out-fw, out-bw,...) which is needed for correlation matrix

    if nsubsteps > 1: #if physical time steps are subdivided, integrate out the "auxiliary" legs
        #rotate from folded basis into Grassmann basis
        U = np.zeros_like(B)
        for i in range (nsteps*nsubsteps):
            U[4*i, 2*nsteps*nsubsteps - (2*i) -1] = 1
            U[4*i + 1, 2*nsteps*nsubsteps + (2*i)] = 1
            U[4*i + 2, 2*nsteps*nsubsteps - (2*i) -2] = 1
            U[4*i + 3, 2*nsteps*nsubsteps + (2*i) + 1] = 1
        B = U.T @ B @ U
        
        #add intermediate integration measure to integrate out internal legs
        for i in range (2*nsteps):
            for j in range (nsubsteps-1):
                B[2*i*nsubsteps + 1 + 2*j,2*i*nsubsteps+2+ 2*j] += 1  
                B[2*i*nsubsteps+2+ 2*j,2*i*nsubsteps + 1 + 2*j] += -1  

        #select submatrix that contains all intermediate times that are integrated out
        B_sub =  np.zeros((4*nsteps*(nsubsteps-1) , 4*nsteps*(nsubsteps-1)),dtype=np.complex128)
        for i in range (2*nsteps):
            for j in range (2*nsteps):
                B_sub[i*(2*nsubsteps-2):i*(2*nsubsteps-2 )+2*nsubsteps-2,j*(2*nsubsteps-2):j*(2*nsubsteps-2 )+2*nsubsteps-2] = B[2*i*nsubsteps+1:2*(i*nsubsteps + nsubsteps)-1,2*j*nsubsteps+1:2*(j*nsubsteps + nsubsteps)-1]

        #matrix coupling external legs to integrated (internal) legs
        B_coupl =  np.zeros((4*(nsubsteps-1)*nsteps,4*nsteps),dtype=np.complex128)
        for i in range (2*nsteps):
            for j in range (2*nsteps):
                B_coupl[i*(2*nsubsteps-2):i*(2*nsubsteps-2 )+2*nsubsteps-2,2*j] = B[2*i*nsubsteps+1:2*(i*nsubsteps + nsubsteps)-1,2*j*nsubsteps]
                B_coupl[i*(2*nsubsteps-2):i*(2*nsubsteps-2 )+2*nsubsteps-2,2*j+1] = B[2*i*nsubsteps+1:2*(i*nsubsteps + nsubsteps)-1,2*(j+1)*nsubsteps-1]

        #part of matriy that is neither integrated nor coupled to integrated variables
        B_ext = np.zeros((4*nsteps,4*nsteps),dtype=np.complex128)
        for i in range (2*nsteps):
            for j in range (2*nsteps):
                B_ext[2*i,2*j] = B[2*i*nsubsteps,2*j*nsubsteps]
                B_ext[2*i+1,2*j] = B[2*(i+1)*nsubsteps-1,2*j*nsubsteps]
                B_ext[2*i,2*j+1] = B[2*i*nsubsteps,2*(j+1)*nsubsteps-1]
                B_ext[2*i+1,2*j+1] = B[2*(i+1)*nsubsteps-1,2*(j+1)*nsubsteps-1]

        B = B_ext + B_coupl.T @ la.inv(B_sub) @ B_coupl
        U = np.zeros(B.shape)#order in the way specified in pdf for him (forward,backward,forward,backward,...)
        for i in range (B.shape[0] //4):
            U[4*i, B.shape[0] //2 - (2*i) -1] = 1
            U[4*i + 1, B.shape[0] //2 + (2*i)] = 1
            U[4*i + 2, B.shape[0] //2 - (2*i) -2] = 1
            U[4*i + 3, B.shape[0] //2 + (2*i) + 1] = 1
        B = U @ B @ U.T#rotate from Grassmann basis to folded basis: (in-fw, in-bw, out-fw, out-bw,...)
    return B
