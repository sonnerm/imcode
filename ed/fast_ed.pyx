import numpy as np
cimport numpy as np
cdef extern int __builtin_popcountll(unsigned long long) nogil
# def sz_sector(unsigned int L,unsigned int sz):
#   cdef unsigned long long D=2**L
#   counter=0
#   cdef np.ndarray[unsigned long long,ndim=1] sec1=np.zeros((D,),dtype=np.uint64)
#   cdef np.ndarray[unsigned long long,ndim=1] sec2=np.zeros((D,),dtype=np.uint64)
#   for i in range(D):
#     if __builtin_popcountll(i)==sz:
#       sec1[counter]=i
#       sec2[i]=counter
#       counter+=1
#   return (L,sec1,sec2)
def get_xxz(L,sz,t,U,W):
    t=t[::-1]
    U=U[::-1]
    W=W[::-1]
    # mat=sp.dok_matrix((len(sector[2]),len(sector[2])))
    D=binom(L,sz)
    pm_mask=0b01
    mp_mask=0b10
    xor_mask=0b11
    cdef np.ndarray[int,ndim=1] row=np.zeros((D,))
    cdef np.ndarray[int,ndim=1] col=np.zeros((?,))
    cdef np.ndarray[complex,ndim=1] val=np.zeros((?,))
    for i in range(D):
        cdiag=0
        v=backward_sz_sector(L,sz,i)
        for p in range(L-1):
            if (v&(pm_mask<<p)==0) != (v&(mp_mask<<p)==0):
                mat[(i,sector[1][v^(xor_mask<<p)])]=t[p]/2
                cdiag-=U[p]/2
            cdiag+=U[p]/4
            cdiag+=((v&(1<<p))==0)*W[p]
        mat[(i,i)]=cdiag+((v&(1<<(sector[0]-1)))==0)*W[sector[0]-1]-sum(W)/2
    return mat

cdef int[4096] BINOMIAL_COEFFICIENTS
# BINOMIAL_COEFFICIENTS[:]={0}
BINOMIAL_COEFFICIENTS[0]=1
for i in range(1,64):
  BINOMIAL_COEFFICIENTS[i*64]=1
  for j in range(1,i):
    BINOMIAL_COEFFICIENTS[i*64+j]=BINOMIAL_COEFFICIENTS[(i-1)*64+j]+BINOMIAL_COEFFICIENTS[(i-1)*64+j-1]
  BINOMIAL_COEFFICIENTS[i*64+i]=1
cdef unsigned int binom(unsigned int u, unsigned int l):
  if (l>u):
    return 0
  return BINOMIAL_COEFFICIENTS[u*64+l]



cpdef unsigned long long forward_sz_sector(int L, int sz, unsigned long long i):
  cdef unsigned long long res=0
  L-=1
  while L >=0:
    if (1&(i>>L)):
      res+=binom(L,sz)
      sz-=1
      L-=1
    else:
      L-=1
    # if L==sz:
    #   return res
  return res


cpdef unsigned long long backward_sz_sector(int L, int sz, unsigned long long i):
  cdef unsigned long long res=0
  i+=1
  while L>0:
    if i > binom(L-1,sz):
      res<<=1
      res+=1
      i-=binom(L-1,sz)
      sz-=1
      L-=1
    else:
      res<<=1
      L-=1
  return res
