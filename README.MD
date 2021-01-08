## Codes for calculating observables from Influence matrices
- Dense Exact diagonalization
  Mainly usefull for testing the other approaches; Should support all observables, but limited in time/system size
- Sparse Exact diagonalization
  Benchmark for approximative methods, Usefull in regimes where those approximations don't work well
- Matrix Product states using tenpy
  Efficient approximations near low entanglement points
- Shallow bath
  Calculate the influence matrix from the quantum echo representation of short systems but for long times
