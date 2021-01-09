## Codes for calculating observables from Influence matrices
### Dense Exact diagonalization
  Mainly usefull for testing the other approaches; Should support all observables, but limited in time/system size
### Free fermions
  If the system is non-interacting one can calculate observables in polynomic time.
### Sparse Exact diagonalization
  Benchmark for approximative methods, Usefull in regimes where those approximations don't work well
### Matrix Product states using tenpy
  Efficient approximations near low entanglement points
### Shallow bath
  Calculate influence matrix elements using the quantum echo representation
  works only for short systems but for long times
