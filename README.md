# Parallel_LU
This is a GPU-accelerated C++ code that implements Block LU factorization in parallel. To perfrom parallelzation at the level of threads, I used the C++ Boost.threads and to perform parallelization at the level of application, I used Boost.MPI. I then used CUDA to to write a GPU accelerated matrix multiplication to perform the matrix update in each round of factorization.  
