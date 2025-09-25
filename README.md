# Parallel Block LU Factorization with GPU-Accelerated Trailing Update

This repository implements a **parallel block LU factorization** of a dense matrix  
using **distributed-memory parallelism (MPI)** together with a **GPU-accelerated trailing update**.

The algorithm factors a matrix $A$ into lower and upper triangular matrices
$$A = L \cdot U$$
by decomposing the matrix into square blocks and processing one block column at a time.  
The expensive trailing matrix-matrix product
by decomposing the matrix into square blocks and processing one block column at a time.  
The expensive trailing matrix-matrix product
$$A_{11} = A_{11}  - L_{10} * U_{01}$$
is offloaded to an NVIDIA GPU for speedup.

---

## Algorithm Overview

The factorization follows the standard **block LU algorithm**:

1. **Panel factorization (CPU, MPI)**
   Factor the current leading diagonal block
   $A_{00} = L_{00} \cdot U_{00}$.

2. **Block row/column updates (CPU, MPI)**
   Compute
$$U_{01} = L_{00}^{-1} \cdot A_{01}$$
   and
$$L_{10} = A_{10} \cdot U_{00}^{-1}$$

using multi-threaded CPU code across MPI ranks.

3. **Trailing submatrix update (GPU)**
Compute
$$A_{11} = A_{11}  - L_{10} * U_{01}$$
with a custom CUDA kernel (or cuBLAS in future extensions).

4. Repeat steps 1–3 for the next diagonal block until the matrix is fully factorized.

This *blocked* formulation improves cache reuse and makes it possible to
parallelize expensive matrix-matrix products across GPUs.

---

## Implementation Details

| File        | Purpose |
|-------------|---------|
| **LU.cpp**      | Main program. Sets up MPI with Boost.MPI, distributes blocks, performs CPU factorization/updates, prepares device arrays, and calls the GPU kernel for the trailing update. |
| **kernel.h**    | Declares the GPU routine `matrixMult(...)` for host code. |
| **kernel.cu**   | Implements the CUDA kernel performing C = A·B for the trailing update. Uses a 2D grid of thread blocks with shared-memory tiling for performance. |
| **dev_array.h** | RAII C++ wrapper around `cudaMalloc/cudaFree`, with convenient `set()`/`get()` host↔device copy methods. |
| **run.sh**      | Example Slurm batch script for HPC clusters (e.g., NVIDIA NVHPC, HPE/Cray systems). Launches 2 MPI processes with 1 GPU per node. |

### Parallelism
* **MPI** (Boost.MPI) splits work across processes (two ranks in the current code).
* **Threads** (Boost.Thread) parallelize certain CPU-side panel operations.
* **CUDA GPU kernel** handles the computationally heavy trailing update.

### GPU Kernel
The CUDA kernel computes

$$C = A(N \text{by} M)) \cdots B(M \text{by} N)$
in row-major order with shared-memory tiling (`TILE = 32` by default), ensuring
coalesced memory accesses and good occupancy on modern NVIDIA GPUs such as the **RTX 4060 (sm_89)**.

---

## Build Instructions

### Prerequisites
* **NVHPC 25.3 (or newer)** with CUDA 12.x and HPCX MPI
* **Boost** (mpi, serialization, thread, system)
* A recent C++ compiler (C++17 or newer)

### Example: compile host with `mpicxx`, device with `nvcc`, link with `mpicxx`
Adjust paths if your module layout differs.

```bash
module load nvhpc_2025/nvhpc-hpcx-cuda12/25.3
export CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_INC="$CUDA_ROOT/targets/x86_64-linux/include"
export CUDA_LIB="$CUDA_ROOT/targets/x86_64-linux/lib"

# Device code
nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 \
     -c kernel.cu -o kernel.o

# Host code (needs CUDA headers)
mpicxx -O3 -std=c++17 -I"$CUDA_INC" \
       -c LU.cpp -o LU.o

# Link
mpicxx -O3 -std=c++17 -o auto LU.o kernel.o \
       -L"$CUDA_LIB" -lcudart \
       -lboost_mpi -lboost_serialization -lboost_thread -lboost_system \
       -Wl,-rpath,"$CUDA_LIB"
``` 

