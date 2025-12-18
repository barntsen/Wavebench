# Wavebench - Performance and Portability of Seismic Visco Acoustic wave simulation  


![Run times for GH200. Black line is machine generated cuda,
  red line is hand coded cuda with manual memory management, 
  while blue line is hand coded cuda with unified 
  memory management](solvertime-gh200.png)

## Overview

Wavebench is a set of benchmark programs to measure the
efficiency and portability of machine generated cuda/hip code
relative to hand coded cuda/hip for Visco Acoustic Seismic wave simulation.

Machine generated cuda/hip is made via eps, a small Domain Specific Languge.
The main idea is to use the same source code for different architectures.
Parallelization is handled by a parallel construct (aka fortran doconcurrent)
as in

```
# Faxpy is a simple matrix addition test

def int faxpy2d(float [*,*] a, float [*,*] x, float [*,*] y, float b):
 int i,j
 int nx,ny

 nx=len(x,0)
 ny=len(x,1)

 parallel(i=0:nx,j=0:ny):
   a[i,j] = b*y[i,j]+x[i,j]
```

There are also tests for manual memory management and
unified (automatic) memory management.
The benchmarks contain tests relevant for elastic wave simulations
using the finite difference code.

The source code is in cuda/hip or machine generated cuda/hip by the eps 
[Eps](https://github.com/barntsen/Wavebench) transpiler.

Above is a figure showing the performance of vsico acoustic wave simulation
on GH200.  

## Directories

The file tree is organized as shown below

```
.
├── Bin
│   └── __pycache__
├── Src
│   ├── Ac2d
│   │   ├── C
│   │   ├── Cuda
│   │   ├── Cuda-um
│   │   ├── Eps
│   │   ├── Hip
│   │   └── Hip-um
│   ├── Diff
│   │   ├── Cuda-um
│   │   └── Eps
│   └── Faxpy
│       ├── Cuda-um
│       ├── Eps
│       └── F90
└── Tests
    ├── A100
    ├── AMD-EPYC-Turin
    │   ├── C
    │   └── Eps
    ├── GH200
    ├── i7-14700HX
    │   ├── C
    │   └── Eps
    ├── Mi250x
    │   ├── Eps
    │   ├── Hip
    │   └── Hip-um
    ├── Plots
    └── RTX4070

```
The plot directory contains scripts (and pdfs) for displaying
the results.
