# Wavebench - GPU portability benchmark for NVIDA/AMD

## Overview

Wavebench is a set of benchmark programs to measure the
efficiency of machine generated cuda/hip code
relative to hand coded cuda/hip.
There are also tests for manual memory managment and
unified (automatic) memory managment.
The benchmarks contain tests relevant for elastic wave simulations
using the finite difference code.

The source code is in cuda/hip or machine generated cuda/hip by the eps 
[Eps](https://github.com/barntsen/Wavebench) transpiler.

## Directories

The file tree is organized as shown below

```
.
├── Benchmarks
│   ├── Bin
│   │   └── 
│   ├── Src
│   │   ├── Ac2d
│   │   ├── Diff
│   │   └── Faxpy
│   └── Tests
│       ├── Ac2d
│       ├── Diff
│       └── Faxpy
├── Paper
├── Presentations
│   ├── Status-sigma2-2023
│   │   └── Fig
│   └── Status-sigma2-2025
│       └── Figs
└── References
```
The Tests directory contain subdirectories for each
test, which again contains the results for each architecture.
F.ex the Ac2d subdirectory contains:
```
.
├── A100
│   ├── C
│   ├── Cuda
│   ├── Cuda-um
│   └── Eps
├── AMD-EPYC-Turin
│   ├── C
│   └── Eps
├── GH200
│   ├── C
│   ├── Cuda
│   ├── Cuda-um
│   └── Eps
├── i7-14700HX
│   ├── C
│   └── Eps
├── Mi250x
│   ├── Eps
│   ├── Hip
│   └── Hip-um
├── Plots
├── RTX4070
│   ├── C
│   ├── Cuda
│   ├── Cuda-um
│   ├── Eps
│   └── Python-cuda
└── Scripts
```
The plot directory contains scripts (and pdfs) for displaying
the results.
