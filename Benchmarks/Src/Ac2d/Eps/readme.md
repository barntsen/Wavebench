#  Ac2d - 2D Acoustic Finite-Difference simulation of seismic waves

This directory contains the source files for the Ac2d simulation library.
The library is written in a small Domain Specific Language (DSL)
and converted to either standard c, CUDA or c with OpenMP pragma's
using a source-to-source translator (transpiler). 
The transpiler is named eps and found in a separate github repo: 
[Eps](https://github.com/barntsen/Eps.git)

## Compiling
To compile the code  type 

    mk.sh c

for compiling to c. The executable is called ac2dmodc and runs
on cpu's (single core). 

For multicore type

    mk.sh omp

for compiling using OpenMp. The executable is called ac2dmodomp.

For accelaration on gpu, type

    mk.sh c

for compiling to cuda. 
The executable is called ac2dmodcuda and
can run on NVIDIA gpu's.
    

## List of source files
- mk.sh   : Script for compiling the source code
- clean.sh: Clean script
- ac2d.e  : Solver methods
- diff.e  : Differentiator methods
- model.e : Model methods
- rec.e   : Receiver methods
- src.e   : Source methods
- ac2dmod.e : Example code to for using the library
