#!/bin/sh
# mk.sh is a test script for PyAc2d. 

./clean.sh

nt=1501 
resamp=0 
sresamp=0
nthreads=128

Nx=8192 Ny=8192 nblocks=524288
./run-prof.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads > log-prof.txt

