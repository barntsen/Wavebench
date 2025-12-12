#!/bin/sh
# Run tests of the ac2d library

./clean.sh

nt=10 
nt=1501
resamp=0 
sresamp=0
nthreads=128


Nx=8192 Ny=8192 nblocks=524288
./run-prof.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads > log-prof.txt

