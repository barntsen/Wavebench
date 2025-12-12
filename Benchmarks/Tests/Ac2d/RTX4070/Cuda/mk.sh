#!/bin/sh
# mk.sh is a test script for PyAc2d. 

./clean.sh

nt=5001
resamp=0 
sresamp=0
nthreads=128

Nx=256 Ny=256  nblocks=512 
./run.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads > log.txt 

Nx=512 Ny=512 nblocks=2048 
./run.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads >> log.txt

Nx=1024 Ny=1024 nblocks=8192
./run.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads >> log.txt

Nx=2048 Ny=2048 nblocks=32768
./run.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads >> log.txt

Nx=4096 Ny=4096 nblocks=131072
./run.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads >> log.txt

Nx=8192 Ny=8192 nblocks=524288
./run.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads >> log.txt

exit

Nx=16384 Ny=16384 nblocks=2097152
./run.sh $Nx $Ny $nt $resamp $sresamp $nblocks $nthreads >> log.txt
