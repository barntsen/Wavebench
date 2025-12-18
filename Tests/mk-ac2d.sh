#!/bin/sh
# Run tests of the ac2d library

./clean.sh

# Parameters for the modeling
nt=5001      # No of time steps
resamp=0     # Do not save any data
sresamp=0    # Do not save snapshots
nthreads=128 # No of threads for omp
arch=$1      # Name of binary to execute

if test -z $arch ; then
  echo " usage: mk.sh arg "
  echo "        arg is one of c, cuda, or omp"
  exit
fi

Nx=256 Ny=256  nblocks=512 
../run-ac2d.sh $Nx $Ny $nt $resamp $sresamp $nthreads $nblocks $arch > log-$arch.txt 

Nx=512 Ny=512 nblocks=2048 
../run-ac2d.sh $Nx $Ny $nt $resamp $sresamp $nthreads $nblocks $arch >> log-$arch.txt

Nx=1024 Ny=1024 nblocks=8192
../run-ac2d.sh $Nx $Ny $nt $resamp $sresamp $nthreads $nblocks $arch >> log-$arch.txt

Nx=2048 Ny=2048 nblocks=32768
../run-ac2d.sh $Nx $Ny $nt $resamp $sresamp $nthreads $nblocks $arch >> log-$arch.txt

Nx=4096 Ny=4096 nblocks=131072
../run-ac2d.sh $Nx $Ny $nt $resamp $sresamp $nthreads $nblocks $arch >> log-$arch.txt

Nx=8192 Ny=8192 nblocks=524288
../run-ac2d.sh $Nx $Ny $nt $resamp $sresamp $nthreads $nblocks $arch >> log-$arch.txt

Nx=16384 Ny=16384 nblocks=2097152
../run-ac2d.sh $Nx $Ny $nt $resamp $sresamp $nthreads $nblocks $arch >> log-$arch.txt
