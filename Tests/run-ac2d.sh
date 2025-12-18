#!/bin/sh
# mk.sh is a script for running test cases of Acoustic wave modeling.

#./clean.sh

n1=$1  # Nx
n2=$2  # Ny
nt=$3  # nt
resamp=$4   #Resampling factor for data
sresamp=$5  #Resampling factor for snapshots
nthreads=$6 #No of gpu threads per blocks
nblocks=$7  #No of gpu blocks
arch=$8


# Path to Bin directory
B=../../Bin
# Create wavelet
$B/ricker -nt $nt -f0 30.0 -t0 0.100 -dt 0.0005 src.bin 

# Create vp
$B/spike -n1 $n1 -n2 $n2 -val 2500.0 vp.bin

# Create rho 
$B/spike -n1 $n1 -n2 $n2 -val 1000.0 rho.bin

# Create Q 
$B/spike -n1 $n1 -n2 $n2 -val 100000.0 q.bin

# Run modelling
export NBLOCKS=$nblocks
export NTHREADS=$nthreads
OMP_NUM_THREADS=8

$B/$arch $n1 $n2 $nt $resamp $sresamp $nthreads $nblocks 

#./snp.sh

