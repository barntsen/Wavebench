#!/bin/sh
# mk.sh is a test script for PyAc2d. 

./clean.sh

n1=$1  # Nx
n2=$2  # Ny
nt=$3  # nt
mod=$4

#Create wavelet
ricker -nt $nt -f0 30.0 -t0 0.100 -dt 0.0005 src.bin 

#Create vp
spike -n1 $n1 -n2 $n2 -val 2500.0 vp.bin

#Create rho 
spike -n1 $n1 -n2 $n2 -val 1000.0 rho.bin

#Create Qp 
spike -n1 $n1 -n2 $n2 -val 100000.0 q.bin


#Run modelling
BIN=../../../Src/Ac2d-c
$BIN/ac2dmod $n1 $n2 $nt 

#./snp.sh

