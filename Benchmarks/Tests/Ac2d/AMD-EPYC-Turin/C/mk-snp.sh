#!/bin/sh
# mk.sh is a test script for PyAc2d. 

./clean.sh

Nx=256 Ny=256 nt=1501 rsnp=10
./run.sh $Nx $Ny $nt $rsnp > log-1.txt 

