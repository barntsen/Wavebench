#!/bin/sh
# mk.sh is a test script for PyAc2d. 

#./clean.sh

Nx=256 Ny=256 nt=1501
./run.sh $Nx $Ny $nt  > log-1.txt

Nx=512 Ny=512 nt=1501
./run.sh $Nx $Ny $nt > log-2.txt

Nx=1024 Ny=1024 nt=1501
./run.sh $Nx $Ny $nt > log-3.txt

Nx=2048 Ny=2048 nt=1501
./run.sh $Nx $Ny $nt > log-4.txt

Nx=4096 Ny=4096 nt=1501
./run.sh $Nx $Ny $nt  > log-5.txt

Nx=4096 Ny=4096 nt=1501
./run.sh $Nx $Ny $nt  > log-5.txt

Nx=8192 Ny=8192 nt=1501
./run.sh $Nx $Ny $nt  > log-6.txt
