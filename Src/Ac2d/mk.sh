#!/bin/sh

cd C
  ./mk.sh
cd ..

cd Cuda
  ./mk.sh
cd ..

cd Cuda-um
  ./mk.sh
cd ..

cd Eps
  ./mk.sh c
  ./mk.sh cuda
  ./mk.sh omp
cd ..
