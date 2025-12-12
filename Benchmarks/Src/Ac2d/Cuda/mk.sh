#!/bin/sh
opt="-O3 --compiler-options -O3 --compiler-options -ffast-math -x cu -use_fast_math "
#arch=native
arch=sm_80

nvcc   $opt -c ac2dmod.cpp
nvcc   $opt -c ac2d.cpp
nvcc   $opt -c diff.cpp
nvcc   $opt -c model.cpp
nvcc   $opt -c -g rec.cpp
nvcc   $opt -c src.cpp
nvcc   $opt -c util.cpp
nvcc   $opt -c gpu.cpp

ar cr libac2d.a ac2d.o diff.o model.o rec.o src.o util.o gpu.o

nvcc -arch=$arch -O2 --compiler-options -O2 -use_fast_math \
     -o ac2dmod ac2dmod.o ac2d.o model.o rec.o src.o util.o diff.o gpu.o -lcudart 
