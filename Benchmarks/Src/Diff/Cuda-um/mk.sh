#!/bin/sh

nvcc -use_fast_math --compiler-options -O2 --compiler-options -ffast-math  \
     -o t2diff -x cu -O3 t2diff.cpp diff.cpp gpu.cpp util.cpp
