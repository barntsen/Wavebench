#!/bin/sh
export NBLOCKS=2048
export NTHREADS=256

BIN=../../../Bin
nsys nvprof $BIN/tfaxpy2de >log-prof.txt  
#nsys stats 

