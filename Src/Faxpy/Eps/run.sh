#!/bin/sh
export NBLOCKS=2048
export NTHREADS=256

BIN=../../../Bin
$BIN/tfaxpy2de >log.txt  

