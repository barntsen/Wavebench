#!/bin/sh

export NTHREADS=1024
export NBLOCKS=1024

nsys nvprof ./t2diff 
