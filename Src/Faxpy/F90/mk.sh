#!/bin/sh


nvfortran -O2 -stdpar=gpu -gpu=ccnative -Minfo=all -c faxpy2d.f90
nvfortran -O2 -stdpar=gpu -gpu=ccnative -Minfo=all -c tfaxpy2d.f90
nvfortran -O2 -stdpar=gpu -gpu=ccnative -o tfaxpy2d tfaxpy2d.o faxpy2d.o
