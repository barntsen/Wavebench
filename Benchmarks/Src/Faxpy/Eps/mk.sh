#!/bin/sh
# mk is a script for compiling the py2acd ac2d code

if test -z $1 ; then 
  echo " usage: mk.sh arg "
  echo "        arg is one of c, cuda, omp or hip"
  exit
fi

#Set compiler

echo "** Compiling all code with " $1

if  test $1 = c ; then
  cc=ec
  ld=el
  opt=" -d -O "
elif  test $1 = omp ; then
  cc=ec
  ld=el
  f=-f
  omp=-fopenmp
  opt=" -d -O "
elif test $1 = cuda ; then 
  cc=ec 
  ld=elc
  opt=" -O -d -x cuda -O -y sm_80"
else
    echo "argument is one of eps, cuda, or omp"
    exit
fi

echo "Compiling with" $cc

lib=libac2d
# Compile eps code
$cc  $opt $f     faxpy2d.e
$cc  $opt $f     tfaxpy2d.e

$ld $omp -o tfaxpy2d$1 tfaxpy2d.o faxpy2d.o

BIN=../../../Bin
mv tfaxpy2d$1 $BIN

