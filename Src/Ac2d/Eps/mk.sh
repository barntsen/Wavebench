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
  opt=" -O "
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
$cc  $opt $f     src.e
$cc  $opt $f     diff.e
$cc  $opt $f     model.e
$cc  $opt $f     rec.e
$cc  $opt $f     ac2d.e
$cc  $opt $f     ac2dmod.e

B=../../../Bin

ar rcs $lib.a ac2d.o diff.o model.o src.o rec.o
$ld $omp -o ac2dmod-e$1 ac2dmod.o ac2d.o diff.o model.o src.o rec.o
mv ac2dmod-e$1 $B 

