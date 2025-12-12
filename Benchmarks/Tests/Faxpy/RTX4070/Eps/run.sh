#!/bin/sh

# Running faxpy2d test cases

arch=$1
if test -z $arch ; then
  echo " usage: mk.sh arg "
  echo "        arg is one of c, cuda, or omp"
  exit
fi


BIN=../../../../Bin
$BIN/tfaxpy2d$1 > log-$1.txt 

