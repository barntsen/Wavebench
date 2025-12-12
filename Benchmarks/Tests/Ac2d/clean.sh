#!/bin/sh
#Script for cleaning intermediate files

cd A100
./clean.sh
cd ..

cd RTX4070
./clean.sh
cd ..

cd i7-14700HX 
./clean.sh
cd ..

cd GH200 
./clean.sh
cd ..

cd Mi250x
./clean.sh
cd ..

cd AMD-EPYC-Turin
./clean.sh
cd ..
