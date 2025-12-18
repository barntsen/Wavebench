#!/bin/sh

opt=" -ffast-math -O3"
gcc $opt -c ac2dmod.c
gcc $opt -c ac2d.c
gcc $opt -c diff.c
gcc $opt -c model.c
gcc $opt -c rec.c
gcc $opt -c src.c
gcc $opt -c util.c

ar cr libac2d.a ac2d.o model.o rec.o src.o util.o diff.o
gcc -O3 -o ac2dmod-c ac2dmod.o ac2d.o model.o rec.o src.o util.o diff.o -lm
mv ac2dmod-c ../../../Bin
