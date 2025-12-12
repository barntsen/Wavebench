#!/bin/sh
# Test of the diff module

#ec -x cuda -d diff.e
#ec -x cuda -d t1diff.e
#el -o t1diff t1diff.o diff.o

ec -x cuda -d diff.e
ec -x cuda -d t2diff.e
elc -o t2diff t2diff.o diff.o
