#!/bin/sh

nvcc -o tfaxpy2d -x cu tfaxpy2d.cpp faxpy2d.cpp gpu.cpp util.cpp
