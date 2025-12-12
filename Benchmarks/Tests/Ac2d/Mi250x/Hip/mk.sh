#!/bin/sh
SRC=../../../Src

#Compile ac2dmod library
work=`pwd`
cd $SRC/Ac2d-hip
  ./mk.sh hip
cd $work

#Copy library and source
cp $SRC/Ac2d-hip/libac2d.a .
cp $SRC/Ac2d-hip/ac2dmod.cpp .
cp $SRC/Ac2d-hip/*.h .


#Model no 1
cp ac2dmod.cpp ac2dmod1.cpp
sed s/ac2dmod/ac2dmod1/ run.sh > xaa.sh
sed 's/log.txt/log-1.txt/' xaa.sh > run1.sh
hipcc -c ac2dmod1.cpp
hipcc -o ac2dmod1 ac2dmod1.o libac2d.a 
chmod +x run1.sh

#Model no 2
sed s/Nx=251/Nx=512/ ac2dmod.cpp > xaa.cpp
sed s/Ny=251/Ny=512/ xaa.cpp > ac2dmod2.cpp
sed s/n1=251/n1=512/   run.sh > xaa.sh
sed s/n2=251/n2=512/   xaa.sh > xab.sh
sed 's/log.txt/log-2.txt/' xab.sh > xac.sh
sed s/ac2dmod/ac2dmod2/ xac.sh > run2.sh
hipcc -c ac2dmod2.cpp
hipcc  -o ac2dmod2 ac2dmod2.o libac2d.a 
chmod +x run2.sh

#Model no 3
sed s/Nx=251/Nx=1001/ ac2dmod.cpp > xaa.cpp
sed s/Ny=251/Ny=1001/ xaa.cpp > ac2dmod3.cpp
sed s/n1=251/n1=1001/   run.sh > xaa.sh
sed s/n2=251/n2=1001/   xaa.sh > xab.sh
sed 's/log.txt/log-3.txt/' xab.sh > xac.sh
sed s/ac2dmod/ac2dmod3/ xac.sh > run3.sh
hipcc -c ac2dmod3.cpp
hipcc  -o ac2dmod3 ac2dmod3.o libac2d.a 
chmod +x run3.sh

#Model no 4
sed s/Nx=251/Nx=2001/ ac2dmod.cpp > xaa.cpp
sed s/Ny=251/Ny=2001/ xaa.cpp > ac2dmod4.cpp
sed s/n1=251/n1=2001/   run.sh > xaa.sh
sed s/n2=251/n2=2001/   xaa.sh > xab.sh
sed 's/log.txt/log-4.txt/' xab.sh > xac.sh
sed s/ac2dmod/ac2dmod4/ xac.sh > run4.sh
hipcc -c ac2dmod4.cpp
hipcc  -o ac2dmod4 ac2dmod4.o libac2d.a 
chmod +x run4.sh

#Model no 5
sed s/Nx=251/Nx=4001/ ac2dmod.cpp > xaa.cpp
sed s/Ny=251/Ny=4001/ xaa.cpp > ac2dmod5.cpp
sed s/n1=251/n1=4001/   run.sh > xaa.sh
sed s/n2=251/n2=4001/   xaa.sh > xab.sh
sed 's/log.txt/log-5.txt/' xab.sh > xac.sh
sed s/ac2dmod/ac2dmod5/ xac.sh > run5.sh
hipcc -c ac2dmod5.cpp
hipcc  -o ac2dmod5 ac2dmod5.o libac2d.a 
chmod +x run5.sh

#Model no 6
sed s/Nx=251/Nx=8001/ ac2dmod.cpp > xaa.cpp
sed s/Ny=251/Ny=8001/ xaa.cpp > ac2dmod6.cpp
sed s/n1=251/n1=8001/   run.sh > xaa.sh
sed s/n2=251/n2=8001/   xaa.sh > xab.sh
sed 's/log.txt/log-6.txt/' xab.sh > xac.sh
sed s/ac2dmod/ac2dmod6/ xac.sh > run6.sh
hipcc -c ac2dmod6.cpp
hipcc  -o ac2dmod6 ac2dmod6.o libac2d.a 
chmod +x run6.sh

#Model no 7
sed s/Nx=251/Nx=16001/ ac2dmod.cpp > xaa.cpp
sed s/Ny=251/Ny=16001/ xaa.cpp > ac2dmod7.cpp
sed s/n1=251/n1=16001/   run.sh > xaa.sh
sed s/n2=251/n2=16001/   xaa.sh > xab.sh
sed 's/log.txt/log-7.txt/' xab.sh > xac.sh
sed s/ac2dmod/ac2dmod7/ xac.sh > run7.sh
hipcc -c ac2dmod7.cpp
hipcc  -o ac2dmod7 ac2dmod7.o libac2d.a 
chmod +x run7.sh

rm -f x??.sh
chmod +x *.sh
