

def flops(l):
  diffx = (2*l+1)  # Differentiation in x-direction
  diffy = (2*l+1)  # Differentiation in y-direction
  stress=13
  vx=9
  vy=9
  nflop=2*diffx+2*diffy+vx+vy+stress
  return(nflop)

def mops(l):
  mopsdx=(2*l+1)
  mopsdy=(2*l+1)
  mopsvx=13
  mopsvy=13
  mopsstress=21
  nmops=2*mopsdx+2*mopsdy+mopsvx+mopsvy+mopsstress
  return(nmops)

# GH200 
print("=== GH200 =================")
maxgflop=67000
maxbandw=3888.0
maxintensity=maxgflop/maxbandw

print("Maxgflop:                 ", maxgflop)
print("Maxbandwidth:             ", maxbandw)
print("Max arithmetic intensity: ", maxintensity) 

nx=16384
ny=16384
l=6
nt=5001
t=101.0
nflop=flops(l)
gflops=(nx*ny*nt*nflop)/1.0e+09
gflops=gflops/t
print("gflops/sec: ", gflops)

nmops=mops(l)
intensity=nflop/nmops
print("Arithmetic intensity:",intensity)
# A100
print("=== A100 =================")
maxgflop=19254.0
maxbandw=1350.0
maxintensity=maxgflop/maxbandw

print("Maxgflop:                 ", maxgflop)
print("Maxbandwidth:             ", maxbandw)
print("Max arithmetic intensity: ", maxintensity) 

nx=16384
ny=16384
l=6
nt=5001
t=238.9
nflop=flops(l)
gflops=(nx*ny*nt*nflop)/1.0e+09
gflops=gflops/t
print("gflops/sec: ", gflops)

nmops=mops(l)
intensity=nflop/nmops
print("Arithmetic intensity:",intensity)

# RTX4070
print("=== RTX4070 =================")
maxgflop=21290.0
maxbandw=247.0
maxintensity=maxgflop/maxbandw

print("Maxgflop:                 ", maxgflop)
print("Maxbandwidth:             ", maxbandw)
print("Max arithmetic intensity: ", maxintensity) 

#kernel timings:
stress=83
vx    =50
vy    =50
dyminus=13
dyplus=13
dxminus=12
dxplus=12
tkernel = stress+vx+vy+dyminus+dyplus+dxminus+dxplus
print("tkernel: ", tkernel)

nx=8192
ny=8192
l=6
nt=5001
t=237.0
nflop=flops(l)
gflops=(nx*ny*nt*nflop)/1.0e+09
gflops=gflops/tkernel
print("gflops/sec: ", gflops)

nmops=mops(l)
intensity=nflop/nmops
print("Arithmetic intensity:",intensity)

