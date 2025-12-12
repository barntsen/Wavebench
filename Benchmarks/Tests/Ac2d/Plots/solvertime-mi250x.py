#!/usr/bin/python3
import numpy as np
import pylab as pl

def getdata(file) :
    data = np.loadtxt(file)
    n = data.shape[0]
    nrec=int(n/5)
    x = np.zeros((nrec))
    y = np.zeros((nrec))
    j=0
    for i in range(0,nrec) :
      x[i] = data[j] 
      y[i] = data[j+3] 
      j=j+5
    return(x,y)

path = '../Tests/Mi250x/Eps/'
file=path+'log.txt'
x1,y1 = getdata(file)

path = '../Tests/Mi250x/Hip/'
file=path+'log.txt'
x2,y2 = getdata(file)

# Plotting
fig = pl.figure()
#pl.xticks(np.arange(0,3.1,1))
x1=x1*x1
x2=x2*x2
#x3=x3*x3
#x4=x4*x4
#x5=x5*x5
pl.xscale("log", base=10)
pl.yscale("log", base=10)
l1=pl.plot(x1,y1,label='Eps ',color='black',marker='o',linestyle='solid')
l2=pl.plot(x2,y2,label='Cuda',color='red',marker='o',linestyle='solid')

pl.legend(loc='upper left')
pl.xlabel('Model dimension')
pl.ylabel('Run time (sec)')
pl.title("FD simulation Solver time on Mi250x ")
#pl.ylim(0.1,100.0)
#pl.xlim(0,10000.0)

#ax=pl.gca()
#pl.Axes.set_aspect(ax,0.75)

pl.rc('font',size=15)
pl.gcf().tight_layout(h_pad=0,w_pad=0)
pl.savefig('solvertime-mi250x.pdf',bbox_inches='tight')
#pl.show()
