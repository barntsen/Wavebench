#!/usr/bin/python3
import numpy as np
import pylab as pl

def getdata(file,item,lrec) :
    ''' getdata reads an ascii file with timing records
    
        Parameters:
          file : File name

        Returns :

    '''

    data = np.loadtxt(file)
    n = data.shape[0]
    nrec=int(n/lrec)
    print(nrec)
    x = np.zeros((nrec))
    y = np.zeros((nrec))
    j=0
    for i in range(0,nrec) :
      x[i] = data[j] 
      y[i] = data[j+item] 
      j=j+lrec
    return(x,y)

# Record lenght 
lrec=7
#Get the solver time
item=5

path = '../GH200/Eps/'
file=path+'log-cuda.txt'
x1,y1 = getdata(file,item,lrec)

path = '../A100/Eps/'
file=path+'log-cuda.txt'
x2,y2 = getdata(file,item,lrec)

path = '../RTX4070/Eps/'
file=path+'log-cuda.txt'
x3,y3 = getdata(file,item,lrec)

# Plotting
fig = pl.figure()
#pl.xticks(np.arange(0,3.1,1))
x1=x1*x1
x2=x2*x2
x3=x3*x3
pl.xscale("log", base=10)
#pl.yscale("log", base=10)
l1=pl.plot(x1,y1,label='GH200',color='black',marker='o',linestyle='solid')
l2=pl.plot(x2,y2,label='A100',color='red',marker='o',linestyle='solid')
l3=pl.plot(x3,y3,label='RTX4070',color='blue',marker='o',linestyle='solid')

pl.legend(loc='upper left')
pl.xlabel('Model dimension')
pl.ylabel('Run time (sec)')
pl.title("FD Solver time on GH200/A100/RTX4070")
#pl.ylim(0.1,100.0)
#pl.xlim(0,10000.0)

#ax=pl.gca()
#pl.Axes.set_aspect(ax,0.75)

pl.rc('font',size=15)
pl.gcf().tight_layout(h_pad=0,w_pad=0)
pl.savefig('solvertime-eps-gh200-a100-rtx4070.pdf',bbox_inches='tight')
#pl.show()
