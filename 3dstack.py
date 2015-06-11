# -*- coding: utf-8 -*-


import os
import numpy as np
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import readtraces as rt



j=60
thick=5
zmin=-3
zmax=18

randwalk,dum,my=rt.rw3d(25*j,0.05)
randwalk[:,2]=15*(randwalk[:,2]-min(randwalk[:,2]))/(max(randwalk[:,2])-min(randwalk[:,2]))
lindata=(np.array([[range(25*j)]*3])*np.array([0.1,0.07,0.005]).reshape(3,-1)).transpose()
zdata=np.array((list(np.linspace(zmin,zmax,j))+list(np.linspace(zmin,zmax,j)[::-1]))*26)
switchframes=np.linspace(0,3120,53)

a=np.polyfit(range(j-2),zdata[:j-2],1)[0]
print 'a', a
sample=randwalk
coords=[[]]
zmin
for i in range(0,25*j-2*j,j):
    pfz=np.polyfit(range(i,i+j),sample[i:i+j,2],1)
    print pfz
    if (i/j)%2==0:
        t=i+int((zmin-pfz[1]-pfz[0]*i)/(pfz[0]-a))
        print t, zdata[t]
    else: 
        t=i+int((zmax-pfz[1]-pfz[0]*i)/(pfz[0]+a))
        print t, zdata[t]
    for ind in range(t-thick/2,t-thick/2+thick):
       coords+=[[ind,sample[ind,0],sample[ind,1]]]
coords=np.array(coords[1:])
    
np.savetxt('/media/cmdata/datagoe/gunnar/150513/zdata.txt',zdata,fmt='%.3f')
np.savetxt('/media/cmdata/datagoe/gunnar/150513/randwalk.txt',randwalk,fmt='%.3f')
dummy=np.array([0]*len(coords[:,0])).reshape((-1,1))
newcoords=np.hstack((coords[:,:1], dummy,dummy+50.,coords[:,1:],dummy,dummy,dummy+1.))

np.savetxt('/media/cmdata/datagoe/gunnar/150513/randcoords.txt',newcoords,fmt='%.3f')


m=rt.nomovie('/media/cmdata/datagoe/gunnar/150513/')
m.CoordtoTraj(tempfile='/media/cmdata/datagoe/gunnar/150513/randcoords.txt', lossmargin=1, lenlim=3, maxdist=10, consolidate='stcktest.txt')
stck=np.loadtxt('stcktest.txt')

fig=pl.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(sample[:,0], sample[:,1], zs=sample[:,2].flatten(), c='r')
ax.scatter(coords[:,1],coords[:,2],zs=zdata[coords[:,0].astype(np.uint32)], c=coords[:,0]/max(coords[:,0]), cmap=cm.jet)
ax.plot(stck[:,3], stck[:,4], zs=zdata[stck[:,0].astype(np.uint32)], c='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

pl.savefig('test.png')
pl.show()




