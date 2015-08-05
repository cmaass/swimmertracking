# -*- coding: utf-8 -*-


import os
import numpy as np
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import readtraces as rt

np.set_printoptions(precision=3)

j=60
thick=5
zmin=-3
zmax=18
cdir='/media/cmdata/datagoe/gunnar/150513/'
gen_rw=True
if gen_rw: 
    randwalk1,dum,my=rt.rw3d(25*j,0.05, start=[-30.,30.,0.])
    randwalk1[:,2]=15*(randwalk1[:,2]-min(randwalk1[:,2]))/(max(randwalk1[:,2])-min(randwalk1[:,2]))
    randwalk2,dum,my=rt.rw3d(25*j,0.05,start=[20.,-20.,0])
    randwalk2[:,2]=15*(randwalk2[:,2]-min(randwalk2[:,2]))/(max(randwalk2[:,2])-min(randwalk2[:,2]))
    
    lindata=(np.array([[range(25*j)]*3])*np.array([0.1,0.07,0.005]).reshape(3,-1)).transpose()
    zdata=np.array((list(np.linspace(zmin,zmax,j))+list(np.linspace(zmin,zmax,j)[::-1]))*26)
    zdata=np.column_stack((zdata,range(len(zdata))))
    stacksplit=np.linspace(0,3120,53)
    
    a=np.polyfit(range(j-2),zdata[:j-2,0],1)[0]
    print 'a', a
    coords=[[]]
    zmin
    for i in range(0,25*j-2*j,j):
        pfz=np.polyfit(range(i,i+j),randwalk1[i:i+j,2],1)
        print pfz
        if (i/j)%2==0:
            t=i+int((zmin-pfz[1]-pfz[0]*i)/(pfz[0]-a))
            print t, zdata[t,0]
        else: 
            t=i+int((zmax-pfz[1]-pfz[0]*i)/(pfz[0]+a))
            print t, zdata[t,0]
        for ind in range(t-thick/2,t-thick/2+thick):
           coords+=[[ind,randwalk1[ind,0],randwalk1[ind,1]]]   
    for i in range(0,25*j-2*j,j):
        pfz=np.polyfit(range(i,i+j),randwalk2[i:i+j,2],1)
        print pfz
        if (i/j)%2==0:
            t=i+int((zmin-pfz[1]-pfz[0]*i)/(pfz[0]-a))
            print t, zdata[t,0]
        else: 
            t=i+int((zmax-pfz[1]-pfz[0]*i)/(pfz[0]+a))
            print t, zdata[t,0]
        for ind in range(t-thick/2,t-thick/2+thick):
           coords+=[[ind,randwalk2[ind,0],randwalk2[ind,1]]]
    coords=np.array(coords[1:])
    coords=coords[np.argsort(coords[:, 0])]
        
    np.savetxt(cdir+'ztdata.txt',zdata,fmt='%.3f', header="z t")
    np.savetxt(cdir+'turnpoints.txt',stacksplit,fmt='%.3f', header="frame")
    np.savetxt(cdir+'randwalk.txt',randwalk,fmt='%.3f')
    dummy=np.array([0]*len(coords[:,0])).reshape((-1,1))
    dummyalt=np.array([0,1]*(len(coords[:,0])/2)).reshape((-1,1))
    newcoords=np.hstack((coords[:,:1], dummyalt,dummy+50.,coords[:,1:],dummy,dummy,dummy+1.))

    np.savetxt(cdir+'randcoords.txt',newcoords,fmt='%.3f', header='frame particle# blobsize x y split_blob? [reserved] sphericity\n')


m=rt.nomovie(cdir+'')
m.CoordtoTraj(coordFile=cdir+'randcoords.txt', lossmargin=1, lenlim=1, idx=3, maxdist=10, consolidate=cdir+'stackcoord.txt')
##m.CoordtoTraj(tempfile=cdir+'randcoords.txt', lossmargin=1, lenlim=1, maxdist=10, consolidate=False)
inpdata=np.loadtxt(cdir+'stackcoord.txt')
stInd=rt.txtheader(cdir+'stackcoord.txt')
#
#
#
#dummy=np.zeros(inpdata.shape[0])-1
#inpdata=np.column_stack((inpdata[:,0],dummy, inpdata[:,[1,4,2,3]], dummy,dummy))
#
#
#begIdx,endIdx,stfIdx,frIdx,zIdx,tIdx=0,0,1,0,6,7
#for i in range(stacksplit.shape[0]):
#    endIdx=begIdx+np.searchsorted(inpdata[begIdx:,frIdx],stacksplit[i])
#    inpdata[begIdx:endIdx,stfIdx]=i
#    begIdx=endIdx
#    
#tDelta=zdata-np.roll(zdata,-1,axis=0)
#frames=inpdata[:,frIdx].astype(np.uint32)
#frDelta=inpdata[:,frIdx]-frames
#frDelta=np.column_stack((frDelta,frDelta))
#inpdata[:,[zIdx,tIdx]]=zdata[frames,:]+tDelta[frames,:]*frDelta
#print inpdata
#
m.Coord3D()
m.CoordtoTraj(coordFile=cdir+'coord3D.txt',lenlim=1,maxdist=1e6,lossmargin=2,dim=3)
#
fig=pl.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(randwalk1[:,0], randwalk1[:,1], zs=randwalk1[:,2].flatten(), c='k')
ax.plot(randwalk2[:,0], randwalk2[:,1], zs=randwalk2[:,2].flatten(), c='k')
#ax.scatter(coords[:,1],coords[:,2],zs=zdata[coords[:,0].astype(np.uint32),0], c=coords[:,0]/max(coords[:,0]), cmap=cm.jet)
ax.scatter(inpdata[:,3],inpdata[:,4],zs=zdata[inpdata[:,0].astype(np.uint32),0], c=inpdata[:,0]/max(inpdata[:,0]), cmap=cm.jet, marker='s')
t1=np.loadtxt(cdir+'trajectory000001.txt')
t2=np.loadtxt(cdir+'trajectory000002.txt')
ax.plot(t1[:,3], t1[:,4], zs=t1[:,5], c='r')
ax.plot(t2[:,3], t2[:,4], zs=t2[:,5], c='g')
#traj=np.loadtxt(cdir+'coord3D.txt')
#td=rt.txtheader(cdir+'coord3D.txt')
#ax.plot(traj[:,td['x']], traj[:,td['y']], zs=traj[:,td['z']], c='k')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#
#pl.savefig('test.png')
#pl.show()


