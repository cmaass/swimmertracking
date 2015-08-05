# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:55:14 2015

@author: cmaass
"""

# -*- coding: utf-8 -*-


import os
import numpy as np
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import readtraces as rt

np.set_printoptions(precision=3)
#cdir='/media/Corinna2/datagoe/gunnar/150625/'
#cdir='/media/corinna/Corinna2/datagoe/gunnar/verduennt/'
cdir='/media/Corinna2/datagoe/gunnar/trajectories/010715/'
datadir=cdir+cdir.split('/')[-2]+'-data/'
imdir=cdir+cdir.split('/')[-2]+'/'
m=rt.imStack(imdir)
m.extractCoordsThread(Nthread=3,framelim=(0,72395), blobsize=(5,1200), threshold=80,kernel=1, delete=True, mask=False,channel=0, sphericity=-1.,diskfit=True, crop=[0,0,1024,1024], invert=True)
        
m.CoordtoTraj(coordFile=datadir+'coords.txt', lossmargin=1, lenlim=1, idx=3, maxdist=100, consolidate=datadir+'stackcoord.txt')
##m.CoordtoTraj(tempfile=cdir+'randcoords.txt', lossmargin=1, lenlim=1, maxdist=10, consolidate=False)
#inpdata=np.loadtxt(cdir+'stackcoord.txt')
#stInd=rt.txtheader(cdir+'stackcoord.txt')
m.Coord3D(ztFrameFile=datadir+'frameposition.txt', stackSplitFile=datadir+'turningpoints.txt')
m.CoordtoTraj(coordFile=datadir+'coord3D.txt',lenlim=1,maxdist=1e6,lossmargin=2,dim=3)