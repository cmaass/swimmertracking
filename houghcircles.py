# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
from numpy import *
import readtraces as rt
import os
import sys
from matplotlib import pyplot as pl
from matplotlib import cm
from PIL import Image
%matplotlib inline

# <codecell>

#change these if you're not Corinna!
if os.path.exists('/media/corinna/corinna-data/'): dataroot='/media/corinna/corinna-data/'
if os.path.exists('/media/corinna-data/'): dataroot='/media/corinna-data/'
if os.path.exists('/home/cmaass/goettingen/python/swimmertracking/'): workdir='/home/cmaass/goettingen/python/swimmertracking/'
print dataroot, workdir
if not workdir in sys.path: sys.path.append(workdir)
reload(rt)


def circle_invert(pt, cr, integ=True):
    """Inverts point (inside) at circle cirumference. Used to create mirror clusters for Voronoi construction.
    arguments point: (x,y), circle: (xc,yc,r). returns (x,y) as float"""
    d=sqrt((pt[0]-cr[0])**2+(pt[1]-cr[1])**2) #distance to centre
    scf=2*cr[2]/d-1 #scaling factor 
    newpt=[cr[0]+(pt[0]-cr[0])*scf, cr[1]+(pt[1]-cr[1])*scf]
    if integ: newpt=[int(p) for p in newpt]
    return  newpt

# <codecell>

os.chdir(dataroot+'data/20150107_Many_droplet_system_height_test_6mm_wide_15wtpcTTAB_5uL_50um_Droplets_2x_Olympus_4fps/')
mov=rt.movie('0p5mm_high_2_output.avi')
f=mov.getFrame(0)
circ=cv2.HoughCircles(f[:,:,2],cv2.cv.CV_HOUGH_GRADIENT,3,2000,minRadius=450,maxRadius=600).astype(int).flatten()
print circ
b=f.copy()
cv2.circle(b,(circ[0],circ[1]),circ[2]-10, (0,0,255),5)
pl.imshow(b)
mask=zeros(mov.shape[::-1])
cv2.circle(mask,(circ[0],circ[1]),circ[2]-10,1,-1)

# <markdowncell>


# <codecell>

f=mov.getFrame(3000)
vorim=f.copy()
contim=f.copy()
f=f[:,:,2]
fcopy=cv2.GaussianBlur(f,(51,51),0)
fcopy=fcopy*mask+255*(1-mask)
print amax(fcopy), amin(fcopy)
thresh=(fcopy<180).astype(uint8)

print amax(thresh), amin(thresh)
blobs,contours=rt.extract_blobs(thresh, -1, (10,3000), -1, diskfit=False,returnCont=True, outpSpac=1, spherthresh=10)
# thresh=dstack((thresh,thresh,thresh))
cv2.drawContours(contim,contours,-1,(255,0,0),2)
cv2.circle(contim,(circ[0],circ[1]),circ[2], (0,0,255),5)
pl.imshow(contim)

newpoints=[]
vor=rt.Voronoi(blobs[:,3:5])
dists=sum((vor.vertices-array(circ[:2]))**2,axis=1)-circ[2]**2
extinds=[-1]+(dists>0).nonzero()[0]
for i in range(blobs.shape[0]):
    r=vor.regions[vor.point_region[i]]
    if any(j in extinds for j in r):
        newpoints+=[circle_invert(blobs[i,3:5],circ, integ=True)]
        cv2.circle(contim,(newpoints[-1][0],newpoints[-1][1]),10, (0,180,0),-5)
pts=vstack((blobs[:,3:5],array(newpoints)))
vor=rt.Voronoi(pts)
Image.fromarray(contim).save(workdir+'contim.png')
for i in range(blobs.shape[0]):
    r=vor.regions[vor.point_region[i]]
    col=tuple([int(255*c) for c in cm.jet(i*255/len(vor.points))])[:3]
#     pl.plot(vor.vertices[r,1],vor.vertices[r,0], c=col[:3])
    cv2.polylines(vorim, [(vor.vertices[r]).astype(int32)], True, col[:3], 2)
Image.fromarray(vorim).save(workdir+'vorim.png')
print vorim.shape
print workdir

# <codecell>

r=pl.hist(fcopy.flatten(),bins=255,log=True)

# <codecell>

pl.imshow(vorim)

# <codecell>

print workdir

# <codecell>

g=array(contours[68]).flatten().reshape(-1,2)
pl.plot(g[:,0],g[:,1])

# <codecell>

blobs[:,-1]<10

# <codecell>

len(contours[blobs[:,-1]>10])

# <codecell>

dists=sum((vor.vertices-array(circ[:2]))**2,axis=1)-circ[2]**2
print dists
inds= (dists>0).nonzero()[0]

# <codecell>

print vor.regions
ins=0
out=0
for r in vor.regions: 
    if any(i in r for i in inds): out+=1
    else: ins+=1
print ins, out

# <codecell>


