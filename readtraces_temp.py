from matplotlib import pyplot as plt  #this is Python's main scientific plotting library.
from matplotlib import cm,animation
from mpl_toolkits.mplot3d import Axes3D
import cv2 #computer vision library. interfaces with python over numpy arrays.
import numpy as np
from os.path import splitext, basename, exists, sep #some file path handling
import os
from PIL import Image
from time import time
from datetime import timedelta
from glob import glob
from scipy.misc import imresize
from string import digits
import re
import subprocess
import MySQLdb as mdb
import types
import cPickle
import copy
from multiprocessing import Process

try:
    from scipy.spatial import Voronoi
    from scipy.spatial.qhull import QhullError
    vorflag=True
except ImportError:
    vorflag=False
    print "Voronoi module not found. Update your scipy version to >0.12 if possible. Proceeding without Voronoi tesselation features."

from scipy import optimize,stats

np.set_printoptions(precision=3, suppress=True)
digits = frozenset(digits)

COORDHEADER2D='#frame particle# blobsize x y split_blob? [reserved] sphericity\n'
sp=COORDHEADER2D[1:].split()
cheaderdict2D= {sp[i]:i for i in range(len(sp))}
COORDHEADER3D='#frame stackframe particle# blobsize x y z t\n'
sp=COORDHEADER3D[1:].split()
cheaderdict3D= {sp[i]:i for i in range(len(sp))}
TRAJHEADER2D="#frame particle# blobsize x y\n"
sp=TRAJHEADER2D[1:].split()
theaderdict2D= {sp[i]:i for i in range(len(sp))}
TRAJHEADER3D="#frame stackframe particle# blobsize x y z t\n"#TODO: needs time column and z!!!
sp=TRAJHEADER3D[1:].split()
theaderdict3D= {sp[i]:i for i in range(len(sp))}

ZDATAHEADER="#frame t z\n"
sp=ZDATAHEADER[1:].split()
zheaderdict= {sp[i]:i for i in range(len(sp))}
SPLITHEADER="#frame\n"
sp=SPLITHEADER[1:].split()
sheaderdict= {sp[i]:i for i in range(len(sp))}



class trajectory():
    """Single particle trajectory.
        Attributes:
            data:  lengthX4 array, columns frame #, x, y coordinates, size
            opened: Boolean flag. Marks whether particle is still being tracked.
            maxdist: maximum distance between blobs on consecutive frames to be counted into the same trajectory, in px. Default 
            number: trajectory ID.
            dim: 
        Methods:
            findNeighbour: Finds the nearest neighbour in particle coordinate set from next frame."""

    def __init__(self,data,number, maxdist=-1, dim=2):
        self.data=np.array(data) #initiated by line for first blob
        self.opened=True #flag: trajectory considered lost when no particle closer than max. distance in next frame.
        self.maxdist=maxdist #set maximum distance
        self.number=number #trajectory ID
        self.lossCnt=0
        self.dim=dim

    def findNeighbour(self,nxt, frNum, stFrNum=0, idx=False, lossmargin=10, spacing=1, consolidate=False):
        if frNum==4000: print self.number, self.data.shape
        """Finds the nearest neighbour in particle coordinate set from next frame.
        Accepts numpy array with coordinate data from the following movie frame with blobsize, x and y data.
        Keywords:
            idx (int or dict): integer index of x column assuming column sequence (blobsize, x, y). If that order is not present in the input, pass a tuple/list of indices in order area,x,y or a dictionary as in {'blobsize':2,'x':3,'y':4} (use these exact keys)
            lossmargin (int): if no particle is found inside the trajectory's self.mindist radius, trajectory is linearly extrapolated for this # of frames (helps with 'blinking' blobs). Particle is considered lost beyond that frame range.
            spacing (int): For non consecutive frame numbers if the movie was decimated during analysis. E.g. spacing=5 for frames 0,5,10,15,...
        Extends/extrapolates self.data array with new coordinates if successful, Closes trajectory if not.
        Returns next neighbour numpy array with matching particle removed for speedup and to avoid double counting."""
        
        #sort out what index information is given (OK, this is overkill - and resource intensive):
        #TODO: SCRAP IT ASAP!!!!!
        #note: zIdx and tIdx are always set but not used in case of 2D data. Probably more robust.
        if type(idx) is bool: #scalar, x column index
                 if self.dim==3: szIdxIn,xIdxIn,yIdxIn,zIdxIn,tIdxIn=2,3,4,5,6
                 if self.dim==2: szIdxIn,xIdxIn,yIdxIn,zIdxIn,tIdxIn=2,3,4,False,False                  
        else: #iterable
                szIdxIn,xIdxIn,yIdxIn,zIdxIn,tIdxIn=int(idx[0]),int(idx[1]),int(idx[2]),int(idx[3]),int(idx[4]) 
        if frNum-self.data[-1,0]>(lossmargin+1)*spacing: #if there are frame continuity gaps bigger than the loss tolerance, close trajectory!
            self.opened=False
            return nxt

        if nxt.size>0:
            if self.dim==3 and not consolidate: dist=(self.data[-1,4]-nxt[:,xIdxIn])**2+(self.data[-1,4]-nxt[:,yIdxIn])**2+(self.data[-1,4]-nxt[:,zIdxIn])**2
            else: dist=(self.data[-1,3]-nxt[:,xIdxIn])**2+(self.data[-1,4]-nxt[:,yIdxIn])**2
            m=min(dist)
        else:
            m=self.maxdist+1 #this will lead to trajectory closure
            if self.lossCnt in [0, lossmargin]: print "no particles left in frame %d for trajectory %d, loss cnt. %d"%(frNum, self.number, self.lossCnt)
        if m<self.maxdist:
            ptNum=(dist==m).nonzero()[0][0]
            try:
                if self.dim==2: self.data=np.vstack((self.data,np.array([[frNum,ptNum, nxt[ptNum,szIdxIn], nxt[ptNum,xIdxIn],nxt[ptNum,yIdxIn]]]))) #append new coordinates to trajectory
                if self.dim==3: self.data=np.vstack((self.data,np.array([[frNum,stFrNum, ptNum, nxt[ptNum,szIdxIn], nxt[ptNum,xIdxIn],nxt[ptNum,yIdxIn], nxt[ptNum,zIdxIn], nxt[ptNum,tIdxIn]]]))) #append new coordinates to trajectory

            except IndexError:
                print "SOMETHING WRONG HERE!", self.data.shape, nxt.shape, frNum, self.number #not sure what is.
                self.lossCnt+=1
                if self.lossCnt>lossmargin:
                    self.opened=False #close trajectory, don't remove particle from coordinate array.
                else:
                    predCoord=lin_traj(self.data[-lossmargin:,2],self.data[-lossmargin:,3])
                    if np.isnan(sum(predCoord)): predCoord=self.data[-1][2:2+self.dim]# if extrapolation fails, just keep particle where it is. 
                    if self.dim==2: self.data=np.vstack((self.data,np.array([[frNum, -1, self.data[-1,szIdxIn],predCoord[0], predCoord[1]]])))
                    if self.dim==3: self.data=np.vstack((self.data,np.array([[frNum, stFrNum,-1, self.data[-1,szIdxIn], predCoord[0], predCoord[1], predCoord[2], self.data[-1,-1]]])))
                return nxt
#            except ValueError:
#                print "Value Error in frame %d and trajectory %d. Closing."%(frNum,self.number)
#                self.opened=False #close trajectory, don't remove particle from coordinate array.
#                return nxt
            self.lossCnt=0
            return np.delete(nxt,ptNum,0) #remove particle and return coordinate set.
        else:
            self.lossCnt+=1
            if self.lossCnt>lossmargin:
                self.data=self.data[:-lossmargin,:] #cut extrapolated values
                self.opened=False #close trajectory, don't remove particle from coordinate array.
            else:
                if self.dim==2: predCoord=lin_traj(self.data[-lossmargin:,xIdxIn],self.data[-lossmargin:,yIdxIn])
                if self.dim==3: predCoord=lin_traj(self.data[-lossmargin:,xIdxIn],self.data[-lossmargin:,yIdxIn], self.data[-lossmargin:,zIdxIn])
                if np.isnan(sum(predCoord)): predCoord=self.data[-1][xIdxIn:xIdxIn+self.dim]
                if self.dim==2: self.data=np.vstack((self.data,np.array([[frNum, -1,self.data[-1,szIdxIn],predCoord[0], predCoord[1]]])))
                if self.dim==3: self.data=np.vstack((self.data,np.array([[frNum, stFrNum,-1,self.data[-1,szIdxIn],predCoord[0], predCoord[1], predCoord[2], self.data[-1,-1]]])))
            return nxt




class movie():
    """Class for handling 2D video microscopy data. Additionally requires mplayer in the PATH.
        Argument: video filename.
        Keyword parameters: TTAB concentration and background filename.
        Attributes:
            moviefile: filename (string)
            trajectories: particle trajectory objects (dictionary, keys are particle IDs)
            bg: background. boolean (False if no bg), filename (string), or numpy image array (1 channel greyscale)
            shape: image dimensions. xy-tuple of 2 integers.
            datadir: directory to hold trajectory data, background image, extraction parameters etc. Default: movie filename without extension.
            TTAB: TTAB concentration.
            kernel: diameter of the structuring ellipse used for image dilation/erosion. Value int, or False for no morphological operation.
            threshold: binarisation threshold (int). Applied to greyscale image _after_ bg subtraction and contrast rescaling. Default 128 (mid grey).
            blobsize: minimum/maximum 2d particle sizes for tracking. (int, int) tuple.
            framelim: frame number boundaries for extraction. (int, int) tuple. Default either (0,1e8) or (0, max frame #).
        Methods:
            sqlCoords, extractCoords,testFrame, getFrame, loadBG, getBG, gotoFrame, getFrame, findTrajectories, plotMovie, loadTrajectories, stitchTrajectories, gridAnalysis
    """
    def __init__(self,moviefile, TTAB=-1, bg=''):
        """Initialises movie object. Parameter video filename, keywords TTAB (surfactant concentration), path to extrapolated background file.
        """
        self.typ="Particles"
        self.moviefile=moviefile
        self.trajectories={}
        self.bg=False
        self.datadir=splitext(moviefile)[0]+'-data'+sep
        if bg!='':
            if type(bg).__name__=='ndarray':
                self.bg=bg
                shape=bg.shape[:2]
            if type(bg).__name__=='str':
                try:
                    im=np.array(Image.open(bg))
                    shape=im.shape[:2]
                    if len(im.shape)==3: im=im[:,:,0]
                    self.bg=im
                except: pass
        self.TTAB=TTAB
        if os.name=='posix': #this assumes you installed mplayer! We're also quite possibly doing the mplayer output multiple times. Better safe than sorry. TODO cleanup
            result = subprocess.check_output(['mplayer','-vo','null','-ao','null','-identify','-frames','0',self.moviefile])
        if os.name=='nt': #this assumes you installed mplayer and have the folder in your PATH!
            result = subprocess.check_output(['mplayer.exe','-vo','null','-ao', 'null','-identify','-frames','0',self.moviefile])
        try:
            shape=(int(re.search('(?<=ID_VIDEO_WIDTH=)[0-9]+',result).group()),int(re.search('(?<=ID_VIDEO_HEIGHT=)[0-9]+',result).group()))
            framerate=np.float(re.search('(?<=ID_VIDEO_FPS=)[0-9.]+',result).group())
            frames=int(np.round(np.float(re.search('(?<=ID_LENGTH=)[0-9.]+',result).group())*framerate))
            framelim=(0,frames)
        except:
            shape=(0,0)
            framerate=0.
            frames=0.
            framelim=(0,1e8)
        self.parameters={
            'framerate':framerate, 'sphericity':-1.,'xscale':-1.0,'yscale':-1.0,'zscale':-1.0,#float
            'imsize':shape,'blobsize':(0,30),'crop':[0,0,shape[0],shape[1]], 'framelim':framelim, 'circle':[shape[0]/2, shape[1]/2, int(np.sqrt(shape[0]**2+shape[1**2]))],'BGrange':[128,255],#tuples
            'channel':0, 'blur':1, 'spacing':1, 'struct':1, 'threshold':128, 'frames':frames,'imgspacing':-1,'maxdist':1e4,'lossmargin':10, 'lenlim':1,#ints
            'sizepreview':True, 'invert':False, 'diskfit':False, 'mask':True #bools
        }


    def readParas(self, fname='paras.txt'):
        #self.parameters={}
        with open(self.datadir+fname) as f:
            text=f.read()
        text=text.split('\n')
        for t in text:
            t=t.split(': ')
            if t[0].strip() in ['struct','threshold','frames', 'channel','blur','spacing','imgspacing','maxdist','lossmargin','lenlim']:#integer parameters
                self.parameters[t[0]]=int(float(t[1]))
            if t[0].strip() in ['blobsize','imsize', 'crop','framelim', 'circle', 'BGrange']:#tuple parameters
                tsplit=re.sub('[\s\[\]\(\)]','',t[1]).split(',')
                self.parameters[t[0]]=tuple([int(float(it)) for it in tsplit])
            if t[0].strip() in ['framerate','sphericity','xscale','yscale','zscale']:#float parameters
                self.parameters[t[0]]=float(t[1])
            if t[0].strip() in ['sizepreview','mask','diskfit','invert']:#boolean parameters
                self.parameters[t[0]]=str_to_bool(t[1])
        if self.parameters['struct']>1: self.kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.parameters['struct'],self.parameters['struct']))
        else: self.kernel=False
        
    def saveParas(self,fname='paras.txt'):
        text=''
        for k in self.parameters.keys():
            text+="%s: %s\n"%(k,self.parameters[k])
        text=text[:-1]
        with open(self.datadir+'paras.txt','w') as f: 
            f.write(text)

    def sqlCoords(self,dbname,csvfile):
        """Connects to a SQL database and dumps the particle coordinate data into a table. Logs on as """
        db = mdb.connect(host="localhost", user="cmaass",passwd="swimmers", local_infile=True)
        cur=db.cursor()
        try:
            cur.execute('CREATE DATABASE IF NOT EXISTS %s;'%dbname)
            cur.execute('USE %s;'%dbname)
            cur.execute('DROP TABLE IF EXISTS coords;')
            cur.execute('create table coords(frame INT, id INT, size INT, x DOUBLE, y DOUBLE, usg INT);')
            cur.execute("LOAD DATA LOCAL INFILE '%s' INTO TABLE coords FIELDS TERMINATED BY ' ' LINES TERMINATED BY '\n';"%csvfile)
            print 'done'
        except:
            cur.close()
            db.close()
            raise
        cur.close()
        db.close()

    def extractCoords(self,framelim=False, blobsize=False, threshold=False, kernel=False, delete=False, mask=False, channel=0, sphericity=-1, diskfit=True, blur=1,crop=False, invert=False):
        if not framelim: framelim=self.parameters['framelim']
        if not blobsize: blobsize=self.parameters['blobsize']
        if not threshold: threshold=self.parameters['threshold']
        if type(kernel).__name__!='ndarray': kernel=np.array([1]).astype(np.uint8)
        if type(mask).__name__=='str':
            try:
                im=np.array(Image.open(mask))
                if len(im.shape)==3: im=im[:,:,channel]
                mask=(im>0).astype(float)
            except: mask=False
        tInit=time()
        success=True #VideoCapture read method returns False when running out of frames.
        mov=cv2.VideoCapture(self.moviefile) #open movie (works on both live feed and saved movies)
        framenum=framelim[0]
        if framenum>1: dum,my,p=self.gotoFrame(mov,framenum-1)
        if not exists(self.datadir):
            os.mkdir(self.datadir)
        if type(self.bg).__name__=='ndarray':
            bgtrue=True
            bg=self.bg.astype(float)
            if blur>1: bg=cv2.GaussianBlur(bg,(blur,blur),0)
        else: bgtrue=False
        try: os.remove(self.datadir+'temp')
        except OSError: pass
        dumpfile=open(self.datadir+'temp','a')
        allblobs=np.array([]).reshape(0,8)
        dumpfile.write(COORDHEADER2D)
        counter=0
        while success and framenum<framelim[1]: #loop through frames
            framenum+=1
            if framenum%200==0:
                print 'frame',framenum, 'time', str(timedelta(seconds=time()-tInit)), '# particles', counter #progress marker
                np.savetxt(dumpfile,allblobs,fmt="%.2f")
                allblobs=np.array([]).reshape(0,8)
            success,image=mov.read()
            if success:
                im=image[:,:,channel].astype(float)
                if blur>1: im=cv2.GaussianBlur(im,(blur,blur),0)
                if bgtrue:
                    im-=bg
                if type(mask).__name__=='ndarray':
                    im=im*mask
                im=mxContr(im) #TODO: this might be a few rescalings too many. try to make this simpler, but make it work first
                thresh=mxContr((im<threshold).astype(int))
                if type(kernel).__name__=='ndarray': 
                    thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                if invert: 
                    thresh=255-thresh
                if np.amax(thresh)!=np.amin(thresh): blobs=extract_blobs(thresh,framenum,sphericity=sphericity,blobsize=blobsize,diskfit=diskfit)
                else: blobs=np.array([]).reshape(0,8)
                counter=blobs.shape[0]
                try: allblobs=np.vstack((allblobs,blobs))
                except ValueError:
                    pass
                    #print "Value Error!", allblobs.shape, blobs.shape
        np.savetxt(dumpfile,allblobs,fmt="%.2f")
        dumpfile.close()
        with open(self.datadir+'temp','r') as f: tempdata=f.read()[:-1]
        with open(self.datadir+'temp','w') as f: f.write(tempdata)
            
    def getMeanSizes(self,coordfile=False,ROI=False, glyph=False, Nszbins=10, Nfrbins=100):
        if not coordfile: 
            coordfile=self.datadir+'coords.txt'
        if not ROI:
            ROI=self.parameters['crop']
        if os.path.exists(coordfile):
            data=np.loadtxt(coordfile,skiprows=1)
            if glyph: 
                comment,area,frame='','A', 'image'
            else: comment,area,frame='#','blobsize', 'frame'
            header=txtheader(coordfile, comment=comment)
            xInd, yInd,szInd, frInd=header['x'],header['y'],header[area], header[frame]
            mask=np.all(np.vstack((data[:,xInd]>ROI[0],data[:,xInd]<ROI[2],data[:,yInd]>ROI[1],data[:,yInd<[3]])),axis=0)
            data=data[mask,:][:,[frInd,szInd]]
            frspace=np.append(np.linspace(min(data[:,0]),max(data[:,0]),Nfrbins),np.array([np.max(data[:,0])+1]))
            frbins=np.searchsorted(data[:,0],frspace)
            frSzAr=[]
            for i in range(Nfrbins):
                subdata=data[frbins[i]:frbins[i+1],:]
                print subdata.shape
                if subdata.size>0:
                    szbins=np.linspace(min(subdata[:,1]),max(subdata[:,1]),Nszbins+1)
                    dg=np.digitize(subdata[:,1],szbins)
                    bc=np.bincount(dg)
                    maxbin=(bc==max(bc)).nonzero()[0][0]
                    frSzAr+=[list(np.mean(subdata[dg==maxbin,:],axis=0))]
            return np.array(frSzAr)
        else:
            print "please provide valid coordinate file path."
            return np.array([])

    def getGlyphTraj(self,glyphfile):
       try: self.readParas()
       except:
           print "no preexisting parameter file. Saving default values."
       if not os.path.exists(self.datadir): 
           os.mkdir(self.datadir)
       data=np.loadtxt(glyphfile,skiprows=1) 
       header=txtheader(glyphfile,comment='')
       frInd,ptInd,xInd,yInd,szInd=header['image'],header['shape'],header['x'],header['y'],header['A']
       self.parameters['frames']=max(data[:,frInd])
       if self.parameters['framelim'][1]>self.parameters['frames']:
           self.parameters['framelim']=(0,self.parameters['frames'])
       self.saveParas()
       shapes=list(set(data[:,ptInd]))
       for s in shapes:
           trajFname=self.datadir+'glyph_trajectory%06d.txt'%s
           if os.path.exists(trajFname):
               os.remove(trajFname)
           with open(trajFname,'a') as trajfile:
               trajfile.write(TRAJHEADER2D)
               subdata=data[data[:,ptInd]==s,:][:,[frInd,ptInd,szInd,xInd,yInd]]
               np.savetxt(trajfile,subdata,fmt='%.03f')
            
    def getMeanSpeeds(self, glyph=False, mask='trajectory??????.txt',smoothlen=5, Nfrbins=100,Nspbins=10):
        smoothlen=int(smoothlen)
        if smoothlen%2==0: smoothlen+=1
        if glyph:
            self.getGlyphTraj(glyph)
            files=glob(self.datadir+'glyph_trajectory??????.txt')
        else:
            files=glob(self.datadir+mask)
        self.readParas()
        if self.parameters['frames']==0: 
            try: 
                data=np.loadtxt(self.datadir+'coords.txt')
                header=txtheader(self.datadir+'coords.txt')
                frmax=int(data[-1,header['frames']])
            except: 
                frmax=1e5
        else: frmax=self.parameters['frames']            
        speedlist=[[] for i in range(Nfrbins+1)]
        framelist=[[] for i in range(Nfrbins+1)]
        speedspace=np.linspace(0,frmax+1,Nfrbins+1)
        frSpAr=[]
        for f in files:
            data=np.loadtxt(f)
            if data.shape[0]>5*smoothlen: #we'll only count trajectories of reasonable size. Everything else is probably junk anyway
                header=txtheader(f)
                frInd,xInd,yInd=header['frame'],header['x'],header['y']
                xsm=smooth(data[:,xInd],window_len=smoothlen)[smoothlen/2:-smoothlen/2]
                ysm=smooth(data[:,yInd],window_len=smoothlen)[smoothlen/2:-smoothlen/2]
                speeds=np.vstack((xsm,ysm)).transpose()
                speeds=np.sqrt(np.sum((speeds-np.roll(speeds,-1,axis=0))**2,axis=1)[:-1])
                speedbins=np.searchsorted(speedspace,data[:-2,frInd])
                for i in set(speedbins):
                    speedlist[i]+=list(speeds[speedbins==i])
                    framelist[i]+=list(data[speedbins==i,frInd])
        for i in range(Nfrbins):
            spbins=np.linspace(min(speedlist[i]),max(speedlist[i]),Nspbins+1)
            dg=np.digitize(speedlist[i],spbins)
            bc=np.bincount(dg)
            maxbin=(bc==max(bc)).nonzero()[0][0]
            mask=(dg==maxbin)
            frSpAr+=[[np.mean(np.array(framelist[i])[mask]),np.mean(np.array(speedlist[i])[mask])]]
        return np.array(frSpAr)
                    
            
            
    def getFrame(self,framenum):
        """Retrieves frame of number framenum from open movie. Returns numpy array image, or False if unsuccessful.
        Due to movie search/keyframe issues, framenumber might not be exact."""
        mov=cv2.VideoCapture(self.moviefile)
        if framenum>1: s,r,p=self.gotoFrame(mov,framenum-1)
        success,image=mov.read()
        if success: return image
        else: return False

    def loadBG(self, filename=''):
        if filename=="": filename=self.datadir+'bg.png'
        self.bg=np.array(Image.open(filename))

    def getBGold(self, num=50, spac=50, prerun=1000, cutoff=100, save=False, channel=0):
        mov=cv2.VideoCapture(self.moviefile)
        #loop through to start frame
        if prerun>1:
            s,r,p=self.gotoFrame(mov,prerun-1)
            print "target frame reached..."
        else: s,r=mov.read()
        tInit=time()
        print "Extracting images..."
        bgs=r
        print r.shape
        if len(bgs.shape)==3:
            rgb=True
            bgs=bgs[:,:,channel]
        else:rgb=False
        for i in range(num*spac):
            success,r=mov.read()
            #we decimate by 'spac' to get more even statistics. It would be nice if opencv had a way to skipframes!
            if i%spac==0 and success:
                if rgb: r=r[:,:,channel]
                bgs=np.dstack((bgs,r.astype(int)))
        print "Elapsed time %.1f seconds.\n Averaging..."%(time()-tInit), "shape: ", bgs.shape
            # initialise averaged images
        bg=np.empty(bgs.shape[:2])
        for i in range(bgs.shape[0]):
            for j in range(bgs.shape[1]):
                #for each pixel (don't know how to do this more elegantly than loops over x and y):
                #treshold to filter out dark (moving) structures
                #assign most common (argmax) pixel colour to averaged image (this could be more sophisticated, but I don't really care)
                f=bgs[i,j,:]
                g=f[f>cutoff]
                if len(g)>0: bg.itemset((i,j),np.argmax(np.bincount(g.astype(int))))
                # if this is a dark image area anyway, just take the pixel average and hope it'soutside the ROI (like the cell walls)
                else: bg.itemset((i,j), np.mean(f))
        print "Elapsed time %.1f seconds.\n Done."%(time()-tInit)
        mov.release()
        if not exists(self.datadir):
            os.mkdir(self.datadir)
        if save: Image.fromarray(bg.astype(np.uint8)).save(self.datadir+'bg.png')
        self.bg=bg
        return bg

    def getBG(self, num=50, spac=50, prerun=1000, rng=100, save=False, channel=0):
        mov=cv2.VideoCapture(self.moviefile)
        #loop through to start frame
        if prerun>1:
            s,r,p=self.gotoFrame(mov,prerun-1)
            print "target frame reached..."
        else: s,r=mov.read()
        tInit=time()
        print "Extracting images..."
        bgs=r
        print r.shape
        if len(bgs.shape)==3:
            rgb=True
            bgs=bgs[:,:,channel]
        else:rgb=False
        for i in range(num*spac):
            success,r=mov.read()
            #we decimate by 'spac' to get more even statistics. It would be nice if opencv had a way to skip frames! (better than gotoFrame, anyway)
            if i%spac==0 and success:
                if rgb: r=r[:,:,channel]
                bgs=np.dstack((bgs,r.astype(int)))
        print "Elapsed time %s.\n Averaging..."%str(timedelta(seconds=time()-tInit)), "shape: ", bgs.shape
        # initialise averaged images
        bg=np.empty(bgs.shape[:2])
        for i in range(bgs.shape[0]):
            for j in range(bgs.shape[1]):
                #for each pixel (don't know how to do this more elegantly than loops over x and y):
                #treshold to filter out dark (moving) structures
                #assign most common (argmax) np.pixel colour to averaged image (this could be more sophisticated, but I don't really care)
                f=bgs[i,j,:]
                if hasattr(rng, "__len__"): g=f[np.logical_and(rng[0]<f, f<rng[1])]
                else: g=f[f>rng]
                if len(g)>0: bg.itemset((i,j),np.argmax(np.bincount(g.astype(int))))
                # if this is a dark image area anyway, just take the pixel average and hope it's outside the ROI (like the cell walls)
                else: bg.itemset((i,j), np.mean(f))
        print "Elapsed time %s.\n Done."%str(timedelta(seconds=time()-tInit))
        mov.release()
        if not exists(self.datadir):
            os.mkdir(self.datadir)
        if save: Image.fromarray(bg.astype(np.uint8)).save(self.datadir+'bg.png')
        self.bg=bg
        return bg

    def gotoFrame(self,mov,position, channel=0):
        positiontoset = position
        pos = -1
        success=True
        mov.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, position)
        while pos < position:
            success, image = mov.read()
            pos = mov.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            print pos
            if pos == position:
                mov.release()
                return success,image,pos
                if pos > position:
                    positiontoset -= 1
                    mov.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, positiontoset)
                    pos = -1
        if success: return success, image,pos
        else: return success, None,-1

    def getFrames(self,position, length=1, channel=0,spacing=1):
        mov=cv2.VideoCapture(self.moviefile)
        if position>0: success,images,p=self.gotoFrame(mov,position)
        else: success,images=mov.read()
        images=images[:,:,channel]
        for i in range(length-1)*spacing:
            s,im=mov.read()
            if i%spacing==0:
                images=np.dstack((images, im[:,:,channel]))
        mov.release()
        return images

    def CoordtoTraj(self, coordFile='coords.txt',lenlim=-1, delete=True, maxdist=-1, lossmargin=-1, spacing=1, idx=3, consolidate=False, dim=2):#TODO Adjust for frame jumps!!!
        """Needs more documentation!!
        Input (only keywords):
            coordFile (default self.datadir+'coords.txt'): path of file containing coordinate data
            lenlim (default -1 i.e. use self.parameters value): minimum length of extracted trajectories (cleans out fragments)
            delete (default True): delete all previously generated trajectory files (always a good idea, as you might end up with an old/new muddle)
            maxdist (default -1 -> self.parameters): maximum squared (!) distance between blobs on consecutive frames to be counted into the same trajectory, in px. See trajectory object.
            lossmargin (default -1, ->self.parameters): See trajectory.findNeighbour method, number of extrapolated coordinates/frames before particle is considered lost.
            spacing (default 1): for decimated movies. See trajectory.findNeighbour method.
            idx (default=3): column index of 'x' data. This keyword is only used if the file has no usable header from which columns can be mapped (see COORDHEADER2D variable and txtheader function) 
            consolidate (default False, change to filename/path string): Used for 3D stacks only. Instead of writing each trajectory into a separate file, put mean of frame, x, y and sum of areas into file (name provided by keyword). 
        Output: no return value. generates either lots of numbered 'trajectory000xx.txt' files or a single new coordinate file in the movie's data directory.
        """
        t0=time()
        if delete:
            for f in glob(self.datadir+'trajectory*.txt'): os.remove(f)
        if coordFile=='temp':coordFile=self.datadir+'temp'
        if coordFile=='coords.txt':coordFile=self.datadir+'coords.txt'
        if maxdist<0: maxdist=self.parameters['maxdist'] #if not set, take parameter file value. SQUARED VALUE!
        if lossmargin<0: lossmargin=self.parameters['lossmargin'] 
        if lenlim<0: lenlim=self.parameters['lenlim'] 
        #the 'consolidate' flag produces an average coordinate/frame number and 3D size (voxel) for each closed trajectory and writes them into a single coordinate file to get z stack tracing. A second tracking run will then generate time tracking. No single trajectory output.
        colheaders=txtheader(coordFile)
        if consolidate: 
        #TODO: for particles at the upper/lower boundaries, this would result in trajectories spanning 2 stacks. FIX THIS! (means we need the reversal points. Consider writing a new CoordtoTraj for the stack data type)
            #test for stackheader and time/z here!
            dim=3 #has to be 3 anyway. 
            if set(['stackframe','z','t']).issubset(set(colheaders.keys())):
                np.set_printoptions(precision=3,suppress=True)
                if type(consolidate) is not str: stckf=open('stackcoord.txt','w')
                else: stckf=open(consolidate,'w')
                stckf.write('#frame stackframe particle# blobsize x y z t\n') #reshuffle here already
                stFrIdx=colheaders['stackframe']
            else:
                print "Please provide coordinate file with stack frame, z and t columns. (i.e. run addZT first)"
                return False
        print """
        maxdist: %f
        lossmargin: %d
        lenlim: %d
        """%(maxdist, lossmargin,lenlim)
        dataArr=np.loadtxt(coordFile)
        #here, extract column headers
        try:
            szIdx,ptIdx,frIdx,xIdx,yIdx=colheaders['blobsize'],colheaders['particle#'],colheaders['frame'],colheaders['x'],colheaders['y']
            try: zIdx,tIdx=colheaders['z'],colheaders['t']
            except KeyError: zIdx,tIdx=False,False#if there's no z do nothing.
            if dim ==3 and 'stackframe' in colheaders.keys() and not consolidate: frIdx=colheaders['stackframe'] #use stack 'frame' instead of frame if this is 3D extrapolated stack data.
        except KeyError:
            if dim==3: szIdx,frIdx, ptIdx,xIdx, yIdx, zIdx,tIdx =idx-1,0,1, idx, idx+1, idx+2,idx+3
            if dim==2: szIdx,frIdx, ptIdx,xIdx, yIdx, zIdx,tIdx =idx-1,0,1, idx, idx+1, False,False
            print "Warning: Couldn't extract column headers from data file.\nAssuming default/keyword values for column indices:\n\t blobsize: %d\n\tframe #: %d\n\tparticle # in frame: %d\n\tx: %d\n\ty %d\n\tz: %d\n\tt: %d"%(szIdx,frIdx,ptIdx,xIdx,yIdx, zIdx, tIdx)            
        trajectorycount=0
        frames=sorted(list(set(dataArr[:,frIdx])))
        #put in frame range here!
        activetrajectories={}
        stackframe,newstackframe=0,0        
        for i in range(1,len(frames)):
            try: arrIdx=np.searchsorted(dataArr[:,frIdx], frames[i])
            except IndexError: raise
            blobs,dataArr=np.split(dataArr, [arrIdx])
            if consolidate: newstackframe=blobs[0,stFrIdx]
#            print i, arrIdx, blobs
            if frames[i]%400==0:
                print "framenum", frames[i], 'remaining data', dataArr.shape, 'active traj.', len(activetrajectories), 'time', time()-t0
            for tr in activetrajectories.values():
                blobs=tr.findNeighbour(blobs, frames[i-1], stFrNum=stackframe, idx=(szIdx,xIdx,yIdx,zIdx,tIdx), lossmargin=lossmargin, consolidate=consolidate) #for each open trajectory, find corresponding particle in circle set
                if not tr.opened: #if a trajectory is closed in the process (no nearest neighbour found), move to closed trajectories.
#                    print "data: \n", tr.data
                    
                    if tr.data.shape[0]>lenlim:
                        if not consolidate: 
                            if dim==2: np.savetxt(self.datadir+'trajectory%06d.txt'%tr.number, tr.data, fmt='%.2f', header=TRAJHEADER2D)
                            if dim==3: np.savetxt(self.datadir+'trajectory%06d.txt'%tr.number, tr.data, fmt='%.2f', header=TRAJHEADER3D)
                            print "closed trajectory %d, length %d max squared dist. %.2f"%(tr.number, tr.data.shape[0],tr.maxdist)
                        else:
                            trmean=list(np.mean(tr.data,axis=0))
                            trmean[2]=tr.data.shape[0]*trmean[2]#TODO: flexible size coordinate!!!
                            stckf.write(" ".join(["%.3f"%j for j in trmean])+'\n')                            
                    del activetrajectories[tr.number]
            for blob in blobs: #if any particles are left in the set, open a new trajectory for each of them
                if dim==2: activetrajectories[trajectorycount]=trajectory(np.array([[frames[i-1],blob[ptIdx],blob[szIdx],blob[xIdx],blob[yIdx]]]),trajectorycount, maxdist=maxdist, dim=2)
                if dim==3: activetrajectories[trajectorycount]=trajectory(np.array([[frames[i-1],stackframe,blob[ptIdx],blob[szIdx],blob[xIdx],blob[yIdx],blob[zIdx],blob[tIdx]]]),trajectorycount, maxdist=maxdist, dim=3)
                trajectorycount+=1
                
            stackframe=newstackframe
            
#        print "trajectories:", len(activetrajectories)
        for tr in activetrajectories.values():
#            print "data: \n", tr.data
            if not consolidate:
                if tr.data.shape[0]>lenlim:
                    np.savetxt(self.datadir+'trajectory%06d.txt'%tr.number, tr.data, fmt='%.2f',  header=TRAJHEADER2D)
                    print "closed trajectory %d, length %d max squared dist. %.2f"%(tr.number, tr.data.shape[0],tr.maxdist)
            else:
                trmean=list(np.mean(tr.data,axis=0))
                trmean[2]=tr.data.shape[0]*trmean[2]
                stckf.write(" ".join(["%.3f"%j for j in trmean])+'\n')
        try: stckf.close()
        except: pass

    def plotMovie(self, outname=None, decim=10,scale=2, crop=[0,0,0,0], mask='trajectory',frate=10, cmap=cm.jet, bounds=(0,1e8), tr=True, channel=0, lenlim=1):
        """crop values: [left, bottom, right, top]"""
        if not outname: outname=self.datadir+basename(self.moviefile)[:-4]+'-traced.avi'
        mov=cv2.VideoCapture(self.moviefile)
        success,image=mov.read()
        if crop[2]==0: crop[2]=-image.shape[0]
        if crop[3]==0: crop[3]=-image.shape[1]
        test=image[:,:,channel]
        image=np.dstack((test,test,test))
        test=test.copy()[crop[0]:-crop[2],crop[1]:-crop[3]]
        print test.shape
        print crop
        size=(int(test.shape[0]/scale),int(test.shape[1]/scale))
        print size
        out=cv2.VideoWriter(outname,cv2.cv.CV_FOURCC('D','I','V','X'),frate,(size[1],size[0]))
        count=0.
        trajectories=[]
        if tr:
            for ob in glob(self.datadir+mask+'*.txt'):
                tr=np.loadtxt(ob)
                #tr[:,0]=np.around(tr[:,0]/tr[0,0])
                if tr.shape[0]>lenlim:
                    try:
                        minNaN=np.min(np.isnan(np.sum(tr[:,2:4], axis=1)).nonzero()[0])
                        print 'NaN', ob, minNaN
                    except:
                        minNaN=tr.shape[0]
                    trajectories+=[tr[:minNaN,:]]
        print '# of trajectories', len(trajectories)
        while success:
            if (bounds[0] <= count <= bounds[1]) and count%decim==0:
                for i in range(len(trajectories)):
                    if trajectories[i][-1,0]<count:
                        pts = trajectories[i][:,3:5].astype(np.int32) #check indices and shape!!!
                        colour=tuple([int(255*r) for r in cmap(np.float(i)/len(trajectories))[:3]])[::-1]
                        #colour=(0,120,0)
                        cv2.polylines(image,[pts],isClosed=False,color=colour,thickness=int(np.round(scale)))
                    else:
                        w=(trajectories[i][:,0]==count).nonzero()[0]
                        if len(w)>0:
                            pts = trajectories[i][:w[0],3:5].astype(np.int32) #check indices and shape!!!
                            colour=tuple([int(255*r) for r in cmap(np.float(i)/len(trajectories))[:3]])[::-1]
                            cv2.polylines(image,[pts],isClosed=False,color=colour,thickness=int(np.round(scale)))
                image=image[crop[0]:-crop[2],crop[1]:-crop[3]]
                outim=imresize(image,1./scale)
                if count%min(self.parameters['framelim'][1]/10,1000)==0: Image.fromarray(outim).save(self.datadir+'testim%06d.png'%count)
                out.write(outim[:,:,::-1])
            success,image=mov.read()
            if success:
                image=image[:,:,channel]
                image=np.dstack((image,image,image))
            count+=1
            if count%min(self.parameters['framelim'][1]/10,1000)==0: print count
            if count > bounds[1]:
                success=False
        Image.fromarray(outim).save(self.datadir+'testim%06d.png'%count)
        mov.release()
        #out.release()

    def loadTrajectories(self, directory=None, mask='trajectory??????.txt'):
        self.trajectories={}
        print directory
        if not directory: directory=self.datadir
        trajectoryfiles=glob(directory+mask)
        print trajectoryfiles
        for tr in trajectoryfiles:
            data=np.loadtxt(tr)
            num=int(''.join(c for c in basename(tr) if c in digits))
            self.trajectories[num]=trajectory(data,num)

    def plotTrajectories(self,outname='plottrajectories.png', ntrajectories=-1, lenlimit=-1, mpl=False, text=False, idx=2, cmap=False):
        f0=self.getFrames(0, 1)
        print 'mpl', mpl
        if mpl:
            f0=f0.astype(float)
            if len(f0.shape)==3: f0=f0[:,:,0]
            plt.imshow(f0, cmap=cm.gray)
        else:
            f0=np.dstack((f0,f0,f0))
        keys=self.trajectories.keys()
        if ntrajectories>0: trmax=ntrajectories
        else: trmax=len(keys)
        for i in range(trmax):
            tr=self.trajectories[keys[i]]
            cl=cm.jet(i/np.float(len(keys)))
            if tr.data.shape[0]>lenlimit and len(tr.data.shape)>1:
                print tr.number, tr.data.shape
                if mpl:
                    plt.plot(tr.data[:,idx],tr.data[:,idx+1],lw=.3, color=cl)
                    if text: plt.text(tr.data[0,idx], tr.data[0,idx+1], str(tr.number), color=cl,fontsize=6)
                else:
                    cv2.polylines(f0,[tr.data[idx:idx+2].astype(np.int32)],isClosed=False,color=cl,thickness=2)
        if mpl:
            plt.axis('off')
            plt.savefig(self.datadir+outname,dpi=600)
            plt.close('all')
        else:
            Image.fromarray(f0).save(self.datadir+outname)


    def plotTrajectory(self, num): #note xy-indices changed! Rerun analysis if trajectories look strange!
        self.loadTrajectories()
        traj=self.trajectories[num]
        try: f0=self.getFrames(int(traj.data[-1,0]), 1)
        except: f0=self.getFrames(int(traj.data[-1,0])-1, 1)
        f0=np.dstack((f0,f0,f0))
        print f0.shape
        cv2.polylines(f0,[traj.data[:,2:4].astype(np.int32)], False, (255,0,0), 2)
        Image.fromarray(f0.astype(np.uint8)).save(self.datadir+"trajPlot%06d-frame%06d.jpg"%(num, traj.data[-1,0]))


    def stitchTrajectories(self, maxdist, maxtime,timelim=50, save=False):#TODO: fix for particle indices!!!
        tInit=time()
        if self.trajectories=={}:
            print "please load trajectories first (self.loadTrajectories())!"
        else:
            trBegEnd=np.array([-1.]*7)
            nums=np.sorted(self.trajectories.keys())
            for n in nums:
                tr=self.trajectories[n]
                trBegEnd=np.vstack((trBegEnd,np.array([tr.number]+list(tr.data[0,:])+list(tr.data[-1,:]))))
            trBegEnd=trBegEnd[1:,:]
            tMin,tMax=min(trBegEnd[:,1]),max(trBegEnd[:,1])
            #remove
            incompl=np.any(trBegEnd[:,[1,4]]!=[tMin,tMax],axis=1)
            trBegEnd=trBegEnd[incompl,:]
            print 'tmin',tMin,', tmax', tMax,'; number of trajectories', len(nums),', of which incomplete', np.sum(incompl)
            precnum,folnum=-1,-1
            while len(trBegEnd)>0 and time()-tInit<timelim:
                #remove first line
                no1st=trBegEnd[1:,:]
                #print trBegEnd[0,0], 'test 0'
                if trBegEnd[0,1]>tMin:
                    thisnum=trBegEnd[0,0]
                    #print 'test 1'
                    delT=trBegEnd[0,1]-no1st[:,4]
                    delRsq= (trBegEnd[0,2]-no1st[:,5])**2+(trBegEnd[0,3]-no1st[:,6])**2
                    #select all that match min < maxtime and dist < maxdist
                    candidates=all([delT>0, delT<maxtime, delRsq<maxdist**2],axis=0)
                    if np.sum(candidates)>0:
                        precnum=int(no1st[candidates,0][delRsq[candidates]==min(delRsq[candidates])][0])
                        precdata=self.trajectories[precnum].data
                    else: precdata=np.zeros((0,3))
                    delT=no1st[:,1]-trBegEnd[0,4]
                    delRsq= (no1st[:,2]-trBegEnd[0,5])**2+(no1st[:,3]-trBegEnd[0,6])**2
                    candidates=all([delT>0, delT<maxtime, delRsq<maxdist**2],axis=0)
                    if np.sum(candidates)>0:
                        folnum=int(no1st[candidates,0][delRsq[candidates]==min(delRsq[candidates])][0])
                        foldata=self.trajectories[folnum].data
                    else: foldata=np.zeros((0,3))
                if precnum+folnum>-2:
                    print 'stitched trajectories %d,%d and %d, remaining trajectories  %d'%(precnum,thisnum,folnum, len(no1st[:,0]))
                    newnum=max(nums)+1
                    nums=np.append(nums,newnum)
                    self.trajectories[newnum]=trajectory(np.vstack((precdata,self.trajectories[thisnum].data,foldata)),newnum)
                    if precnum>0: del self.trajectories[precnum]
                    if folnum>0: del self.trajectories[folnum]
                    del self.trajectories[thisnum]
                    no1st=no1st[no1st[:,0]!=precnum,:]
                    no1st=no1st[no1st[:,0]!=folnum,:]
                trBegEnd=no1st
                precnum,folnum=-1,-1
            if save:
                for tr in self.trajectories.values():
                    np.savetxt(save+'%06d.txt'%(tr.number),tr.data,fmt='%.03f',  header=TRAJHEADER2D)

    def Histogram(self, fnum, fname="temphist.png", channel=0):
        """plots the RGB histogram for frame # fnum. Auxiliary function for remote parameter setting. Replaces HistoWin in parameter GUI."""
        image=self.getFrame(fnum)
        if type(image).__name__=='ndarray':
            if len(image.shape)==3 and image.shape[2]!=3: image=np.dstack((image[:,:,channel],image[:,:,channel],image[:,:,channel]))
            plt.figure(figsize=[6,4])
            for i in range(3):
                plt.hist(image[:,:,i].flatten(), bins=256, log=True, histtype='step',align='mid',color='rgb'[i])
            plt.savefig(fname,dpi=100,format='png')
            plt.close("all")

    def testImages(self,fnum, mask=False,BGrng=(100,255), channel=0):
        """analogue for ImgDisplay in parametergui.py"""
        if type(mask).__name__=='str':
            try:
                im=np.array(Image.open(self.datadir+'mask.png'))
                if len(im.shape)==3: im=im[:,:,channel]
                mask=(im>0).astype(float)
            except: mask=np.zeros(self.movie.shape[::-1])+1.
        else: mask=np.zeros(self.movie.shape[::-1])+1.
        if type(self.bg).__name__!='ndarray':
            if os.path.exists(self.datadir+'bg.png'):
                self.loadBG()
            else:
                self.bg=self.getBG(rng=BGrng, num=50, spac=int(self.parameters['frames']/51), prerun=100, save=True)
        image=self.getFrame(fnum)
        orig=image.copy()
        Image.fromarray(orig.astype(np.uint8)).save(self.datadir+'orig.png')
        if len(image.shape)>2: image=image[:,:,channel]
        bgsub=image.astype(float)-self.bg
        bgsub=mxContr(bgsub)*mask
        Image.fromarray(bgsub.astype(np.uint8)).save(self.datadir+'bgsub.png')
        thresh=mxContr((bgsub<self.parameters['threshold']).astype(int))
        if self.parameters['struct']>0:
            self.kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.parameters['struct'],self.parameters['struct']))
            thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh=thresh*mask
        Image.fromarray(thresh.astype(np.uint8)).save(self.datadir+'thresh.png')
        contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cont=[]
        particles=orig.copy()
        for i in range(len(contours)):
            M=cv2.moments(contours[i])
            area= M['m00']
            if self.parameters['blobsize'][0]< area < self.parameters['blobsize'][1]:
                cont+=[contours[i]]
                cv2.fillPoly(particles,cont,color=(255,120,0))
        Image.fromarray(particles.astype(np.uint8)).save(self.datadir+'particles.png')

    def gridAnalysis(self, rspacing=10, tspacing=10, trajectory=False, corr=100, plot=True):
            #to make things tractable, use curvature spacing
            #make zero array, dims imagedims/spacing, x movielength/timespacing
            #
        t1=time()
        if self.trajectories!={}:
            if not trajectory:
                alldata=np.vstack([tr.data[tspacing/2::tspacing,:] for tr in self.trajectories.values()])
                bins=(np.arange(self.parameters['frames']/tspacing+1)*tspacing,np.arange(self.parameters['imsize'][0]/rspacing+1)*rspacing,np.arange(self.parameters['imsize'][1]/rspacing+1)*rspacing)
                try:
                    hist=np.histogramdd(alldata, bins)[0]
                    tcorr=[]
                    t1=time()
                    for i in range(corr):
                        tcorr+=[np.mean((hist*np.roll(hist,-i, axis=0))[:-i])]
                        if i%20==0: print i,str(timedelta(seconds=time()-t1))
                    np.savez_compressed(self.datadir+'gridanalysis',np.array(tcorr),hist,bins[0],bins[1],bins[2])
                    np.savetxt(self.datadir+'crossing.txt', np.array(tcorr), fmt='%.5e')

                    return np.array(tcorr), hist,bins

                except MemoryError:
                    print "Excessive histogram size! Please increase granularity!"
                    return np.zeros((0)),np.zeros((0,0,0)), (np.zeros((0)),np.zeros((0)),np.zeros((0)))
            else:
                alldata=np.vstack([tr.data[tspacing/2::tspacing,:] for tr in self.trajectories.values() if tr.number != trajectory])#time subset corresponding to trajectory length
                thistrajectory=self.trajectories[trajectory]
                bins=(np.arange(self.parameters['frames']/tspacing+1)*tspacing,np.arange(self.parameters['imsize'][0]/rspacing+1)*rspacing,np.arange(self.parameters['imsize'][1]/rspacing+1)*rspacing)
                try:
                    hist=np.histogramdd(alldata, bins)[0]
                    tcorr=[]
                    trtimes=thistrajectory.data[tspacing/2::tspacing,0]/np.float(tspacing)
                    for i in range(corr):
                        tcorr+=[np.mean((hist*np.roll(hist,-i, axis=0))[:-i])]
                        if i%20==0: print i,str(timedelta(seconds=time()-t1))
                        np.savez_compressed(self.datadir+'gridanalysis',gA[0],gA[1],gA[2][0],gA[2][1],gA[2][2])
                        np.savetxt(self.datadir+'crossing.txt', gA[0], fmt='%.5e')

                        return np.array(tcorr), hist,bins

                except MemoryError:
                    print "Excessive histogram size! Please increase granularity!"
                    return np.zeros((0)),np.zeros((0,0,0)), (np.zeros((0)),np.zeros((0)),np.zeros((0)))

        else:
            print "no trajectories found!"
            return np.zeros((0,0,0))

    def maze(self, ROI1, ROI2, rootdir='', fmask='trajector*.txt', destination='cut', idx=2):
        self.mazetraj={}
        if rootdir=='': rootdir=self.datadir
        fmask=rootdir+fmask
        files=sorted(glob(fmask))
        destination=rootdir+destination
        print destination, os.path.exists(destination)
        if not os.path.exists(destination): os.mkdir(destination)
        tlist=[]
        for f in files:
            data=np.loadtxt(f)
            num=int(''.join(c for c in basename(f) if c in digits))
            try:
                ind_in=np.argmax((data[:,idx]>ROI1[0])*(data[:,idx]<ROI1[2])*(data[:,idx+1]>ROI1[1])*(data[:,idx+1]<ROI1[3]))
                ind_out=np.argmax((data[:,idx]>ROI2[0])*(data[:,idx]<ROI2[2])*(data[:,idx+1]>ROI2[1])*(data[:,idx+1]<ROI2[3]))
                if ind_out*ind_in>0:
                    duration=ind_out-ind_in
                    dist=sum(np.sqrt((np.roll(data[ind_in:ind_out,1],-1)-data[ind_in:ind_out,1])**2+(np.roll(data[ind_in:ind_out,2],-1)-data[ind_in:ind_out,2])**2)[:-1])
                    tlist+=[[data[ind_in,0],duration,dist]]
                    np.savetxt(destination+os.sep+os.path.basename(f)[:-4]+'-maze.txt', data[ind_in:ind_out,:],fmt="%.2f")
                    self.mazetraj[num]=data[ind_in:ind_out,:]
            except:
                raise
                print f, 'error!'
        return np.array(tlist)


class clusterMovie(movie):
    def __init__(self,fname, TTAB=-1, bg=''):
        movie.__init__(self,fname,TTAB=TTAB,bg=bg)
        self.typ="Clusters"
        self.bg=False

    def getClusters(self,thresh=128,gkern=61,clsize=(1,1e5),channel=0,rng=(1,1e8),spacing=100, maskfile='', circ=[0,0,1e4], imgspacing=-1):
        print 'thresh', thresh, 'gkern',gkern, 'clsize', clsize, 'channel', channel, 'rng', rng, 'spacing', spacing, 'mask', maskfile, 'circle',circ

        t0=time()
        if os.path.exists(maskfile):
            mask=np.array(Image.open(maskfile))[:,:,0]
            mask=(mask>0).astype(float)
        else:
            mask=np.zeros(self.movie.shape[::-1])+1.
        mov=cv2.VideoCapture(self.moviefile)
        framenum=rng[0]
        if rng[0]>1:
            success,image,p= self.gotoFrame(mov,rng[0]-1)
        gkern=int(gkern)
        if gkern%2==0:
            print "warning: Gaussian kernel size has to be odd. New size %d."%(gkern+1)
            gkern=gkern+1
        allblobs=np.empty((0,6))
        success=True
        while success and framenum<rng[1]:
            framenum+=1
            if framenum%500==0: print framenum, time()-t0, allblobs.shape
            success,image=mov.read()
            if not success: break
            if framenum%spacing==0:
                if imgspacing!=-1: 
                    vorIm=image.copy()
                    clustIm=image.copy()
                image=image[:,:,channel]
                blurIm=(mxContr(image)*mask+255*(1-mask))
                blurIm=cv2.GaussianBlur(blurIm,(gkern,gkern),0)
                threshIm=mxContr((blurIm<thresh).astype(int))
                if framenum==100:
                    Image.fromarray(threshIm).save(self.datadir+'thresh.png')
                    Image.fromarray(mxContr(blurIm).astype(np.uint8)).save(self.datadir+'blur.png')
                    Image.fromarray(image).save(self.datadir+'orig.png')
                cnt,hier=cv2.findContours(threshIm,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                blobs=np.empty((0,6))
                count=0
                if framenum%(spacing*imgspacing)==0 and imgspacing>0:
                    savecnt=[]
                for c in range(len(cnt)):
                    mnt=cv2.moments(cnt[c])
                    if clsize[0]<mnt['m00']<clsize[1]:
                        count+=1
                        blobs=np.vstack((blobs,np.array([framenum,count,mnt['m00'], mnt['m10']/mnt['m00'], mnt['m01']/mnt['m00'],-1])))
                        if framenum%(spacing*imgspacing)==0 and imgspacing>0:
                            savecnt+=[cnt[c]]
                if vorflag and blobs.shape[0]>1 and circ[0]!=0:
                    try:
                        newpoints=[]
                        vor=Voronoi(blobs[:,3:5])
                        dists=np.sum((vor.vertices-np.array(circ[:2]))**2,axis=1)-circ[2]**2
                        #extinds=[-1]+(dists>0).nonzero()[0]
                        for i in range(blobs.shape[0]):
                            r=vor.regions[vor.point_region[i]]
                            newpoints+=[circle_invert(blobs[i,3:5],circ, integ=True)]
                        pts=np.vstack((blobs[:,3:5],np.array(newpoints)))
                        vor=Voronoi(pts)
                        for i in range(blobs.shape[0]):
                            r=vor.regions[vor.point_region[i]]
                            if -1 not in r:
                                blobs[i,-1]=PolygonArea(vor.vertices[r])
                                if framenum%(spacing*imgspacing)==0 and imgspacing>0:
                                    col=tuple([int(255*c) for c in cm.jet(i*255/blobs.shape[0])])[:3]
                                    cv2.polylines(vorIm, [(vor.vertices[r]).astype(np.int32)], True, col[:3], 2)
                                    cv2.circle(vorIm, (int(blobs[i,3]),int(blobs[i,4])),5,(255,0,0),-1)
                        if framenum%(spacing*imgspacing)==0 and imgspacing>0:
                            cv2.circle(vorIm, (int(circ[0]),int(circ[1])), int(circ[2]),(0,0,255),2)
                            Image.fromarray(vorIm).save(self.datadir+'vorIm%05d.jpg'%framenum)
                    except QhullError:
                        print "Voronoi construction failed!"
                if framenum%(spacing*imgspacing)==0 and imgspacing>0:
                    count = 0
                    for b in range(len(blobs)):
                        cv2.putText(clustIm,str(count), (int(blobs[count,3]),int(blobs[count,4])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
                        count +=1
                        cv2.drawContours(clustIm,[savecnt[b]],-1,(0,255,120),2)
                    Image.fromarray(clustIm).save(self.datadir+'clustIm%05d.jpg'%framenum)
                allblobs=np.vstack((allblobs,blobs))
        np.savetxt(self.datadir+'clusters.txt',allblobs,fmt="%.2f", header="framenum cluster# blobsize x y voronoiarea")
        print 'thresh', thresh, 'gkern',gkern, 'clsize', clsize, 'channel', channel, 'rng', rng, 'spacing', spacing, 'mask', maskfile

        return allblobs


class imStack(movie):
    """
    Stack data from 3D tracking. File structure assumed: parent folder 
    containing two subfolders, one containing the images, the other named 'data',
    containing all evaluation output (the latter is created if necessary).
    """
    def __init__(self,stackdir):
        """
        Argument: the folder containing images (string). 
        Assumes this folder contains _only_ sequentially named images and no other files
        so it will not check for file extensions.
        """
        self.typ="3D stack"
        self.bg=False
        self.stackdir=stackdir
        if self.stackdir[-1]!=os.sep: self.stackdir+=os.sep
        self.parentdir=os.sep.join(self.stackdir.split(os.sep)[:-2])+os.sep
        self.datadir=self.stackdir[:-1]+'-data'+os.sep
        self.stack=sorted(glob(os.path.dirname(self.stackdir)+os.sep+'*.*'))
        self.coordFile=self.datadir+'coords.txt'
        self.ztFile=self.datadir+'frameposition.txt'
        self.turnFile=self.datadir+'turningpoints.txt'
        if len(self.stack)>0:        
            self.offset=int(stringdiff(self.stack[0],self.stack[-1])[0])
        elif os.path.exists(self.ztFile):
            header=txtheader(self.ztFile)
            self.offset=np.loadtxt(self.ztFile)[header['frame#'],0]
        else: self.offset=0
        try:
            im=cv2.imread(self.stack[0],1)
            shape=im.shape[:2]
            framerate=-1
            frames=len(self.stack)
            framelim=(0,frames)
        except:
            shape=(0,0)
            framerate=0.
            frames=0.
            framelim=(0,1e8)
        self.parameters={
        'framerate':framerate, 'sphericity':-1.,'xscale':1.0,'yscale':1.0,'zscale':1.0,#floats
        'blobsize':(0,30),'imsize':shape,'crop':[0,0,shape[0],shape[1]], 'framelim':framelim,'circle':[shape[0]/2, shape[1]/2, int(np.sqrt(shape[0]**2+shape[1**2]))],'BGrange':[128,255],#tuples
            'channel':0, 'blur':1,'spacing':1,'struct':1,'threshold':128, 'frames':frames,  'imgspacing':-1,'maxdist':-1,'lossmargin':10, 'lenlim':1,#ints
            'sizepreview':True, 'invert':False, 'diskfit':False, 'mask':True
            }
            
    def __call__(self,kwargs):
        self.extractCoords(**kwargs)
        
    def getFrame(self,framenum,offset=0):
        """Retrieves frame of number framenum from opened stack. Returns numpy array image, or False if unsuccessful."""
        try:
            image=cv2.imread(self.stack[framenum-offset],1)
            return image
        except:
            return False

    def extractCoords(self,framelim=False, blobsize=False, threshold=False, kernel=False, delete=True, mask=False, channel=False, sphericity=-1, diskfit=True, blur=1,invert=True,crop=False, contours=False, dumpfile=False, ztfile=False): #fix the argument list! it's a total disgrace...
        tInit=time()
        contdict={}
        if not framelim: 
            framelim=self.parameters['framelim']
        mn,mx=framelim[0],framelim[1]
        print framelim 
        if not blobsize: blobsize=self.parameters['blobsize']
        if not threshold: threshold=self.parameters['threshold']
        if not channel: channel=self.parameters['channel']
        if not crop: crop=self.parameters['crop']
        if type(kernel).__name__!='ndarray': kernel=np.array([1]).astype(np.uint8)
        if not exists(self.datadir):
            os.mkdir(self.datadir)
        if dumpfile==False: 
            dumpfile=self.datadir+'coords.txt'
        if delete:
            try: os.remove(dumpfile)
            except: pass        
        print dumpfile
        dumpfile=open(dumpfile,'a')
        if ztfile: 
            dumpfile.write('#frame particle# blobsize x y z t split_blob? sphericity\n')
            ncol=9
            ztdata=np.loadtxt(ztfile)
            header=txtheader(ztfile)
            tIdx=header['timestamp']
            zIdx=header['height']
        else:
            dumpfile.write('#frame particle# blobsize x y split_blob? [reserved] sphericity\n')
            ncol=8
        allblobs=np.array([]).reshape(0,ncol)
        counter=0
        for i in range(mn,mx):
            if i%200==0:
                print 'frame',i, 'time', str(timedelta(seconds=time()-tInit)), '# particles', counter, 'blob array: ', allblobs.shape #progress marker
                np.savetxt(dumpfile,allblobs,fmt="%.2f")
                allblobs=np.array([]).reshape(0,ncol)
            image=self.getFrame(i)
            if type(image).__name__=='ndarray':
                if image.shape[:2]!=(crop[2]-crop[0],crop[3]-crop[1]):
                    if len(image.shape)==2: image=image[crop[0]:crop[2],crop[1]:crop[3]]
                    if len(image.shape)==3: image=image[crop[0]:crop[2],crop[1]:crop[3],:]
                if len(image.shape)>2:
                    image=image[:,:,channel].astype(float)
                #image=mxContr(image) #TODO: this might be a few rescalings too many. try to make this simpler, but make it work first
                thresh=mxContr((image<threshold).astype(int), warning=False)
                if type(kernel).__name__=='ndarray': thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                if invert: thresh=255-thresh
                if np.amin(thresh)!=np.amax(thresh):
                    if contours: blobs,conts=extract_blobs(thresh,i,sphericity=sphericity,blobsize=blobsize,diskfit=diskfit, returnCont=True)
                    else: blobs,conts=extract_blobs(thresh,i,sphericity=sphericity,blobsize=blobsize,diskfit=diskfit, returnCont=False),[]
                else: blobs,conts=np.array([]).reshape(0,ncol),[]
                counter=blobs.shape[0]
                if ztfile and counter>0:
                    #ack ... clean up for frame nums...
                    ind=np.searchsorted(ztdata[:,0],i)
                    times=np.array([ztdata[ind,tIdx]]*counter).transpose()
                    heights=np.array([ztdata[ind,zIdx]]*counter).transpose()
                    blobs=np.hstack((blobs[:,:5], times, heights,blobs[:,5:]))
                try: allblobs=np.vstack((allblobs,blobs))
                except ValueError:
                    pass
                    #print "Value Error!", allblobs.shape, blobs.shape
                for i in range(len(conts)): contdict["%d-%d"%(blobs[0,0],i)]=conts[i]
        np.savetxt(dumpfile,allblobs,fmt="%.2f")
        dumpfile.close()
        with open(dumpfile,'r') as f: tempdata=f.read()[:-1]
        with open(dumpfile,'w') as f: f.write(tempdata)
        if len(contdict)>0:
            with open(self.datadir+'contours.pkl','wb') as f:
                cPickle.dump(contdict,f,cPickle.HIGHEST_PROTOCOL)
    
    def extractCoordsThread(self,Nthread=1,**kwargs):
        threads=[]
#        pool=Pool(processes=Nthread)
#        read out kwargs/movie's frame limit 
        if type(kwargs['framelim']) is bool:
            kwargs['framelim']=[0,self.parameters['frames']]
        spacing=kwargs['framelim'][1]/Nthread
        dumpfiles=[]
        kwdicts=[]
        print Nthread, len(kwargs)
        print 'test'
        for i in range(Nthread):
            kwdicts+=[copy.deepcopy(kwargs)]
            kwdicts[i]['dumpfile']=self.datadir+'coords%03d.txt'%i
            dumpfiles+=[kwdicts[i]['dumpfile']]
            kwdicts[i]['framelim']=[i*spacing,(i+1)*spacing]
            print i, kwdicts[i]
#            pool.apply_async(self,(kwdicts[i],))
            threads.append(Process(target=self, args=(kwdicts[i],)))
            
        for t in threads:
            t.start()
        for t in threads:
            t.join()
#        pool.close()
#        pool.join()
        print dumpfiles
        with open(self.datadir+'coords.txt','wb') as newf:
            for filename in dumpfiles:
                with open(filename,'rb') as hf:
                    newf.write(hf.read()) 
    
    def addZT(self, coordfile=False, ztfile=False, offset=False,outfile=False, stackSplitFile=False):
        if not coordfile: coordfile=self.datadir+'coords.txt'
        if not outfile: outfile=self.datadir+'coords.txt'
        if not ztfile: ztfile=self.datadir+'frameposition.txt'
        if not stackSplitFile: stackSplitFile=self.datadir+'turningpoints.txt'
        if not offset:
            offset=self.offset            
        dHeader=txtheader(coordfile)
        ztHeader=txtheader(ztfile)
        frIdx,pIdx,szIdx,xIdx,yIdx, splIdx, spherIdx=dHeader['frame'], dHeader['particle#'], dHeader['blobsize'], dHeader['x'], dHeader['y'], dHeader['split_blob?'], dHeader['sphericity']
        ztfrIdx,zIdx,tIdx=ztHeader['frame#'],ztHeader['height'],ztHeader['timestamp']
        ztdata=np.loadtxt(ztfile)
        turnHeader=txtheader(stackSplitFile)
        splitIdx=turnHeader['frame#']
        turndata=np.loadtxt(stackSplitFile)[:,splitIdx]
        coorddata=np.loadtxt(coordfile)
        dummy=np.zeros(coorddata.shape[0]) #dummy columns to be filled with stack, time and z info
        coorddata=np.column_stack((coorddata[:,frIdx],dummy,coorddata[:,pIdx],coorddata[:,szIdx],coorddata[:,xIdx],coorddata[:,yIdx],dummy,dummy,coorddata[:,splIdx],coorddata[:,spherIdx]))
        ind0=0        
        for i in range(ztdata.shape[0]):
            ind1=np.searchsorted(coorddata[:,0],ztdata[i,ztfrIdx],side='right')
            stack=np.searchsorted(turndata,ztdata[i,ztfrIdx],side='left')
            coorddata[ind0:ind1,1]=stack #insert stack number
            coorddata[ind0:ind1,-4]=ztdata[i,zIdx] #insert height data
            coorddata[ind0:ind1,-3]=ztdata[i,tIdx] #insert time data
            ind0=ind1
        try: os.remove(outfile)
        except: pass
        with open(outfile,'a') as f:
            f.write('#frame stackframe particle# blobsize x y z t split_blob? sphericity\n')
            np.savetxt(f,coorddata, fmt='%.3f')  
        
        
    def Coord3D(self, coordFile='',ztFrameFile='',stackSplitFile='', outFile=''):
        """Helper function to reshuffle the coordinate data created by the CoordtoTraj method with the consolidate keyword. 
        Input data:
            coordFile containing coordinates with columns as in header: %s
                (CoordtoTraj output)
            zframefile: file containing list of z positions per frame and frame recording time (2 columns)
            stacksplitfile: file containing list of reversal frames for stack splitting locations.
        Output file:
            data format as in %s"""%(TRAJHEADER2D,COORDHEADER3D)
        if coordFile=='': coordFile=self.datadir+'stackcoord.txt'
        if ztFrameFile=='': ztFrameFile=self.datadir+'ztdata.txt'
        if stackSplitFile=='': stackSplitFile=self.datadir+'turnpoints.txt'
        if outFile=='': outFile=self.datadir+'coord3D.txt'
        try: os.remove(outFile)
        except OSError: print "No previous output to delete."
        output= open(outFile, 'a')
        output.write(COORDHEADER3D)
        colheaders=txtheader(coordFile)
        try:
            szIdxIn,frIdxIn,ptIdxIn,xIdxIn,yIdxIn=colheaders['blobsize'],colheaders['frame'],colheaders['particle#'],colheaders['x'],colheaders['y']
        except KeyError:
            szIdxIn,frIdxIn,ptIdxIn,xIdxIn,yIdxIn=theaderdict2D['blobsize'],colheaders['blobsize'],theaderdict2D['particle#'],theaderdict2D['x'],theaderdict2D['y']
            print "Warning: Couldn't extract column headers from data file.\nAssuming default values for column indices:\n\t blobsize: %d\n\tframe #: %d\n\tparticle # in frame: %d\n\tx: %d\n\ty %d"%(szIdxIn,frIdxIn,ptIdxIn,xIdxIn,yIdxIn) 
        frIdxOut=cheaderdict3D['frame']
        inData=np.loadtxt(coordFile)
        dummy=np.zeros(inData.shape[0])-1.
        #OK, this column order is inflexible. Probably overthinking this anyway. Always make sure it reflects COORDHEADER3D
        outData=np.column_stack((inData[:,frIdxIn],dummy, inData[:,[ptIdxIn,szIdxIn,xIdxIn,yIdxIn]], dummy,dummy))
        ztData=np.loadtxt(ztFrameFile)
        dummy=txtheader(ztFrameFile)
        try: zIdxIn,tIdxIn=dummy['height'],dummy['timestamp']
        except KeyError:
            zIdxIn,tIdxIn=0,1
            print "Warning: Couldn't extract column headers from z and t data file.\nAssuming default values for column indices:\n\tz: %d\n\tt %d"%(zIdxIn,tIdxIn) 
        stacksplit=np.loadtxt(stackSplitFile)
        splitHdic=txtheader(stackSplitFile)
        stacksplit=stacksplit[:,splitHdic['frame#']]
        begIdx,endIdx,stfIdxOut=0,0,cheaderdict3D['stackframe']
        for i in range(stacksplit.shape[0]):
            endIdx=begIdx+np.searchsorted(outData[begIdx:,frIdxOut],stacksplit[i])
            outData[begIdx:endIdx,stfIdxOut]=i
            begIdx=endIdx
        ztData=ztData[:,[zIdxIn,tIdxIn]]
        ztDelta=ztData-np.roll(ztData,-1,axis=0)
        frames=outData[:,frIdxIn].astype(np.uint32)
        frDelta=outData[:,frIdxIn]-frames
        frDelta=np.column_stack((frDelta,frDelta))
        outData[:,-2:]=ztData[frames,:]+ztDelta[frames,:]*frDelta#TODO: use  header dictionary? in and out indices?
        np.savetxt(output, outData, fmt='%.3f')
        output.close()

    def animateStack(self,elev=0.3*np.sin(np.linspace(0,np.pi,360)),azi=np.arange(360),output=False, dpi=150, frate=15, bitrate=1200):
        if not output: 
            output=self.datadir+'stack_ani.avi'
        fig=plt.figure()
        axes = Axes3D(fig)
        try:
            if os.path.exists(self.datadir+'coords.txt'): fname=self.datadir+'coords.txt'
            if os.path.exists(self.datadir+'stackcoord.txt'): fname=self.datadir+'stackcoord.txt'
            if os.path.exists(self.datadir+'coord3D.txt'): fname=self.datadir+'coord3D.txt'
            print fname
            header=txtheader(fname)
            if len(header)>0: 
                xInd=header['x']
                yInd=header['y']
                sInd=header['blobsize']
                if 'z' in header.keys(): zInd=header['z']
                else: zInd=header['frame']
                if 't' in header.keys(): tInd=header['t']
                else: tInd=header['frame']
            else: xInd,yInd,zInd,tInd,sInd=3,4,0,0,2
            data=np.loadtxt(fname)
            xsc,ysc,zsc=self.parameters['xscale'],self.parameters['yscale'],self.parameters['zscale']
            xs=data[:,xInd]*xsc
            ys=data[:,yInd]*ysc
            zs=data[:,zInd]*zsc
            ss=data[:,sInd]*xsc/72.
            cs=data[:,tInd]
            cs=(cs-np.min(cs))/(np.max(cs)-np.min(cs))
            print cs
            
            def an_init():
                axes.scatter(xs, ys, zs,s=ss, color=cm.jet(cs))
                trajfiles=glob(self.datadir+'trajectory??????.txt')
                for f in trajfiles:
                    d=np.loadtxt(f)
                    hDic=txtheader(f)
                    try: axes.plot(d[:,hDic['x']],d[:,hDic['y']], zs=d[:,hDic['z']])
                    except KeyError:
                        print hDic
                        print f
                        
            def an_animate(i):
                axes.view_init(elev=elev[i],azim=azi[i])
            
            anim = animation.FuncAnimation(fig, an_animate, init_func=an_init,
                               frames=360, interval=5, blit=True)
            anim.save(output, fps=frate, dpi=dpi, bitrate=bitrate, extra_args=['-vcodec', 'libx264'])
        except:
            print "sorry, plot failed! Is there a coordinate file?"
            raise

    def plotStack(self, noscaling=False, frlim=1e6, fname=None):
        fig=plt.figure()
        axes = Axes3D(fig)
        try:
            if not fname:
                if os.path.exists(self.datadir+'coords.txt'): fname=self.datadir+'coords.txt'
                if os.path.exists(self.datadir+'stackcoord.txt'): fname=self.datadir+'stackcoord.txt'
                if os.path.exists(self.datadir+'coord3D.txt'): fname=self.datadir+'coord3D.txt'
            print fname
            header=txtheader(fname)
            if len(header)>0: 
                xInd=header['x']
                yInd=header['y']
                sInd=header['blobsize']
                if 'z' in header.keys(): zInd=header['z']
                else: zInd=header['frame']
                if 't' in header.keys(): tInd=header['t']
                else: tInd=header['frame']
            else: xInd,yInd,zInd,tInd,sInd=3,4,0,0,2
            data=np.loadtxt(fname)[:frlim,:]
            xsc,ysc,zsc=self.parameters['xscale'],self.parameters['yscale'],self.parameters['zscale']
            xs=data[:,xInd]*xsc
            ys=data[:,yInd]*ysc
            zs=data[:,zInd]*zsc
            ss=data[:,sInd]*xsc/72.
            if noscaling: ss=noscaling
            cs=data[:,tInd]
            cs=(cs-np.min(cs))/(np.max(cs)-np.min(cs))
            print cs
            
            axes.scatter(xs, ys, zs,s=ss, color=cm.jet(cs))
            axes.set_xlabel('x')
            axes.set_ylabel('y')
            axes.set_zlabel('z')
            trajfiles=glob(self.datadir+'trajectory??????.txt')
            for f in trajfiles:
                d=np.loadtxt(f)
                hDic=txtheader(f)
                try: axes.plot(d[:,hDic['x']],d[:,hDic['y']], zs=d[:,hDic['z']])
                except KeyError:
                    print hDic                        
        except:
            print "sorry, plot failed! Is there a coordinate file?"
            raise


    def blenderPrep(self, nfacets=10, smoothlen=5):
        self.loadTrajectories()
        if len(self.trajectories)>0 and os.path.isfile(self.datadir+'contours.pkl'):
            with open(self.datadir+'contours.pkl', 'rb') as f:
                conts=cPickle.load(f)
            todelete=glob(self.datadir+'pointfile*.txt')+glob(self.datadir+'vertfile*.txt')
            for delFile in todelete: os.remove(delFile)
            for j in self.trajectories.keys():
                t1=self.trajectories[j]
                if len(t1.data.shape)==2:
                    keys=[r[0].replace('.','-') for r in t1.data[:,:1].astype(str)]
                    with open(self.datadir+'pointfile%03d.txt'%t1.number, 'a') as pointfile:
                        data=conts[keys[0]].flatten().reshape(-1,2)
                        pointfile.write('%.2f %.2f %.2f \n'%(np.mean(data[:,0]),np.mean(data[:,1]),float(keys[0].split('-')[0])))
                        for i in range(len(keys)):
                            zvals=np.array([int(keys[i].split('-')[0])]*nfacets)
                            data=conts[keys[i]].flatten().reshape(-1,2)
                            x,y=data[:,0],data[:,1]
                            xnew,ynew=smooth(x,smoothlen),smooth(y,smoothlen)
                            inds=list(np.linspace(0,len(xnew)-1,nfacets).astype(int))
                            np.savetxt(pointfile,np.vstack((xnew[inds],ynew[inds],zvals)).T, fmt='%.2f')
                        pointfile.write('%.2f %.2f %.2f'%(np.mean(x),np.mean(y),zvals[0]))

                    verts=[[0,i-1,i,-1] for i in range(2,nfacets+1)]+[[0,nfacets,1,-1]]
                    verts+=[[(j-1)*nfacets+k-1,(j-1)*nfacets+k,j*nfacets+k,j*nfacets+k-1] for j in range(1,len(keys)) for k in range(2,nfacets+1)]
                    verts+=[[(j-1)*nfacets+nfacets, (j-1)*nfacets+1, j*nfacets+1, j*nfacets+nfacets] for j in range(1,len(keys))]
                    verts+=[[nfacets*len(keys)+1,nfacets*len(keys)-i-1,nfacets*len(keys)-i,-1] for i in range(nfacets-1)]+[[nfacets*len(keys)+1,nfacets*len(keys), nfacets*(len(keys)-1)+1,-1]]
                    np.savetxt(self.datadir+'vertfile%03d.txt'%t1.number,np.array(verts),fmt="%d")
                    
        

class nomovie(movie,imStack):    
    def __init__(self,datadir):
        """This is a convenience class to be able to work with extracted data without having to provide the rather large movie data. 
        E.g., while the movie class needs a <somename>.avi (or different extension) file for identification and analysis 
        and generates a <somename>-data directory to store data, the nomovie class just works on the <somename>-data directory, 
        which contains coordinate/trajectory/parameter/image data from previous analyses. 
        Methods requiring movie access have to be adapted with workarounds.
        NOTE: Use this class at your own risk, not all inherited methods have been adapted for the missing file yet.
        """
        self.typ="none"
        self.bg=False
        self.datadir=datadir
        self.parameters={
            'framerate':-1, 'sphericity':-1.,'xscale':1.0,'yscale':1.0,'zscale':1.0,#floats
            'blobsize':(0,30),'imsize':(1920,1080),'crop':[0,0,1920,1080], 'framelim':(0,1e9),'circle':(1000, 500,500),'BGrange':[128,255],#tuples
            'channel':0, 'blur':1,'spacing':1,'struct':1,'threshold':128, 'frames':1e9,  'imgspacing':-1,'maxdist':-1,'lossmargin':10, 'lenlim':1,#ints
            'sizepreview':True, 'invert':False, 'diskfit':False, 'mask':True
                    }
                    
    def getFrames(self,position, length=1, channel=0,spacing=1, imagefile=''):
        """Dummy method to produce a greyscale still image even when there is no movie file provided. Necessary for inheriting plotTrajectories from the movie class. Uses saved background by default.
        Arguments:
           position: frame number in movie. Only used to mimick the inherited behaviour, as we're always using the same image.
        Keywords:
           length: number of (identical) frames returned. Default 1
           channel: RGB channel. Default 0 (red)
           spacing: frame spacing. Again, only for inheritance, doesn't do anything.
           imagefile: use this to provide a specific file name. Uses self.datadir+'bg.png' by default.
        Returns: (xsize X ysize X length) numpy array with image data.
        """
        if imagefile=='': imagefile=self.datadir+'bg.png'
        im=np.array(Image.open(imagefile))
        if len(im.shape)==3: im=im[:,:,channel]
        images=np.dstack([im]*length)
        return images
        
class granMovie(movie):
    """Class for handling stuff that's still not finished from Corinna's PhD work. Argh."""
    def __init__(self,fname, TTAB=-1, bg=''):
        movie.__init__(self,fname,TTAB=TTAB,bg=bg)
        self.typ="Granular"
        self.bg=False
        
    def CircleMask(self,bg=False,threshold=False):
        if not threshold: threshold=self.parameters['threshold']
        if not bg:
            if not self.bg: 
                try: self.readParas()
                except:
                    print "No parameter file found. Using defalut values to generate background."
                bg=self.getBG()
        else: 
            if type(bg).__name__=='str':
                im=np.array(Image.open(bg))
                if len(im.shape)==3: im=im[:,:,0]
                bg=im
            elif type(bg).__name__=='ndarray':
                pass
            else:
                print "Warning, no valid background provided."
                bg=self.getBG()                
        sz=self.parameters['imsize'][0]*self.parameters['imsize'][1]
        circ=extract_blobs(thr,-1,blobsize=(sz/10,sz))[0][[3,4,2]]
    

                    

def extract_blobs(bwImg, framenum, blobsize=(0,1e5), sphericity=-1, diskfit=True, outpSpac=200,returnCont=False, spherthresh=1e5):
    contours, hierarchy=cv2.findContours(bwImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    blobs=[]
    if returnCont: listcont=[]
    for i in range(len(contours)):
        cont=contours[i]
        M=cv2.moments(cont)
        (xm,ym),rm=cv2.minEnclosingCircle(cont)
        area= M['m00'] #adapt for aggregates
        marea=np.pi*rm**2
        if area>0: spher=marea/area
        else:spher=0
        if sphericity>=1 and area>blobsize[0] and spher<spherthresh:
            if spher>sphericity:
                hull = cv2.convexHull(cont,returnPoints = False)
                try:
                    defects = cv2.convexityDefects(cont,hull)
                    farpts=[]
                    for j in range(defects.shape[0]):
                        s,e,f,d = defects[j,0]
                        if d>500: farpts+=[f]
                    defectflag=True
                except AttributeError, IndexError:
                #no proper defects? just try to fit something into this contour
                    if blobsize[0]<area<blobsize[1]:
                        if diskfit: blobs=blobs+[[framenum,0,marea,xm, ym, 0,-1, spher]]
                        else: blobs=blobs+[[framenum,0,area,M['m10']/M['m00'], M['m01']/M['m00'], 0,-1,spher]]
                        if returnCont: listcont+=[cont]
                    defectflag=False
                if defectflag:
                    farpts=np.array(farpts)
                    farpts.sort()
                    if len(farpts)>1:
                        cts=[]
                        for j in range(len(farpts)-1):
                            x=[cont[p][0][0] for p in range(farpts[j],farpts[j+1]+1)]
                            y=[cont[p][0][1] for p in range(farpts[j],farpts[j+1]+1)]
                            try:
                                xc,yc,r,res=leastsq_circle(x,y)
                                cts+=[[xc,yc,np.pi*r**2]]
                            except: pass
                        x=[cont[p][0][0] for p in range(farpts[-1], len(cont))]+[cont[p][0][0] for p in range(farpts[0])]
                        y=[cont[p][0][1] for p in range(farpts[-1], len(cont))]+[cont[p][0][1] for p in range(farpts[0])]
                        try:
                            xc,yc,r,res=leastsq_circle(x,y)
                            cts+=[[xc,yc,np.pi*r**2]]
                        except: pass
                        cts=np.array(cts)
                        inds=np.arange(len(cts[:,0]))
                        newcts=[]
                        while len(inds)>0:
                            this=inds[0]
                            try:
                                inds=inds[1:]
                                if blobsize[0]<cts[this,2]<blobsize[1]:
                                    newcts+=[cts[this]]
                                    distAr=np.pi*((cts[this,0]-cts[inds,0])**2+(cts[this,1]-cts[inds,1])**2)
                                    inds=inds[(distAr-.5*cts[this,2])>0]
                            except IndexError: break
                        for circle in newcts:
                            blobs=blobs+[[framenum,0,circle[2],circle[0], circle[1], 1,-1,spher]]
                            if returnCont: listcont+=[cont]
            elif area<blobsize[1]:
                if diskfit:
                    blobs=blobs+[[framenum,0,marea,xm, ym, 0,-1,spher]]
                    if returnCont: listcont+=[cont]
                else:
                    x,y=M['m10']/M['m00'],M['m01']/M['m00']
                    blobs=blobs+[[framenum,0,area,x,y, 0,-1,spher]]
                    if returnCont: listcont+=[cont]
        elif blobsize[0]<area<blobsize[1] and spher<spherthresh:
            if diskfit:
                blobs=blobs+[[framenum,0,marea,xm, ym, 0, -1,spher]]
                if returnCont: listcont+=[cont]
            else:
                x,y=M['m10']/M['m00'],M['m01']/M['m00']
                blobs=blobs+[[framenum,0,area,x,y, 0,-1,spher]]
                if returnCont: listcont+=[cont]
    blobs=np.array(blobs)
    if len(blobs)>0:
        try: blobs[:,1]=np.arange(blobs.shape[0])
        except IndexError:
            print blobs
    if framenum%outpSpac==0: print "# blobs: ", len(blobs)
    if returnCont:
        return blobs,listcont
    else: return blobs




def msqd(data,length):
    data=data[~np.isnan(data).any(1)]
    return np.array([np.mean(np.sum((data-np.roll(data,i, axis=0))[i:]**2,axis=1)) for i in range(length)])

def mxContr(data,mval=255.,warning=True):
    mx,mn=np.float(np.amax(data)),np.float(np.amin(data))
    if mx!=mn:
        return (mval*(np.array(data)-mn)/(mx-mn)).astype(np.uint8)
    else:
        if warning: print 'Warning, monochrome image!'
        return 0.*np.array(data).astype(np.uint8)

def stitchMovies(mlist, outname=None, decim=10,scale=1, crop=[0,0,0,0], frate=24,channel=-1, invert=False, ims=False, rotation=0):
    """crop values: [left, bottom, right, top]"""
    if not outname: outname=mlist[0]+'-stitched.avi'
    mov=cv2.VideoCapture(mlist[0])
    success,image=mov.read()
    if success:
        for i in range(4): crop[i]=scale*16*(int(crop[i])/(scale*16))
        sh=image.shape
        nsize=(sh[1]-crop[0]-crop[2])/scale,(sh[0]-crop[1]-crop[3])/scale
        bds=[crop[3],sh[0]-crop[1],crop[0],sh[1]-crop[2]]
        print sh,nsize,crop
        out=cv2.VideoWriter(outname,cv2.cv.CV_FOURCC('D','I','V','X'),frate,(nsize[0],nsize[1]))
        mov.release()
        for movie in mlist:
            mov=cv2.VideoCapture(movie)
            success=True
            count=0.
            while success:
                success,image=mov.read()
                if count%decim==0 and not isinstance(image, types.NoneType):
                    if invert: image=255-image
                    if rotation!=0:
                        M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), rotation, 1.)
                        image= cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                    if channel==-1: image=image[bds[0]:bds[1],bds[2]:bds[3],:]
                    else: image=cv2.cvtColor(image[bds[0]:bds[1],bds[2]:bds[3],channel],cv2.cv.CV_GRAY2RGB)
                    outim=imresize(image,1./scale)
                    out.write(outim)
                count+=1
                if count%50==0: print count,outim.shape
            mov.release()

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def rfunc(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(rfunc, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def lin_traj(x,y,z=False):
    yslope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    if type(z) is not bool: zslope, intercept, r_value, p_value, std_err = stats.linregress(x,z)
    mx=np.mean((x-np.roll(x,1))[1:])
    if type(z) is  bool: return np.array([x[-1]+mx, y[-1]+yslope*mx])
    else: return np.array([x[-1]+mx, y[-1]+yslope*mx, z[-1]+zslope*mx])

#http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

#http://stackoverflow.com/questions/21732123/convert-true-false-value-read-from-file-to-boolean
def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("Cannot convert {} to bool".format(s))


    #scipy cookbook http://wiki.scipy.org/Cookbook/SignalSmooth
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def circle_invert(pt, cr, integ=True):
    """Inverts point (inside) at circle cirumference. Used to create mirror clusters for Voronoi construction.
    arguments point: (x,y), circle: (xc,yc,r). returns (x,y) as float"""
    d=np.sqrt((pt[0]-cr[0])**2+(pt[1]-cr[1])**2) #distance to centre
    scf=2*cr[2]/d-1 #scaling factor
    newpt=[cr[0]+(pt[0]-cr[0])*scf, cr[1]+(pt[1]-cr[1])*scf]
    if integ: newpt=[int(p) for p in newpt]
    return  newpt
    
    
def randwalk(lgth,stiff,start=[0.,0.], step=1.,adrift=0.,anoise=.2, dist="const"):
    """generates a 2d random walk with variable angular correlation and optional constant (+noise) angular drift.
    Arguments: walk length "lgth", stiffness factor "stiff" - consecutive orientations are derived by adding stiff*(random number in [-1,1]).
    Parameters "adrift" and "anoise" add a constant drift angle adrift modulated by a noise factor adrift*(random number in [-1,1]).
    The dist parameter accepts non-constant step length distributions (right now, only cauchy/gaussian distributed random variables)"""
    rw=[list(start)] #user provided initial coordinates: make (reasonably) sure it's a nested list type.
    #step lengths are precalculated for each step to account for 
    if dist=="cauchy": steps=stats.cauchy.rvs(size=lgth)
    elif dist=="norm": steps=stats.norm.rvs(size=lgth)
    else: steps=np.array([step]*lgth) #Overkill for constant step length ;)
    #first generate angular progression via cumulative sum of increments (random/stiff + drift terms)
    angs=np.cumsum(stiff*stats.uniform.rvs(size=lgth,loc=-1,scale=2)+adrift*(1.+anoise*stats.uniform.rvs(size=lgth,loc=-1,scale=2)))
    #x/y trace via steplength and angle for each step, some array reshuffling (2d conversion and transposition)
    rw=np.concatenate([np.concatenate([np.array(start[:1]),np.cumsum(steps*np.sin(angs))+start[0]]),np.concatenate([np.array(start)[1:],np.cumsum(steps*np.cos(angs))+start[1]])]).reshape(2,-1).transpose()
    return rw, angs


def rw3d(lgth,stiff,start=[0.,0.,0.], step=1.,adrift=0.,anoise=.2, dist="const"):
    rw=[list(start)] #uscer provided initial coordinates: make (reasonably) sure it's a nested list type.
    #step lengths are precalculated for each step to account for 
    if dist=="cauchy": steps=stats.cauchy.rvs(size=lgth)
    elif dist=="norm": steps=stats.norm.rvs(size=lgth)
    else: steps=np.array([step]*lgth) #Overkill for constant step length ;)fa
    #first generate angular progression via cumulative sum of increments (random/stiff + drift terms)
    thetas=np.cumsum(stiff*stats.uniform.rvs(size=lgth,loc=-1,scale=2)+adrift*(1.+anoise*stats.uniform.rvs(size=lgth,loc=-1,scale=2)))
    phis=np.cumsum(stiff*stats.uniform.rvs(size=lgth,loc=-1,scale=2)+adrift*(1.+anoise*stats.uniform.rvs(size=lgth,loc=-1,scale=2)))
    #x/y trace via steplength and angle for each step, some array reshuffling (2d conversion and transposition)
    rw=np.concatenate([#
    np.concatenate([np.array(start[:1]),np.cumsum(steps*np.sin(thetas)*np.sin(phis))+start[0]]),#
    np.concatenate([np.array(start)[1:2],np.cumsum(steps*np.sin(thetas)*np.cos(phis))+start[1]]),#
    np.concatenate([np.array(start)[2:],np.cumsum(steps*np.cos(thetas))+start[2]])
    ]).reshape(3,-1).transpose()
    return rw, thetas, phis


def scp(a1,a2,b1,b2): 
    """returns the cosine between two vectors a and b via the normalised dot product"""
    return (a1*b1+a2*b2)/(np.sqrt(a1**2+a2**2)*np.sqrt(b1**2+b2**2))
    

def correl(d,tmax, smooth=5, verbose=False):
    """Cosine correlation function: """
    c=[]
    for dT in range(tmax):
        c+=[np.mean(np.array([scp(d[j,0]-d[j-smooth,0], d[j,1]-d[j-smooth,1], d[j+dT,0]-d[j+dT-smooth,0], d[j+dT,1]-d[j+dT-smooth,1]) for j in range(smooth,len(d[:,0])-tmax)]))]
        if verbose:
            if dT%verbose==0: print dT
    return c
    
    
def txtheader(fname, comment='#'):
    with open(fname) as f: header=f.readline()[:-1]
    if comment!='':
        if header[0]!=comment: return {}
        while header[0]==comment: header=header[1:]
    header=header.strip()
    sp=header.split()
    return {sp[i]:i for i in range(len(sp))}
    
def deinterlace(ar,axis=0, first=0):
    """splits interlaced image into A and B frame blown up to dimensions of original image..
    Input: numpy array
    Keyword arguments*: 
    -axis, value 0 (default) or 1. Standard interlacing assumes interpolated y axis, axis=0.
    -first: value 0 (default) or 1. Determines order of output arrays. Even rows first is default.
    Output:  two interpolated arrays."""
    ar0,ar1=ar.copy(),ar.copy()
    if axis==0:
        ar0[1::2,:]=ar[::2,:]
        ar1[0::2,:]=ar[1::2,:]
    if axis==1:    
        ar0[:,1::2]=ar[:,::2]
        ar1[:,0::2]=ar[:,1::2]
    if first==0: return ar0,ar1
    if first==1: return ar1,ar0

def contComp(cont,contSet):
    M=cv2.moments[cont]
    area,x,y= M['m00'],M['m10']/M['m00'], M['m01']/M['m00']
    clist=[]
    for i in range(len(contSet)):
        clist+=[np.hstack([np.array([[i]]*len(contSet[i])),np.array(contSet[i][:,0,:])])]
    clist=np.vstack(clist)
    sqdist=(x-clist[:,1])**2+(y-clist[:,2]**2)
    nextcnt=clist[sqdist==min(sqdist),1:]
    cont=np.array(cont[:,0,:])
    refinedist=(nextcnt[0]-cont[0,:])**2+(nextcnt[1]-cont[1,:])**2
    


        
    
def stringdiff(str1,str2):
    diff1,diff2='',''
    for i in range(len(str1)):
        if str1[i]!=str2[i]: 
            diff1+=str1[i]
            diff2+=str2[i]
    return diff1,diff2
            
        