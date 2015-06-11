import matplotlib
#matplotlib.use('WXAgg')
from matplotlib import pyplot as pl  #this is Python's main scientific plotting library.
from matplotlib import cm
import cv2 #computer vision library. interfaces with python over numpy arrays.
import numpy as np
from scipy import stats
from sys import argv #for command line arguments
from os.path import splitext, basename, exists, sep #some file path handling
import os
from PIL import Image
from time import time
from datetime import timedelta
from itertools import chain
from glob import glob
from scipy.misc import imresize
from string import digits
import re
import subprocess
import MySQLdb as mdb
import types
import cPickle
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
sp=COORDHEADER2D.split()
cheaderdict2D= {sp[i]:i for i in range(len(sp))}
COORDHEADER3D='#frame stackframe particle# blobsize x y z t\n'
sp=COORDHEADER3D.split()
cheaderdict3D= {sp[i]:i for i in range(len(sp))}
TRAJHEADER="frame particle# x y blobsize"
sp=TRAJHEADER.split()
ttheadheaderdict= {sp[i]:i for i in range(len(sp))}




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
        self.data=np.array(data) #trajectory coordinate series, lengthX3 array
        self.opened=True #flag: trajectory considered lost when no particle closer than max. distance in next frame.
        self.maxdist=maxdist #set maximum distance
        self.number=number #trajectory ID
        self.lossCnt=0
        self.dim=dim

    def findNeighbour(self,nxt, frIdx, idx=3, lossmargin=10, spacing=1):
        """Finds the nearest neighbour in particle coordinate set from next frame.
        Accepts numpy array with coordinate data from the following movie frame with blobsize, x and y data.
        Keywords:
            idx (int or dict): integer index of x column assuming column sequence (blobsize, x, y). If that order is not present in the input, pass a tuple/list of indices in order area,x,y or a dictionary as in {'blobsize':2,'x':3,'y':4} (use these exact keys)
            lossmargin (int): if no particle is found inside the trajectory's self.mindist radius, trajectory is linearly extrapolated for this # of frames (helps with 'blinking' blobs). Particle is considered lost beyond that frame range.
            spacing (int): For non consecutive frame numbers if the movie was decimated during analysis. E.g. spacing=5 for frames 0,5,10,15,...
        Extends/extrapolates self.data array with new coordinates if successful, Closes trajectory if not.
        Returns next neighbour numpy array with matching particle removed for speedup and to avoid double counting."""
        
        #sort out what index information is given (OK, this is overkill):
        #note: zIdx is always set but not used in case of 2D data. Probably more robust.
        try:
            if type(idx) is int or float: #scalar
                szIdx,xIdx,yIdx, zIdx=int(idx)-1,int(idx),int(idx)+1,int(idx)+2                
            elif hasattr(idx,'__getitem__'): #iterable
                szIdx,xIdx,yIdx=int(idx[0]),int(idx[1]),int(idx[2])
                if self.dim==3: zIdx=int(idx[3])
                else: zIdx=-1
            else:
                raise TypeError("Scalar or iterable, please.")
            if type(idx) is dict:
                szIdx,xIdx,yIdx=idx['blobsize'],idx['x'],idx['y']
                if self.dim==3: zIdx=idx['z']
                else: zIdx=-1             
        except (KeyError, ValueError, TypeError, IndexError):
            szIdx,xIdx,yIdx,zIdx=2,3,4,5
            print "Warning: Couldn't detect column indices.\nAssuming default values:\n\tsize: %d\n\tx: %d\n\ty: %d%d\n\tz: %d"%(szIdx,xIdx,yIdx,zIdx)

        if frIdx-self.data[-1,0]>(lossmargin+1)*spacing: #if there are frame continuity gaps bigger than the loss tolerance, close trajectory!
            self.opened=False
            return nxt

        if nxt.size>0:
            if self.dim==2: dist=(self.data[-1,2]-nxt[:,xIdx])**2+(self.data[-1,3]-nxt[:,yIdx])**2
            if self.dim==3: dist=(self.data[-1,2]-nxt[:,xIdx])**2+(self.data[-1,3]-nxt[:,yIdx])**2+(self.data[-1,4]-nxt[:,zIdx])**2
            m=min(dist)
        else:
            m=self.maxdist+1 #this will lead to trajectory closure
            print "no particles left in this frame", frIdx, self.number
        if m<self.maxdist:
            ind=(dist==m).nonzero()[0]
            try:
                if self.dim==2: self.data=np.vstack((self.data,np.array([[frIdx,ind, nxt[ind,xIdx],nxt[ind,yIdx], nxt[ind,szIdx]]]))) #append new coordinates to trajectory
                if self.dim==2: self.data=np.vstack((self.data,np.array([[frIdx,ind, nxt[ind,xIdx],nxt[ind,yIdx], nxt[ind,zIdx], nxt[ind,szIdx]]]))) #append new coordinates to trajectory

            except IndexError:
                print "SOMETHING WRONG HERE!", self.data.shape, nxt.shape, frIdx, self.number #not sure what is.
                self.lossCnt+=1
                if self.lossCnt>lossmargin:
                    self.opened=False #close trajectory, don't remove particle from coordinate array.
                else:
                    predCoord=lin_traj(self.data[-lossmargin:,2],self.data[-lossmargin:,3])
                    if np.isnan(predCoord[0]): predCoord=self.data[-1][2:2+self.dim]# if extrapolation fails, just keep particle where it is. 
                    if self.dim==2: self.data=np.vstack((self.data,np.array([[frIdx, -1, predCoord[0], predCoord[1], self.data[-1,-1]]])))
                    if self.dim==3: self.data=np.vstack((self.data,np.array([[frIdx, -1, predCoord[0], predCoord[1], predCoord[2], self.data[-1,-1]]])))
                return nxt
            self.lossCnt=0
            return np.delete(nxt,ind,0) #remove particle and return coordinate set.
        else:
            self.lossCnt+=1
            if self.lossCnt>lossmargin:
                self.data=self.data[:-lenlim,:] #cut extrapolated values
                self.opened=False #close trajectory, don't remove particle from coordinate array.
            else:
                if self.dim==2: predCoord=lin_traj(self.data[-lossmargin:,2],self.data[-lossmargin:,3])
                if self.dim==3: predCoord=lin_traj(self.data[-lossmargin:,2],self.data[-lossmargin:,3], self.data[-lossmargin:,4])
                if np.isnan(predCoord[0]): predCoord=self.data[-1][2:2+self.dim]
                if self.dim==2: self.data=np.vstack((self.data,np.array([[frIdx, -1,predCoord[0], predCoord[1],self.data[-1,-1]]])))
                if self.dim==3: self.data=np.vstack((self.data,np.array([[frIdx, -1,predCoord[0], predCoord[1], predCoord[2], self.data[-1,-1]]])))
            return nxt




class movie():
    """Class for handling 2D video microscopy data. Additionally requires mplayer in the PATH.
        Argument: video filename.
        Keyword parameters: TTAB concentration and background filename.
        Attributes:
            fname: filename (string)
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
    def __init__(self,fname, TTAB=-1, bg=''):
        """Initialises movie object. Parameter video filename, keywords TTAB (surfactant concentration), path to extrapolated background file.
        """
        self.typ="Particles"
        self.fname=fname
        self.trajectories={}
        self.bg=False
        self.datadir=splitext(fname)[0]+'-data'+sep
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
            result = subprocess.check_output(['mplayer','-vo','null','-ao','null','-identify','-frames','0',self.fname])
        if os.name=='nt': #this assumes you installed mplayer and have the folder in your PATH!
            result = subprocess.check_output(['mplayer.exe','-vo','null','-ao', 'null','-identify','-frames','0',self.fname])
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
            'imsize':shape,'blobsize':(0,30),'crop':[0,0,shape[0],shape[1]], 'framelim':framelim, 'circle':[shape[0]/2, shape[1]/2, int(np.sqrt(shape[0]**2+shape[1**2]))],#tuples
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
                self.parameters[t[0]]=int(t[1])
            if t[0].strip() in ['blobsize','imsize', 'crop','framelim', 'circle']:#tuple parameters
                tsplit=re.sub('[\s\[\]\(\)]','',t[1]).split(',')
                self.parameters[t[0]]=tuple([int(it) for it in tsplit])
            if t[0].strip() in ['framerate','sphericity','xscale','yscale','zscale']:#float parameters
                self.parameters[t[0]]=float(t[1])
            if t[0].strip() in ['sizepreview','mask','diskfit','invert']:#boolean parameters
                self.parameters[t[0]]=str_to_bool(t[1])
        if self.parameters['struct']>1: self.kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.parameters['struct'],self.parameters['struct']))
        else: self.kernel=False


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
        mov=cv2.VideoCapture(self.fname) #open movie (works on both live feed and saved movies)
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
                if type(kernel).__name__=='ndarray': thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
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

    def getFrame(self,framenum):
        """Retrieves frame of number framenum from open movie. Returns numpy array image, or False if unsuccessful.
        Due to movie search/keyframe issues, framenumber might not be exact."""
        mov=cv2.VideoCapture(self.fname)
        if framenum>1: s,r,p=self.gotoFrame(mov,framenum-1)
        success,image=mov.read()
        if success: return image
        else: return False

    def loadBG(self, filename=''):
        if filename=="": filename=self.datadir+'bg.png'
        self.bg=np.array(Image.open(filename))

    def getBGold(self, num=50, spac=50, prerun=1000, cutoff=100, save=False, channel=0):
        mov=cv2.VideoCapture(self.fname)
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
        mov=cv2.VideoCapture(self.fname)
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
        mov=cv2.VideoCapture(self.fname)
        if position>0: success,images,p=self.gotoFrame(mov,position)
        else: success,images=mov.read()
        images=images[:,:,channel]
        for i in range(length-1)*spacing:
            s,im=mov.read()
            if i%spacing==0:
                images=np.dstack((images, im[:,:,channel]))
        mov.release()
        return images

    def CoordtoTraj(self, tempfile='temp',lenlim=-1, delete=True, maxdist=-1, lossmargin=-1, spacing=1, idx=3, consolidate=False):#TODO Adjust for frame jumps!!!
        """Needs more documentation!!
        Input (only keywords):
            tempfile (default self.datadir+'temp'): path of file containing coordinate data
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
        if tempfile=='temp':tempfile=self.datadir+'temp' #TODO: 'coords.txt'!!!
        if tempfile=='coords.txt':tempfile=self.datadir+'coords.txt'
        if maxdist<0: maxdist=self.parameters['maxdist'] #if not set, take parameter file value. SQUARED VALUE!
        if lossmargin<0: lossmargin=self.parameters['lossmargin'] 
        if lenlim<0: lenlim=self.parameters['lenlim'] 
        #the 'consolidate' flag produces an average coordinate/frame number and 3D size (voxel) for each closed trajectory and writes them into a single coordinate file to get z stack tracing. A second tracking run will then generate time tracking. No single trajectory output.
        if consolidate: 
        #TODO: for particles at the upper/lower boundaries, this would result in trajectories spanning 2 stacks. FIX THIS! (means we need the reversal points. Consider writing a new CoordtoTraj for the stack data type)
            np.set_printoptions(precision=3,suppress=True)
            if type(consolidate) is not str: stckf=open('stackcoord.txt','w')
            else: stckf=open(consolidate,'w')
            stckf.write(TRAJHEADER)
        print """
        maxdist: %f
        lossmargin: %d
        lenlim: %d
        """%(maxdist, lossmargin,lenlim)
        dataArr=np.loadtxt(tempfile)
        #here, extract column headers
        colheaders=txtheader(tempfile)
        try:
            szIdx,frIdx,ptIdx,xIdx,yIdx=colheaders['blobsize'],colheaders['particle#'],colheaders['frame'],colheaders['x'],colheaders['y']
            try: zIdx=colheaders['z']
            except KeyError: zIdx=-1 #if there's no z do nothing.
        except KeyError:
            szIdx,frIdx, ptIdx,xIdx, yIdx, zIdx =idx-1,0,1, idx, idx+1, idx+2
            print "Warning: Couldn't extract column headers from data file.\nAssuming default/keyword values for column indices:\n\t blobsize: \n\tframe #: %d\n\tparticle # in frame: %d\n\tx: %d\n\ty %d\n\tz: %d"%(szIdx,frIdx,ptIdx,xIdx,yIdx, zIdx)            
        trajectorycount=0
        frames=sorted(list(set(dataArr[:,frIdx])))
        #put in frame range here!
        activetrajectories={}
        for i in range(1,len(frames)):
            try: arrInd=np.searchsorted(dataArr[:,frIdx], frames[i])
            except IndexError: break
            blobs,dataArr=np.split(dataArr, [arrInd])
            if frames[i]%400==0:
                print "framenum", frames[i], 'remaining data', dataArr.shape, 'active traj.', len(activetrajectories), 'time', time()-t0
            for tr in activetrajectories.values():
                blobs=tr.findNeighbour(blobs, frames[i], idx=(szIdx,xIdx,yIdx,zIdx), lossmargin=lossmargin) #for each open trajectory, find corresponding particle in circle set
                if not tr.opened: #if a trajectory is closed in the process (no nearest neighbour found), move to closed trajectories.
                    if tr.data.shape[0]>lenlim:
                        if not consolidate: 
                            np.savetxt(self.datadir+'trajectory%06d.txt'%tr.number, tr.data, fmt='%.2f', header=TRAJHEADER)
                            print "closed trajectory: ", tr.number, tr.maxdist
                        else:
                            trmean=list(np.mean(tr.data,axis=0))
                            trmean[-1]=tr.data.shape[0]*trmean[-1]
                            stckf.write(str(trmean)[1:-1].strip()+'\n')                            
                    del activetrajectories[tr.number]
            for blob in blobs: #if any particles are left in the set, open a new trajectory for each of them
                trajectorycount+=1
                activetrajectories[trajectorycount]=trajectory(np.array([[frames[i],blob[1],blob[3],blob[4],blob[2]]]),trajectorycount, maxdist=maxdist)
        print "trajectories:", len(activetrajectories)
        for tr in activetrajectories.values():
            if not consolidate:
                np.savetxt(self.datadir+'trajectory%06d.txt'%tr.number, tr.data, fmt='%.2f',  header=TRAJHEADER)
                print "closed trajectory: ",tr.number, np.sqrt(tr.maxdist)
            else:
                trmean=list(np.mean(tr.data,axis=0))
                trmean[-1]=tr.data.shape[0]*trmean[-1]
                stckf.write(str(trmean)[1:-1].strip()+'\n')
        try: stckf.close()
        except: pass

    def plotMovie(self, outname=None, decim=10,scale=2, crop=[0,0,0,0], mask='trajectory',frate=10, cmap=cm.jet, bounds=(0,1e8), tr=True, channel=0, lenlim=1):
        """crop values: [left, bottom, right, top]"""
        if not outname: outname=self.datadir+basename(self.fname)[:-4]+'-traced.avi'
        mov=cv2.VideoCapture(self.fname)
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
                        pts = trajectories[i][:,2:4].astype(np.int32) #check indices and shape!!!
                        colour=tuple([int(255*r) for r in cmap(np.float(i)/len(trajectories))[:3]])[::-1]
                        #colour=(0,120,0)
                        cv2.polylines(image,[pts],isClosed=False,color=colour,thickness=int(np.round(scale)))
                    else:
                        w=(trajectories[i][:,0]==count).nonzero()[0]
                        if len(w)>0:
                            pts = trajectories[i][:w[0],2:4].astype(np.int32) #check indices and shape!!!
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

    def loadTrajectories(self, directory=None, mask='trajectory*.txt'):
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
            pl.imshow(f0, cmap=cm.gray)
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
                    pl.plot(tr.data[:,idx],tr.data[:,idx+1],lw=.3, color=cl)
                    if text: pl.text(tr.data[0,idx], tr.data[0,idx+1], str(tr.number), color=cl,fontsize=6)
                else:
                    cv2.polylines(f0,[tr.data[idx:idx+2].astype(np.int32)],isClosed=False,color=cl,thickness=2)
        if mpl:
            pl.axis('off')
            pl.savefig(self.datadir+outname,dpi=600)
            pl.close('all')
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
                    np.savetxt(save+'%06d.txt'%(tr.number),tr.data,fmt='%.03f',  header=TRAJHEADER)

    def Histogram(self, fnum, fname="temphist.png", channel=0):
        """plots the RGB histogram for frame # fnum. Auxiliary function for remote parameter setting. Replaces HistoWin in parameter GUI."""
        image=self.getFrame(fnum)
        if type(image).__name__=='ndarray':
            if len(image.shape)==3 and image.shape[2]!=3: image=dstack((image[:,:,channel],image[:,:,channel],image[:,:,channel]))
            pl.figure(figsize=[6,4])
            for i in range(3):
                pl.hist(image[:,:,i].flatten(), bins=256, log=True, histtype='step',align='mid',color='rgb'[i])
            pl.savefig(fname,dpi=100,format='png')
            pl.close("all")

    def testImages(self,fnum, mask=False,BGrng=(100,255), channel=0):
        """analogue for ImgDisplay in parametergui.py"""
        if type(mask).__name__=='str':
            try:
                im=np.array(Image.open(self.datadir+'mask.png'))
                if len(im.shape)==3: im=im[:,:,channel]
                mask=(im>0).astype(float)
            except: mask=zeros(self.movie.shape[::-1])+1.
        else: mask=zeros(self.movie.shape[::-1])+1.
        if type(self.bg).__name__!='ndarray':
            if os.path.exists(self.datadir+'bg.png'):
                self.loadBG()
            else:
                bg=self.getBG(rng=BGrng, num=50, spac=int(self.parameters['frames']/51), prerun=100, save=True)
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
                    np.savez_compressed(mov.datadir+'gridanalysis',np.array(tcorr),hist,bins[0],bins[1],bins[2])
                    np.savetxt(mov.datadir+'crossing.txt', np.array(tcorr), fmt='%.5e')

                    return np.array(tcorr), hist,bins

                except MemoryError:
                    print "Excessive histogram size! Please increase granularity!"
                    return numpy.np.zeros((0)),numpy.np.zeros((0,0,0)), (numpy.np.zeros((0)),numpy.np.zeros((0)),numpy.np.zeros((0)))
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
                        np.savez_compressed(mov.datadir+'gridanalysis',gA[0],gA[1],gA[2][0],gA[2][1],gA[2][2])
                        np.savetxt(mov.datadir+'crossing.txt', gA[0], fmt='%.5e')

                        return np.array(tcorr), hist,bins

                except MemoryError:
                    print "Excessive histogram size! Please increase granularity!"
                    return numpy.np.zeros((0)),numpy.np.zeros((0,0,0)), (numpy.np.zeros((0)),numpy.np.zeros((0)),numpy.np.zeros((0)))

        else:
            print "no trajectories found!"
            return numpy.np.zeros((0,0,0))

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
        mov=cv2.VideoCapture(self.fname)
        framenum=rng[0]
        if rng[0]>1:
            success,image,p= self.gotoFrame(mov,rng[0]-1)
        gkern=int(gkern)
        if gkern%2==0:
            print "warning: Gaussian kernel size has to be odd. New size %d."%(gkern+1)
            blur=blur+1
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
    def __init__(self,fname):
        self.typ="3D stack"
        self.bg=False
        self.fname=fname
        spex=os.path.splitext(os.path.basename(fname))
        search=re.sub('[0-9]',"?",spex[0])
        self.stack=sorted(glob(os.path.dirname(fname)+os.sep+search+spex[1]))
        test0,test1=self.stack[:2]
        while test0!=test1: test0,test1=test0[:-1],test1[:-1]
        while test0[-1] in '0123456789': test0=test0[:-1]
        self.datadir=test0+'-data'+sep
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
        'blobsize':(0,30),'imsize':shape,'crop':[0,0,shape[0],shape[1]], 'framelim':framelim,'circle':[shape[0]/2, shape[1]/2, int(np.sqrt(shape[0]**2+shape[1**2]))],#tuples
            'channel':0, 'blur':1,'spacing':1,'struct':1,'threshold':128, 'frames':frames,  'imgspacing':-1,'maxdist':-1,'lossmargin':10, 'lenlim':1,#ints
            'sizepreview':True, 'invert':False, 'diskfit':False, 'mask':True
            }

    def getFrame(self,framenum):
        """Retrieves frame of number framenum from opened stack. Returns numpy array image, or False if unsuccessful."""
        try:
            image=cv2.imread(self.stack[framenum],1)
            return image
        except:
            return False

    def extractCoords(self,framelim=False, blobsize=False, threshold=False, kernel=False, delete=True, mask=False, channel=False, sphericity=-1, diskfit=True, blur=1,invert=True,crop=False, contours=False): #fix the argument list! it's a total disgrace...
        tInit=time()
        contdict={}
        if not framelim: framelim=self.parameters['framelim']
        if not blobsize: blobsize=self.parameters['blobsize']
        if not threshold: threshold=self.parameters['threshold']
        if not channel: channel=self.parameters['channel']
        if not crop: crop=self.parameters['crop']
        if type(kernel).__name__!='ndarray': kernel=np.array([1]).astype(np.uint8)
        if not exists(self.datadir):
            os.mkdir(self.datadir)
        if delete:
            try: os.remove(self.datadir+'coords.txt')
            except: pass
        dumpfile=open(self.datadir+'coords.txt','a')
        dumpfile.write('#frame particle# blobsize x y split_blob? [reserved] sphericity\n')
        allblobs=np.array([]).reshape(0,8)
        counter=0
        for i in range(len(self.stack)):
            if i%200==0:
                print 'frame',i, 'time', str(timedelta(seconds=time()-tInit)), '# particles', counter #progress marker
                np.savetxt(dumpfile,allblobs,fmt="%.2f")
                allblobs=np.array([]).reshape(0,8)
            image=self.getFrame(i)
            if type(image).__name__=='ndarray':
                if image.shape[:2]!=(crop[2]-crop[0],crop[3]-crop[1]):
                    if len(image.shape)==2: image=image[crop[0]:crop[2],crop[1]:crop[3]]
                    if len(image.shape)==3: image=image[crop[0]:crop[2],crop[1]:crop[3],:]
                if len(image.shape)>2:
                    image=image[:,:,channel].astype(float)
                #image=mxContr(image) #TODO: this might be a few rescalings too many. try to make this simpler, but make it work first
                thresh=mxContr((image<threshold).astype(int))
                if type(kernel).__name__=='ndarray': thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                if invert: thresh=255-thresh
                if np.amin(thresh)!=np.amax(thresh):
                    if contours: blobs,conts=extract_blobs(thresh,i,sphericity=sphericity,blobsize=blobsize,diskfit=diskfit, returnCont=True)
                    else: blobs,conts=extract_blobs(thresh,i,sphericity=sphericity,blobsize=blobsize,diskfit=diskfit, returnCont=False),[]
                else: blobs,conts=np.array([]).reshape(0,8),[]
                counter=blobs.shape[0]
                try: allblobs=np.vstack((allblobs,blobs))
                except ValueError:
                    pass
                    #print "Value Error!", allblobs.shape, blobs.shape
                for i in range(len(conts)): contdict["%d-%d"%(blobs[0,0],i)]=conts[i]
        np.savetxt(dumpfile,allblobs,fmt="%.2f")
        dumpfile.close()
        with open(self.datadir+'coords.txt','r') as f: tempdata=f.read()[:-1]
        with open(self.datadir+'coords.txt','w') as f: f.write(tempdata)
        if len(contdict)>0:
            with open(self.datadir+'contours.pkl','wb') as f:
                cPickle.dump(contdict,f,cPickle.HIGHEST_PROTOCOL)
    
    def Coord3D(coordfile,zframefile,stacksplitfile, outfile):
        """Helper function to reshuffle the coordinate data created by the CoordtoTraj method with the consolidate keyword. 
        Input data:
            coordfile containing coordinates with columns as in header: %s
                (CoordtoTraj output)
            zframefile: file containing list of z positions per frame (1 column)
            stacksplitfile: file containing list of reversal frames for stack splitting locations.
        Output file:
            data format as in %s"""%(TRAJHEADER,COORDHEADER3D)
        try: os.remove(self.datadir+'temp')
        except OSError: pass
        output= open(outfile, 'a')
        output.write(COORDHEADER3D)
        colheaders=txtheader(coordfile)
        try:
            szIdx,frIdx,ptIdx,xIdx,yIdx=colheaders['blobsize'],colheaders['particle#'],colheaders['frame'],colheaders['x'],colheaders['y']
        except KeyError:
            szIdx,frIdx,ptIdx,xIdx,yIdx=theaderdict['blobsize'],theaderdict['particle#'],theaderdict['frame'],theaderdict['x'],theaderdict['y']
            print "Warning: Couldn't extract column headers from data file.\nAssuming default values for column indices:\n\t blobsize: \n\tframe #: %d\n\tparticle # in frame: %d\n\tx: %d\n\ty %d"%(szIdx,frIdx,ptIdx,xIdx,yIdx) 
        inpdata=np.loadtxt(coordfile)
        
        
            
            

    def blenderPrep(self, nfacets=10, smoothlen=5):
        self.loadTrajectories()
        if len(self.trajectories)>0 and os.path.isfile(self.datadir+'contours.pkl'):
            with open(self.datadir+'contours.pkl', 'rb') as f:
                conts=cPickle.load(f)
            todelete=glob(self.datadir+'pointfile*.txt')+glob(self.datadir+'vertfile*.txt')
            for fname in todelete: os.remove(fname)
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

class nomovie(movie):    
    def __init__(self,datadir):
        self.typ="none"
        self.bg=False
        self.datadir=datadir
        self.parameters={
            'framerate':-1, 'sphericity':-1.,'xscale':1.0,'yscale':1.0,'zscale':1.0,#floats
            'blobsize':(0,30),'imsize':(1920,1080),'crop':[0,0,1920,1080], 'framelim':1e9,'circle':(1000, 500,500),#tuples
            'channel':0, 'blur':1,'spacing':1,'struct':1,'threshold':128, 'frames':1e9,  'imgspacing':-1,'maxdist':-1,'lossmargin':10, 'lenlim':1,#ints
            'sizepreview':True, 'invert':False, 'diskfit':False, 'mask':True
                    }
                    
    def getFrames(self,position, length=1, channel=0,spacing=1, fname=''):
        if fname=='': fname=self.datadir+'bg.png'
        im=np.array(Image.open(fname))
        if len(im.shape)==3: im=im[:,:,channel]
        images=np.dstack([im]*length)
        return images
                    

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

def mxContr(data):
    mx,mn=np.float(np.amax(data)),np.float(np.amin(data))
    if mx!=mn:
        return (255*(np.array(data)-mn)/(mx-mn)).astype(np.uint8)
    else:
        print 'Warning, monochrome image!'
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
    if z: zslope, intercept, r_value, p_value, std_err = stats.linregress(x,z)
    mx=np.mean((x-np.roll(x,1))[1:])
    if not z: return np.array([x[-1]+mx, y[-1]+yslope*mx])
    else: return np.array([x[-1]+mx, y[-1]+yslope*mx], z[-1]+zslope*mx])

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
        w=ones(window_len,'d')
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
    ang=[0]
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
    rw=[list(start)] #user provided initial coordinates: make (reasonably) sure it's a nested list type.
    ang=[0]
    #step lengths are precalculated for each step to account for 
    if dist=="cauchy": steps=stats.cauchy.rvs(size=lgth)
    elif dist=="norm": steps=stats.norm.rvs(size=lgth)
    else: steps=np.array([step]*lgth) #Overkill for constant step length ;)
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
    
    
def correl(d,tmax):
    """Cosine correlation function: """
    c=[]
    for dT in range(tmax):
        c+=[np.mean(np.array([scp(d[j,0]-d[j-1,0], d[j,1]-d[j-1,1], d[j+dT,0]-d[j+dT-1,0], d[j+dT,1]-d[j+dT-1,1]) for j in range(1,len(d[:,0])-tmax)]))]
    return c
    
    
def txtheader(fname, comment='#'):
    with open(fname) as f: header=f.readline()[:-1]
    while header[0]==comment: header=header[1:]
    sp=header.split()
    return {sp[i]:i for i in range(len(sp))}
    