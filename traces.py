#!/usr/bin/env python
# -*- coding: utf-8 -*-

#imported modules
from numpy import array, cumsum, concatenate, sqrt,cos,sin,arctan,pi,e, mean, sum, loadtxt, hstack, savetxt, polyfit, unwrap
from scipy.stats import cauchy,norm, uniform
from scipy import optimize
from os.path import splitext,isfile

from matplotlib import pyplot as pl
from matplotlib import ticker

from pygame import mixer as pgmixer
from pygame import init as pginit

#global declarations
window=6

def scp(a1,a2,b1,b2): 
    """returns the cosine between two vectors a and b via the normalised dot product"""
    return (a1*b1+a2*b2)/(sqrt(a1**2+a2**2)*sqrt(b1**2+b2**2))


def correl(d,tmax):
    """Cosine correlation function: """
    c=[]
    for dT in range(tmax):
        c=c+[mean(array([scp(d[j,0]-d[j-1,0], d[j,1]-d[j-1,1], d[j+dT,0]-d[j+dT-1,0], d[j+dT,1]-d[j+dT-1,1]) for j in range(1,len(d[:,0])-tmax)]))]
    return c

    
#source http://wiki.scipy.org/Cookbook/Least_Squares_Circle
#https://gist.github.com/lorenzoriano/6799568

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return sqrt((x-xc)**2 + (y-yc)**2)
 
def rdist(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()
 
def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = mean(x)
    y_m = mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(rdist, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = sum((Ri - R)**2)
    return xc, yc, R, residu 
    
def mSqDisp(traces):
    """input: l x 2 x n array of n traces of length l. Output: mean squared displacement"""
    pass
    
    
def radGyr(d,window=window):
    """Calculates the local radius of gyration via a least squares circle fit to a window around each point on the particle trace. Output: List of r_gs."""
    rgs=[] #initialise list
    d=array(d) #make sure it's an array
    for t in range(window,len(d[:,0]-window)):
        xc, yc, R, residu = leastsq_circle(d[t-window:t+window,0],d[t-window:t+window,1])
        rgs=rgs+[R]
    return rgs

def ang_prog(d,window=window,jump=2.):
    """Calculates the local orientation from the arctan of a linear fit to a data window around each trace point. 
    Arguments: 2xlength array.
    Optional parameters: data window width "window", discontinuity threshold "jump".
    Data jumps resulting from the pole at +- pi/2 are smoothed by adding +- pi where the angular trace is discontinuous by more than the "jump" parameter.
    Output: angular progression as list."""
    d=array(d) #make sure it's an array
    orient=[polyfit(d[j-window:j+window,0], d[j-window:j+window,1], 1)[0] for j in range(window,len(d[:,0])-window)] #linear fit (poly fit to first order)
    ang=array([arctan(o) for o in orient]) #arctan of local gradient
    for i in range(1,len(ang)): #jump correction
        if ang[i]-ang[i-1]<-2: ang[i:]=ang[i:]+pi
        if ang[i]-ang[i-1]>2: ang[i:]=ang[i:]-pi
    return ang
        
    
def randwalk(lgth,stiff,start=[0.,0.], step=1.,adrift=0.,anoise=.2, dist="const"):
    """generates a 2d random walk with variable angular correlation and optional constant (+noise) angular drift.
    Arguments: walk length "lgth", stiffness factor "stiff" - consecutive orientations are derived by adding stiff*(random number in [-1,1]).
    Parameters "adrift" and "anoise" add a constant drift angle adrift modulated by a noise factor adrift*(random number in [-1,1]).
    The dist parameter accepts non-constant step length distributions (right now, only cauchy/gaussian distributed random variables)"""
    rw=[list(start)] #user provided initial coordinates: make (reasonably) sure it's a nested list type.
    ang=[0]
    #step lengths are precalculated for each step to account for 
    if dist=="cauchy": steps=cauchy.rvs(size=lgth)
    elif dist=="norm": steps=norm.rvs(size=lgth)
    else: steps=array([step]*lgth) #Overkill for constant step length ;)
    #first generate angular progression vila cumulative sum of increments (random/stiff + drift terms)
    angs=cumsum(stiff*uniform.rvs(size=lgth,loc=-1,scale=2)+adrift*(1.+anoise*uniform.rvs(size=lgth,loc=-1,scale=2)))
    #x/y trace via steplength and angle for each step, some array reshuffling (2d conversion and transposition)
    rw=concatenate([concatenate([array(start[:1]),cumsum(steps*sin(angs))+start[0]]),concatenate([array(start)[1:],cumsum(steps*cos(angs))+start[1]])]).reshape(2,-1).transpose()
    return rw, angs
    

def savetrace(fname,data,ident=0.):
    l=data.shape[0]
    if not isfile(fname):
        with open(fname,'w') as f: f.write("#\n")
    with open(fname,'a') as f:            
        savetxt(f,hstack((array([-1.]*l).reshape(-1,1),array([ident]*l).reshape(-1,1),data)))

    

def quadplot(fname, ident='', sound=False):
    """quadruple plot with trace series, """
    pcdata=loadtxt(fname,skiprows=1) #read data set for each concentration
    pl.close("all") #close all preexisting pyplot figures
    fig=pl.figure(figsize=[20,15])
    for i in set(pcdata[:,1]): 
        trace=pcdata[pcdata[:,1]==i,2:4]#extract data set for specific trace
        print "%s, trace # %.4f, length %d"%(ident,i,len(trace[:,0]))
        ax1=fig.add_subplot(221, aspect='equal')# first plot: just traces
        ax1.plot(trace[:,0], trace[:,1],label="%.4f"%i)
        ax2=fig.add_subplot(222) 
        c=correl(trace,len(trace[:,0])-500)
        ax2.plot(c, label="%.4f"%i)
        ax3=fig.add_subplot(223)            
        ax3.plot(ang_prog(trace,jump=2.2*pi,window=2), label="%.4f"%i) #plot angular progression
        ax4=fig.add_subplot(224)     
        ax4.semilogy(c[:200], label="%.4f"%i) #plot angular progression
    ax1.set_title('traces %s'%ident, fontsize=16)
    ax2.set_title('correl.  %s'%ident, fontsize=16)
    #
    ax3.set_title('angular prog. %s'%ident, fontsize=16)
    ax4.plot([0,200],[1./e,1./e],"k--")
    ax4.set_ylim((.2,1))
    ax4.set_xlim([0,200])
    ax4.set_title('correlation function %s '%ident, fontsize=16)
    ax4.legend(fontsize=15,loc=3)
    loc = ticker.MultipleLocator(base=2*pi) # this locator puts ticks at regular intervals
    ax3.yaxis.set_major_locator(loc)
    ax3.yaxis.set_major_formatter(pl.FuncFormatter(lambda loc, pos: '%d$\pi$'%(loc/pi)))
    ax3.grid()
    pl.savefig(splitext(fname)[0]+'.pdf')
    pl.rcdefaults()
    if sound:
        pginit()
        note=pgmixer.Sound('/usr/share/sounds/KDE-Sys-Special.ogg')
        note.set_volume(1.0)
        note.play()

    
if __name__=="__main__":
    """If script is called from data directory, it will generate some data plots. Warning: takes a while!"""
    pcs=[7,10,15,20,25]
    for pc in pcs: #loop over all concentrations        
        quadplot('%dwtpcTTAB_23degC_24fps_shape.dat'%pc, ident="%d %%"%pc,sound=True)