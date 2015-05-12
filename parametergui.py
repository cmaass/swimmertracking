#!/usr/bin/env python

#Author: rosencrantz@gmx.net
#License: GPL

import wx
import wx.lib.scrolledpanel as scp
import numpy as np
from PIL import Image
from matplotlib import pylab as pl
from matplotlib import cm
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import os
import io
import pickle
import cv2
import subprocess
import re
import readtraces as rt
from random import randint
from sys import exc_info


#this directory definition is changed in the source code at runtime which is probably a really bad idea but good for portability
moviedir='/media/cmdata/datagoe/mazes/manhattan/'#end

def GetBitmap(width=1, height=1, colour = (0,0,0) ):
    """Helper funcion to generate a wxBitmap of defined size and colour.
    Prettier (and possibly less embarassing) than putting an empty bitmap on your GUI showing whatever garbage is still in your bitmap buffer.
    Source: wxpython wiki"""
    ar = np.zeros( (height, width, 3),'uint8')
    ar[:,:,] = colour
    image = wx.EmptyImage(width,height)
    image.SetData(ar.tostring())
    wxBitmap = image.ConvertToBitmap()       # OR:  wx.BitmapFromImage(image)
    return wxBitmap


class InfoWin(wx.Frame):
    """Child window displaying image detection parameters for the current movie.
    Parameters are taken from the parent window's parameters attribute (a dictionary).
    Update methods needs to be called to catually show anything."""
    #TODO: is TopWindow actually the parent and we don't need to pass it?
    def __init__(self, parent):
        wx.Frame.__init__(self, wx.GetApp().TopWindow, -1, "Movie info", size=(400,400)) #init parent class
        self.infotext=''
        self.text=wx.TextCtrl(self,-1, self.infotext, style=wx.TE_MULTILINE)
        self.text.SetEditable(False)
        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.text,5, wx.EXPAND|wx.ALL,5)
        self.SetSizer(sizer)
        self.parent=parent


    def Update(self,text=''):
        """Reads in and displays current set of parameters from parent."""
        paras=self.parent.parameters
        if text=="":
            self.infotext=""
            for k in paras.keys():
                self.infotext+="%s: %s\n"%(k,paras[k])
            self.infotext=self.infotext[:-1]
        else: self.infotext=text
        self.text.SetValue(self.infotext)

#source:
#http://matplotlib.org/examples/user_interfaces/embedding_in_wx2.html
class StackWin(wx.Frame):
    def __init__(self,parent):
        wx.Frame.__init__(self,parent,-1, 'Stack plot',size=(550,350))
        self.parent=parent
        self.SetBackgroundColour(wx.NamedColor("WHITE"))

        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.axes = Axes3D(self.figure)

        try:
            data=np.loadtxt(self.parent.movie.datadir+'coords.txt')
            xsc,ysc,zsc=self.parent.movie.parameters['xscale'],self.parent.movie.parameters['yscale'],self.parent.movie.parameters['zscale']
            xs=data[:,3]*xsc
            ys=data[:,4]*ysc
            zs=data[:,0]*zsc
            ss=data[:,2]*xsc/72.
            self.axes.scatter(xs, ys, zs,s=ss)
            scaling = np.array([getattr(self.axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            self.axes.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
        except:
            print "sorry, plot failed! Is there a coordinate file?"


        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        self.add_toolbar() # comment this out for no toolbar


    def add_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        if wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()


    def OnPaint(self, event):
        self.canvas.draw()


class HistoWin(wx.Frame):
    """Window displaying RGB histogram of current image in semilog y. """
    def __init__(self,parent,image):
        wx.Frame.__init__(self, wx.GetApp().TopWindow, -1, "Image histogram", size=(600,400)) #init parent class
        if len(image.shape)==2: image=np.dstack((image,image,image)) #in case this isn't 3 channels already (e.g. greyscale)
        if len(image.shape)==3 and image.shape[2]!=3: image=np.dstack((image[:,:,0],image[:,:,0],image[:,:,0]))
        buf=io.BytesIO() #
        pl.figure(figsize=[6,4])
        for i in range(3):
            pl.hist(image[:,:,i].flatten(), bins=256, log=True, histtype='step',align='mid',color='rgb'[i])
        pl.savefig(buf,dpi=100,format='png')
        buf.seek(0)
        im=wx.ImageFromStream(buf)
        self.plot=wx.StaticBitmap(self,bitmap=wx.BitmapFromImage(im),size=(600,400),pos=(0,0))

    def Update(self,parent,image):
        """Updates histogram with data from currently displayed image."""
        pl.close("all")
        buf=io.BytesIO()
        pl.figure(figsize=[6,4])
        if len(image.shape)==2: image=np.dstack((image,image,image))
        if len(image.shape)==3 and image.shape[2]!=3: image=np.dstack((image[:,:,0],image[:,:,0],image[:,:,0]))
        for i in range(3):
            pl.hist(image[:,:,i].flatten(), bins=np.arange(256), log=True, histtype='step',align='mid', color='rgb'[i])
        pl.xlim(0,255)
        pl.savefig(buf,dpi=100,format='png')
        buf.seek(0)
        im=wx.ImageFromStream(buf)
        self.plot.SetBitmap(wx.BitmapFromImage(im))


class MyPanel(scp.ScrolledPanel):
    """Scrolled panel containing movie frame image."""
    def __init__(self, parent):
        scp.ScrolledPanel.__init__(self,parent) #init parent class
        self.SetScrollRate(20,20)
        self.EnableScrolling(True,True)
        self.parent=parent
        self.im=MyImage(self)
        imbox=wx.BoxSizer(wx.HORIZONTAL)
        imbox.Add(self.im)
        self.SetSizer(imbox)
        self.im.SetCursor(wx.StockCursor(wx.CURSOR_CROSS))


class MyImage(wx.StaticBitmap):
    """Image displaying current movie frame (24 bit RGB, scalable, zoomable)."""
    def __init__(self, parent):
        self.parent=parent #stuff needed before parent initialisation
        self.pparent=parent.parent
        col=self.parent.GetBackgroundColour()
        self.scale=1. #implements zooom control in parent
        wx.StaticBitmap.__init__(self, parent,-1,GetBitmap(colour=col), (5,5)) #init parent class
        self.axes=[]
        self.points=[]
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)
        self.Bind(wx.EVT_RIGHT_UP, self.OnRightUp)
        self.savestatus=wx.GetTopLevelParent(self).sb.GetStatusText(0)
        self.savept=(0,0)

    def OnLeftDown(self,event):
        ctrlstate=wx.GetKeyState(wx.WXK_CONTROL)
        if not ctrlstate:
            pt=event.GetPosition()
            pt=self.parent.CalcUnscrolledPosition(pt)
            self.savestatus=self.pparent.sb.GetStatusText(0)
            if len(self.pparent.images[self.pparent.imType].shape)==2:
                RGB=", grey %d"%self.pparent.images[self.pparent.imType][pt[1],pt[0]]
            else:
                RGB=", RGB (%d,%d,%d)"%tuple(self.pparent.images[self.pparent.imType][pt[1],pt[0],:])
            self.pparent.sb.SetStatusText("x %d, y %d"%(pt.x/self.scale,pt.y/self.scale)+RGB, 0)
        else:
            pt=event.GetPosition()
            pt=self.parent.CalcUnscrolledPosition(pt)
            self.savestatus=self.pparent.sb.GetStatusText(0)
            if len(self.pparent.images[self.pparent.imType].shape)==2:
                RGB=", grey %d"%self.pparent.images[self.pparent.imType][pt[1],pt[0]]
            else:
                RGB=", RGB (%d,%d,%d)"%tuple(self.pparent.images[self.pparent.imType][pt[1],pt[0],:])
            self.pparent.sb.SetStatusText("x %d, y %d with control"%(pt.x/self.scale,pt.y/self.scale)+RGB, 0)


    def OnLeftUp(self,event):
        self.pparent.sb.SetStatusText(self.savestatus, 0)

    def OnRightDown(self,event):
        if self.pparent.movie.typ=="3D stack":
            pt=event.GetPosition()
            self.savept=self.parent.CalcUnscrolledPosition(pt)


    def OnRightUp(self,event):
        if self.pparent.movie.typ=="3D stack":
            oldcrop=self.pparent.movie.parameters['crop']
            pt=event.GetPosition()
            pt=self.parent.CalcUnscrolledPosition(pt)
            self.pparent.movie.parameters['crop']=[oldcrop[1]+int(self.savept[1]/self.scale),oldcrop[0]+int(self.savept[0]/self.scale),
                oldcrop[1]+int(pt[1]/self.scale),oldcrop[0]+int(pt[0]/self.scale)]
            self.pparent.parameters['crop']=self.pparent.movie.parameters['crop']
            self.pparent.stCropContr.SetValue(str(self.pparent.movie.parameters['crop'])[1:-1])
            self.pparent.StImgDisplay()


    def Redraw(self):
        """Actually display an image. Accepts both filename (no existence/file type check, though) as string or imge as numpy array."""
        image=self.pparent.images[self.pparent.imType] #types: orig, bg, subtracted, threshold, orig with particles.
        if image !='':
            if type(image).__name__=='ndarray':
                if len(image.shape)==2: image=np.dstack((image,image,image))
                im = wx.EmptyImage(image.shape[1],image.shape[0])
                im.SetData(image.astype(np.uint8).tostring())
            if type(image).__name__=='str':
                im=wx.Image(image)
            im.Rescale(im.GetSize()[0]*self.scale,im.GetSize()[1]*self.scale)
            bit=wx.BitmapFromImage(im)
            #ds=wx.GetDisplaySize()
            ds=wx.Display(0).GetGeometry().GetSize()
            ws=(im.GetSize()[0]+120,im.GetSize()[1]+300)
            if ws[0]<ds[0] and ws[1]<ds[1]:
                winsize=ws
            else:
                winsize=ds
            self.pparent.SetSize(winsize)
            self.SetBitmap(bit)


class MyFrame(wx.Frame):
    """Main window of movie analysis GUI"""
    def __init__(self):
        wx.Frame.__init__(self, None, -1, "Particle detection parameters", size=(1024,768))
        #buttons, radio buttons and stuff.
        self.sb = self.CreateStatusBar(2)
        self.scp=MyPanel(self)
        paraPanel=wx.Panel(self)
        buttonPanel=wx.Panel(self)
        nbPanel = wx.Panel(self)

        threshLabel=wx.StaticText(paraPanel,-1,'Threshold')
        self.threshContr=wx.TextCtrl(paraPanel,200,'',size=(50,-1),style=wx.TE_PROCESS_ENTER)
        BGrngLabel=wx.StaticText(paraPanel,-1,'BG range')
        self.BGrngContr=wx.TextCtrl(paraPanel,-1,'120,155',size=(50,-1),style=wx.TE_PROCESS_ENTER)
        strLabel=wx.StaticText(paraPanel,-1,'Kernel size')
        self.strContr=wx.TextCtrl(paraPanel,201,'',size=(50,-1), style=wx.TE_PROCESS_ENTER)
        self.frameSldr = wx.Slider(paraPanel,202,value=0, minValue=0, maxValue=100, style=wx.SL_HORIZONTAL)
        self.fwdB=wx.Button(paraPanel,203,">",size=(30,-1))
        self.backB=wx.Button(paraPanel,204,"<",size=(30,-1))
        self.frameContr=wx.TextCtrl(paraPanel,205,'0',size=(60,-1),style=wx.TE_PROCESS_ENTER)
        psizeLabel=wx.StaticText(paraPanel,-1,'Part. size')
        self.psizeContr=wx.TextCtrl(paraPanel,206,'',size=(80,-1), style=wx.TE_PROCESS_ENTER)
        self.sizeCheck=wx.CheckBox(paraPanel, 211, label='Draw sizes')
        self.invCheck=wx.CheckBox(paraPanel, 212, label='Invert')
        self.maskCheck=wx.CheckBox(paraPanel, 213, label='Mask')
        self.diskfitCheck=wx.CheckBox(paraPanel, 214, label='Fit disk')
        channelCBlabel=wx.StaticText(paraPanel,-1,'RGB channel')
        self.channelCB=wx.ComboBox(paraPanel, 207, choices=['0','1','2'], style=wx.CB_READONLY,size=(50,-1))
        self.channelCB.SetValue('0')
        frameminmaxLabel=wx.StaticText(paraPanel,-1,'Range')
        self.frameMinMaxContr=wx.TextCtrl(paraPanel,-1,'',size=(60,-1),style=wx.TE_PROCESS_ENTER)
        framespacLabel=wx.StaticText(paraPanel,-1,'Frame spacing')
        self.frameSpacContr=wx.TextCtrl(paraPanel,210,'',size=(40,-1),style=wx.TE_PROCESS_ENTER)
        sphericityLabel=wx.StaticText(paraPanel,-1,'Sphericity')
        self.sphericityContr=wx.TextCtrl(paraPanel,208,'-1.',size=(40,-1), style=wx.TE_PROCESS_ENTER)
        blurLabel=wx.StaticText(paraPanel,-1,'Blur')
        self.blurContr=wx.TextCtrl(paraPanel,209,'',size=(50,-1), style=wx.TE_PROCESS_ENTER)

        savePsB=wx.Button(buttonPanel,100,"Save parameters...",size=(140,-1))
        readPsB=wx.Button(buttonPanel,101,"Read parameters...",size=(140,-1))
        paraB=wx.Button(buttonPanel,103,"Show parameters...",size=(140,-1))
        histoB=wx.Button(buttonPanel,104,"Histogram...",size=(140,-1))
        expImB=wx.Button(buttonPanel,108,"Export image...",size=(140,-1))
        self.zoomBox = wx.ComboBox(buttonPanel, 400, choices=['20%','50%','100%','150%','200%','300%','400%'], size=(140,-1))
        self.zoomBox.SetValue('100%')

        #note: any control that will be accessed from inside a method needs the "self." prefix to make it available within the scope of the entire class.
        #e.g. the two following buttons are disabled/enabled during movie processing.

        self.nb = wx.Notebook(nbPanel)

        parttab=wx.Panel(self.nb)
        openMovB=wx.Button(parttab,102,"Open movie...",size=(140,-1))
        self.getTrajB=wx.Button(parttab,105,"Get trajectories",size=(140,-1))
        self.getCoordB=wx.Button(parttab,106,"Get coordinates",size=(140,-1))
        self.getBgB=wx.Button(parttab,107,"Get background",size=(140,-1))
        self.rImstate = [wx.RadioButton(parttab, 300, label='Original',style=wx.RB_GROUP),
        wx.RadioButton(parttab, 301, label='Single channel'),
        wx.RadioButton(parttab, 302, label='Background'),
        wx.RadioButton(parttab, 303, label='Mask'),
        wx.RadioButton(parttab, 304, label='BG treated'),
        wx.RadioButton(parttab, 305, label='Threshold'),
        wx.RadioButton(parttab, 306, label='Particles')]
        self.rImstate[0].SetValue(True)

        #next tab here!
        clustertab=wx.Panel(self.nb)
        openClMovB=wx.Button(clustertab,550,"Open movie...",size=(140,-1))
        self.getCluB=wx.Button(clustertab,501,"Get clusters",size=(140,-1))
        self.convTrajCluB=wx.Button(clustertab,502,"Convert trajectories...",size=(140,-1))
        self.voroCheck=wx.CheckBox(clustertab, -1, label='Voronoi')
        self.clustNumCheck=wx.CheckBox(clustertab, -1, label='Label clusters')
        self.clustNumCheck.SetValue(True)
        self.cImstate = [wx.RadioButton(clustertab, 520, label='Original',style=wx.RB_GROUP),
        wx.RadioButton(clustertab, 521, label='Single channel'),
        wx.RadioButton(clustertab, 522, label='Blur'),
        wx.RadioButton(clustertab, 523, label='Mask'),
        wx.RadioButton(clustertab, 524, label='Threshold'),
        wx.RadioButton(clustertab, 525, label='Clusters'),
        wx.RadioButton(clustertab, 526, label='Voronoi')]
        self.cImstate[0].SetValue(True)
        if rt.vorflag:
            self.voroCheck.SetValue(True)
        else:
            self.voroCheck.SetValue(False)
            self.voroCheck.Disable()
            self.cImstate[-1].Disable()

        stacktab=wx.Panel(self.nb)
        openStackB=wx.Button(stacktab,650,"Open 3D stack...",size=(140,-1))
        stCropLabel=wx.StaticText(stacktab,-1,'crop l,b,r,t')
        self.sImstate = [wx.RadioButton(stacktab, 620, label='Original',style=wx.RB_GROUP),
        wx.RadioButton(stacktab, 621, label='Single channel'),
        wx.RadioButton(stacktab, 622, label='Threshold'),
        wx.RadioButton(stacktab, 623, label='Particles')]
        self.stCropContr=wx.TextCtrl(stacktab,651,'',size=(140,-1), style=wx.TE_PROCESS_ENTER)
        stackResetCropB=wx.Button(stacktab,652,"Reset crop",size=(140,-1))
        self.getStCoordB=wx.Button(stacktab,653,"Get coordinates",size=(140,-1))
        plotStackB=wx.Button(stacktab,654,"Plot stack...",size=(140,-1))


        #setting up the window layout with tons of nested sizers.
        hboxBig=wx.BoxSizer(wx.HORIZONTAL)
        vboxLeft=wx.BoxSizer(wx.VERTICAL)
        vboxRight=wx.BoxSizer(wx.VERTICAL)

        vboxLeft.Add(self.scp,1, wx.EXPAND|wx.ALL,5)

        vboxPara=wx.BoxSizer(wx.VERTICAL)
        hboxPara1=wx.BoxSizer(wx.HORIZONTAL)
        hboxPara1.Add(threshLabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(self.threshContr, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(BGrngLabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(self.BGrngContr, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(strLabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(self.strContr, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(psizeLabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(self.psizeContr, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(sphericityLabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(self.sphericityContr, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(blurLabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara1.Add(self.blurContr, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxPara.Add(hboxPara1,0,wx.EXPAND)
        hboxPara2=wx.BoxSizer(wx.HORIZONTAL)
        hboxPara2.Add(self.sizeCheck, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara2.Add(self.invCheck, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara2.Add(self.maskCheck, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara2.Add(self.diskfitCheck, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara2.Add(channelCBlabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        hboxPara2.Add(self.channelCB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxPara.Add(hboxPara2,0,wx.EXPAND)
        vboxPara.Add(self.frameSldr, 0, wx.ALIGN_BOTTOM|wx.ALL|wx.EXPAND,5)
        hboxFrames=wx.BoxSizer(wx.HORIZONTAL)
        hboxFrames.Add(self.backB, 0, wx.ALIGN_CENTER|wx.ALL,5)
        hboxFrames.Add(self.frameContr, 0,wx.ALIGN_CENTER|wx.ALL,5)
        hboxFrames.Add(self.fwdB, 0, wx.ALIGN_CENTER|wx.ALL,5)
        hboxFrames.Add(frameminmaxLabel, 0, wx.ALIGN_RIGHT|wx.ALL,5)
        hboxFrames.Add(self.frameMinMaxContr, 0, wx.ALIGN_RIGHT|wx.ALL,5)
        hboxFrames.Add(framespacLabel, 0, wx.ALIGN_RIGHT|wx.ALL,5)
        hboxFrames.Add(self.frameSpacContr, 0, wx.ALIGN_RIGHT|wx.ALL,5)
        vboxPara.Add(hboxFrames,0,wx.EXPAND)
        paraPanel.SetSizer(vboxPara)
        vboxLeft.Add(paraPanel,0, wx.ALIGN_BOTTOM|wx.ALL,5)

        vboxMov=wx.BoxSizer(wx.VERTICAL)
        vboxMov.Add(self.zoomBox, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxMov.Add(savePsB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxMov.Add(readPsB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxMov.Add(paraB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxMov.Add(histoB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxMov.Add(expImB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        buttonPanel.SetSizer(vboxMov)
        vboxRight.Add(buttonPanel,0, wx.ALIGN_RIGHT|wx.ALL,5)

        vboxNB=wx.BoxSizer(wx.VERTICAL)

        vboxPart=wx.BoxSizer(wx.VERTICAL)
        vboxPart.Add(openMovB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        sbIm = wx.StaticBox(parttab, label="Image display")
        sbsizerIm = wx.StaticBoxSizer(sbIm, wx.VERTICAL)
        for i in range(7): sbsizerIm.Add(self.rImstate[i], 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxPart.Add(sbsizerIm)
        sbAna = wx.StaticBox(parttab, label="Movie analysis")
        sbsizerAna = wx.StaticBoxSizer(sbAna, wx.VERTICAL)
        sbsizerAna.Add(self.getTrajB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        sbsizerAna.Add(self.getCoordB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        sbsizerAna.Add(self.getBgB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxPart.Add(sbsizerAna)

        parttab.SetSizer(vboxPart)
        self.nb.AddPage(parttab,'Particles')

        vboxClusters=wx.BoxSizer(wx.VERTICAL)
        vboxClusters.Add(openClMovB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        sbCl = wx.StaticBox(clustertab, label="Image display")
        sbsizerCl = wx.StaticBoxSizer(sbCl, wx.VERTICAL)
        for but in self.cImstate: sbsizerCl.Add(but, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxClusters.Add(sbsizerCl)
        sbClAn = wx.StaticBox(clustertab, label="Movie analysis")
        sbsizerClAn = wx.StaticBoxSizer(sbClAn, wx.VERTICAL)
        sbsizerClAn.Add(self.getCluB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        sbsizerClAn.Add(self.convTrajCluB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        sbsizerClAn.Add(self.voroCheck, 0, wx.ALIGN_LEFT|wx.ALL,5)
        sbsizerClAn.Add(self.clustNumCheck, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxClusters.Add(sbsizerClAn)

        clustertab.SetSizer(vboxClusters)
        self.nb.AddPage(clustertab,'Clusters')

        vboxStack=wx.BoxSizer(wx.VERTICAL)
        vboxStack.Add(openStackB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxStack.Add(stCropLabel, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxStack.Add(stackResetCropB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxStack.Add(self.stCropContr, 0, wx.ALIGN_LEFT|wx.ALL,5)

        stacktab.SetSizer(vboxStack)
        sbSt = wx.StaticBox(stacktab, label="Image display")
        sbsizerSt = wx.StaticBoxSizer(sbSt, wx.VERTICAL)
        for but in self.sImstate: sbsizerSt.Add(but, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxStack.Add(sbsizerSt)
        vboxStack.Add(plotStackB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        vboxStack.Add(self.getStCoordB, 0, wx.ALIGN_LEFT|wx.ALL,5)
        self.nb.AddPage(stacktab,'3D stack')

        vboxNB.Add(self.nb)
        nbPanel.SetSizer(vboxNB)

        vboxRight.Add(nbPanel,0,wx.ALIGN_RIGHT)

        hboxBig.Add(vboxLeft,1,wx.EXPAND)
        hboxBig.Add(vboxRight,0,wx.ALIGN_RIGHT)
        self.SetSizer(hboxBig)

        #bind button/input events to class methods.
        self.Bind(wx.EVT_BUTTON, self.SaveParas, id=100)
        self.Bind(wx.EVT_BUTTON, self.ReadParasFromFile, id=101)
        self.Bind(wx.EVT_BUTTON, self.OpenMovie, id=102)
        self.Bind(wx.EVT_BUTTON, self.OpenClMovie, id=550)
        self.Bind(wx.EVT_BUTTON, self.OpenStackMovie, id=650)
        self.Bind(wx.EVT_BUTTON, self.ShowParas, id=103)
        self.Bind(wx.EVT_BUTTON, self.ShowHistogram, id=104)
        self.Bind(wx.EVT_BUTTON, self.GetTrajectories, id=105)
        self.Bind(wx.EVT_BUTTON, self.GetCoordinates, id=106)
        self.Bind(wx.EVT_BUTTON, self.GetBG, id=107)
        self.Bind(wx.EVT_BUTTON, self.ExportImage, id=108)
        # a number of controls handled by the same method.
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=200)
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=201)
        self.Bind(wx.EVT_SLIDER, self.ReadParas, id=202)
        self.Bind(wx.EVT_BUTTON, self.ReadParas, id=203)
        self.Bind(wx.EVT_BUTTON, self.ReadParas, id=204)
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=205)
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=206)
        self.Bind(wx.EVT_COMBOBOX, self.ReadParas, id=207)
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=208)
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=209)
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=210)
        self.Bind(wx.EVT_CHECKBOX, self.ReadParas, id=211)
        self.Bind(wx.EVT_CHECKBOX, self.ReadParas, id=212)
        self.Bind(wx.EVT_CHECKBOX, self.ReadParas, id=213)
        self.Bind(wx.EVT_CHECKBOX, self.ReadParas, id=214)
        self.Bind(wx.EVT_TEXT_ENTER, self.ReadParas, id=651)
        # bindings handling the image display (groupof radio buttons)
        self.Bind(wx.EVT_BUTTON, self.GetClusters, id=501)
        self.Bind(wx.EVT_BUTTON, self.ConvClustTraj, id=502)
        self.Bind(wx.EVT_BUTTON, self.ResetCrop, id=652)
        self.Bind(wx.EVT_BUTTON, self.GetCoordinates, id=653)
        self.Bind(wx.EVT_BUTTON, self.PlotStack, id=654)
        for i in range(300,307): self.Bind(wx.EVT_RADIOBUTTON, self.ImgDisplay, id=i)
        for i in range(521,527): self.Bind(wx.EVT_RADIOBUTTON, self.ClImgDisplay, id=i)
        for i in range(621,623): self.Bind(wx.EVT_RADIOBUTTON, self.StImgDisplay, id=i)
        self.Bind(wx.EVT_COMBOBOX, self.Zoom, id=400)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_CHAR_HOOK,self.OnKeyDown) #Handles all key presses!

        #parameters and definitions. We store everything as attributes of the main window.
        self.imType='Original'
        self.images={'Original':'','Single channel':'','Background':'', 'BG treated':'', 'Mask':'','Threshold':'', 'Particles':''}
        self.moviefile=''
        self.movie=rt.nomovie()
        self.framenum=0
        self.parameters={
            'framerate':0.,'sphericity':-1.0,'xscale':1.0,'yscale':1.0,'zscale':1.0,
            'imsize':(0,0),'blobsize':(5,90),'crop':[0]*4, 'framelim':(0,0), 'circle':[0,0,1e4],
            'frames':0,  'threshold':120, 'struct':5,  'channel':0, 'blur':1,'spacing':1, 'imgspacing':-1,'maxdist':-1,'lossmargin':10, 'lenlim':1,
            'sizepreview':True, 'invert':False, 'diskfit':False, 'mask':True
        }
        #erosion/dilation kernel. basically, a circle of radius "struct" as a numpy array.
        if self.parameters['struct']>0: self.kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.parameters['struct'],self.parameters['struct']))
        else: self.kernel=False
        self.strContr.SetValue(str(self.parameters['struct']))
        self.threshContr.SetValue(str(self.parameters['threshold']))
        self.psizeContr.SetValue(str(self.parameters['blobsize'])[1:-1])
        self.frameMinMaxContr.SetValue("%d,%d"%self.parameters['framelim'])
        self.frameSpacContr.SetValue("%d"%(self.parameters['spacing']))
        self.blurContr.SetValue(str(self.parameters['blur']))
        self.sizeCheck.SetValue(self.parameters['sizepreview'])
        self.maskCheck.SetValue(self.parameters['mask'])
        self.invCheck.SetValue(self.parameters['invert'])
        self.diskfitCheck.SetValue(self.parameters['diskfit'])

        self.cdir=moviedir

    def OnKeyDown(self,event):
        key=event.GetKeyCode()
        ctrlstate=wx.GetKeyState(wx.WXK_CONTROL)
        if ctrlstate:
            if key==wx.WXK_LEFT:
                event.SetId(-1)
                self.framenum-=1
                self.ReadParas(event)
            if key==wx.WXK_RIGHT:
                event.SetId(-1)
                self.framenum+=1
                self.ReadParas(event)
        else: event.Skip()


    def SaveParas(self,event):
        """saves """
        if self.movie.typ!='none':
            self.ShowParas()
            if not os.path.exists(self.movie.datadir): os.mkdir(self.movie.datadir)
            try:
                with open(self.movie.datadir+'paras.txt','w') as f: f.write(self.infoWin.infotext)
            except:
                print "Unexpected error:", exc_info()[0]
                pass

    def OpenMovie(self,event=None):
        dlg = wx.FileDialog(self, "Choose image", self.cdir, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.moviefile=dlg.GetPath()
            print self.moviefile
            self.cdir=os.path.dirname(dlg.GetPath())+os.sep
            print self.cdir
            if os.name=='posix': #this assumes you installed mplayer!
                with open(os.path.abspath(__file__), 'r') as f:
                    text=f.read()
                    text=re.sub("(?<=\nmoviedir=').*?(?='#end)",self.cdir,text)
                with open(os.path.abspath(__file__), 'w') as f:
                    f.write(text)

                #print ' '.join(['mplayer','-vo','null','-ao','null','-identify','-frames','0',self.moviefile])
                result = subprocess.check_output(['mplayer','-vo','null','-ao','null','-identify','-frames','0',self.moviefile])
            if os.name=='nt': #this assumes you installed mplayer and have the folder in your PATH!
                result = subprocess.check_output(['mplayer.exe','-vo','null','-ao', 'null','-identify','-frames','0',self.moviefile])
            try:
                self.parameters['imsize']=(int(re.search('(?<=ID_VIDEO_WIDTH=)[0-9]+',result).group()),int(re.search('(?<=ID_VIDEO_HEIGHT=)[0-9]+',result).group()))
                self.parameters['framerate']=float(re.search('(?<=ID_VIDEO_FPS=)[0-9.]+',result).group())
                self.parameters['frames']=int(round(float(re.search('(?<=ID_LENGTH=)[0-9.]+',result).group())*self.parameters['framerate']))
                self.parameters['framelim']=(0,self.parameters['frames'])
            except:
                self.parameters['imsize']=(0,0)
                self.parameters['framerate']=1.
                self.parameters['frames']=0
                self.parameters['framelim']=(0,0)
            self.movie=rt.movie(self.moviefile)
            self.images['Original']=self.movie.getFrame(self.framenum)
            self.frameSldr.SetMin(0)
            self.frameSldr.SetMax(self.parameters['frames'])
            self.frameSldr.SetValue(0)
            self.frameContr.SetValue('0')
            self.framenum=0
            self.zoomBox.SetValue('100%')
            self.scp.im.scale=1.
            image=self.images['Original']
            if type(image).__name__=='ndarray':
                im = wx.EmptyImage(image.shape[1],image.shape[0])
                im.SetData(image.tostring())
            elif type(image).__name__=='str':
                im=wx.Image(image)
            else:
                im=np.zeros(self.parameters['imsize'])
            im.Rescale(im.GetSize()[0]*self.scp.im.scale,im.GetSize()[1]*self.scp.im.scale)
            #ds=wx.GetDisplaySize()
            ds=wx.Display(0).GetGeometry().GetSize()
            ws=(im.GetSize()[0]+120,im.GetSize()[1]+300)
            if ws[0]<ds[0] and ws[1]<ds[1]:
                winsize=ws
            else:
                winsize=ds
                self.SetSize(winsize)
            self.scp.im.SetBitmap(wx.BitmapFromImage(im))
            self.scp.im.points=[]
            self.scp.im.axes=[]
            self.frameMinMaxContr.SetValue("%d,%d"%self.parameters['framelim'])
            if os.path.exists(self.movie.datadir+'paras.txt'): self.ReadParasFromFile(filename=self.movie.datadir+'paras.txt')
            f = self.sb.GetFont()
            dc = wx.WindowDC(self.sb)
            dc.SetFont(f)
            width, height = dc.GetTextExtent(self.moviefile)
            self.sb.SetStatusWidths([winsize[0]-width-50, width+40])
            self.sb.SetStatusText(self.moviefile, 1)


    def OpenStackMovie(self,event=None):
        dlg = wx.FileDialog(self, "Select image", self.cdir, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.moviefile=dlg.GetPath()
            self.cdir=os.path.dirname(dlg.GetPath())+os.sep
            print self.cdir
            if os.name=='posix':
                with open(os.path.abspath(__file__), 'r') as f:
                    text=f.read()
                    text=re.sub("(?<=\nmoviedir=').*?(?='#end)",self.cdir,text)
                with open(os.path.abspath(__file__), 'w') as f:
                    f.write(text)
            self.movie=rt.imStack(self.moviefile)
            self.images['Original']=cv2.imread(self.movie.stack[0],1)
            self.parameters=self.movie.parameters
            self.frameSldr.SetMin(0)
            self.frameSldr.SetMax(self.parameters['frames'])
            self.frameSldr.SetValue(0)
            self.frameContr.SetValue('0')
            self.stCropContr.SetValue(str(self.movie.parameters['crop'])[1:-1])
            self.framenum=0
            self.zoomBox.SetValue('100%')
            self.scp.im.scale=1.
            image=self.images['Original']
            if type(image).__name__=='ndarray':
                im = wx.EmptyImage(image.shape[1],image.shape[0])
                im.SetData(image.tostring())
            if type(image).__name__=='str':
                im=wx.Image(image)
            im.Rescale(im.GetSize()[0]*self.scp.im.scale,im.GetSize()[1]*self.scp.im.scale)
            #ds=wx.GetDisplaySize()
            ds=wx.Display(0).GetGeometry().GetSize()
            ws=(im.GetSize()[0]+120,im.GetSize()[1]+200)
            if ws[0]<ds[0] and ws[1]<ds[1]:
                winsize=ws
            else:
                winsize=ds
            self.SetSize(winsize)
            self.scp.im.SetBitmap(wx.BitmapFromImage(im))
            self.scp.im.points=[]
            self.scp.im.axes=[]
            self.frameMinMaxContr.SetValue("%d,%d"%self.parameters['framelim'])
            if os.path.exists(self.movie.datadir+'paras.txt'): self.ReadParasFromFile(filename=self.movie.datadir+'paras.txt')
            f = self.sb.GetFont()
            dc = wx.WindowDC(self.sb)
            dc.SetFont(f)
            width, height = dc.GetTextExtent(self.moviefile)
            self.sb.SetStatusWidths([winsize[0]-width-50, width+40])
            self.sb.SetStatusText(self.moviefile, 1)


    def OpenClMovie(self,event=None):
        dlg = wx.FileDialog(self, "Choose image", self.cdir, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.moviefile=dlg.GetPath()
            self.cdir=os.path.dirname(dlg.GetPath())+os.sep
            print self.cdir
            if os.name=='posix': #this assumes you installed mplayer!
                with open(os.path.abspath(__file__), 'r') as f:
                    text=f.read()
                    text=re.sub("(?<=\nmoviedir=').*?(?='#end)",self.cdir,text)
                with open(os.path.abspath(__file__), 'w') as f:
                    f.write(text)
                #print ' '.join(['mplayer','-vo','null','-ao','null','-identify','-frames','0',self.moviefile])
                result = subprocess.check_output(['mplayer','-vo','null','-ao','null','-identify','-frames','0',self.moviefile])
            if os.name=='nt': #this assumes you installed mplayer and have the folder in your PATH!
                result = subprocess.check_output(['mplayer.exe','-vo','null','-ao', 'null','-identify','-frames','0',self.moviefile])
            self.parameters['imsize']=(int(re.search('(?<=ID_VIDEO_WIDTH=)[0-9]+',result).group()),int(re.search('(?<=ID_VIDEO_HEIGHT=)[0-9]+',result).group()))
            self.parameters['framerate']=float(re.search('(?<=ID_VIDEO_FPS=)[0-9.]+',result).group())
            self.parameters['frames']=int(round(float(re.search('(?<=ID_LENGTH=)[0-9.]+',result).group())*self.parameters['framerate']))
            self.parameters['framelim']=(0,self.parameters['frames'])
            self.frameSldr.SetMin(0)
            self.frameSldr.SetMax(self.parameters['frames'])
            self.frameSldr.SetValue(0)
            self.frameContr.SetValue('0')
            self.framenum=0
            self.movie=rt.clusterMovie(self.moviefile)
            self.images['Original']=self.movie.getFrame(self.framenum)
            self.zoomBox.SetValue('100%')
            self.scp.im.scale=1.
            image=self.images['Original']
            if type(image).__name__=='ndarray':
                im = wx.EmptyImage(image.shape[1],image.shape[0])
                im.SetData(image.tostring())
            if type(image).__name__=='str':
                im=wx.Image(image)
            im.Rescale(im.GetSize()[0]*self.scp.im.scale,im.GetSize()[1]*self.scp.im.scale)
            #ds=wx.GetDisplaySize()
            ds=wx.Display(0).GetGeometry().GetSize()
            ws=(im.GetSize()[0]+120,im.GetSize()[1]+200)
            if ws[0]<ds[0] and ws[1]<ds[1]:
                winsize=ws
            else:
                winsize=ds
            self.SetSize(winsize)
            self.scp.im.SetBitmap(wx.BitmapFromImage(im))
            self.scp.im.points=[]
            self.scp.im.axes=[]
            self.frameMinMaxContr.SetValue("%d,%d"%self.parameters['framelim'])
            if os.path.exists(self.movie.datadir+'paras.txt'): self.ReadParasFromFile(filename=self.movie.datadir+'paras.txt')
            f = self.sb.GetFont()
            dc = wx.WindowDC(self.sb)
            dc.SetFont(f)
            width, height = dc.GetTextExtent(self.moviefile)
            self.sb.SetStatusWidths([winsize[0]-width-50, width+40])
            self.sb.SetStatusText(self.moviefile, 1)


    def ShowParas(self,event=None, text=''):
        try:
            self.infoWin.Update(text)
        except AttributeError:
            self.infoWin=InfoWin(self)
            self.infoWin.Show()
            self.infoWin.Update(text)
        self.infoWin.Raise()


    def ShowHistogram(self,event):
        try:
            self.HistoWin.Update(self, self.images[self.imType])
        except AttributeError:
            self.HistoWin=HistoWin(self, self.images[self.imType])
            self.HistoWin.Show()
        self.HistoWin.Raise()

    def Zoom(self,event=None):
        try:
            sc=float(self.zoomBox.GetValue()[:-1])/100.
            self.scp.im.scale=sc
            image=self.images[self.imType]
            if type(image).__name__=='ndarray':
                im = wx.EmptyImage(image.shape[0],image.shape[1])
                im.SetData(image.tostring())
            if type(image).__name__=='str':
                im=wx.Image(image)
            #ds=wx.GetDisplaySize()
            ds=wx.Display(0).GetGeometry().GetSize()
            ws=(int(im.GetSize()[0]*sc+300),int(im.GetSize()[1]*sc+200))
            self.SetSize((min(ws[0],ds[0]),min(ws[1],ds[1])))
            im.Rescale(im.GetSize()[0]*sc,im.GetSize()[1]*sc)
            self.scp.im.SetBitmap(wx.BitmapFromImage(im))
            self.scp.im.Redraw()

        except ValueError:
            self.zoomBox.SetValue('{0:d}%'.format(int(self.scp.im.scale*100)))


    def ImgDisplay(self,event=None):
        if self.movie.typ!="none":
            self.parameters['channel']=int(self.channelCB.GetValue())
            if self.maskCheck.GetValue():
                try:
                    im=np.array(Image.open(self.movie.datadir+'mask.png'))
                    if len(im.shape)==3: im=im[:,:,self.parameters['channel']]
                    mask=(im>0).astype(float)
                except: mask=np.zeros(self.movie.parameters['imsize'][::-1])+1.
            else: mask=np.zeros(self.movie.parameters['imsize'][::-1])+1.
            self.images['Mask']=mask.astype(np.uint8)*255
            for item in self.rImstate:
                if item.GetValue(): self.imType=item.GetLabelText()
            if self.imType=='Background':
                if type(self.movie.bg).__name__!='ndarray':
                    if os.path.exists(self.movie.datadir+'bg.png'):
                        self.movie.loadBG()
                    else:
                        self.sb.SetStatusText('Working... Extracting background', 1)
                        s=self.BGrngContr.GetValue() #text field
                        BGrng=(int(s.split(',')[0]),int(s.split(',')[1]))
                        num=int(1.e8/(self.movie.parameters['imsize'][0]*self.movie.parameters['imsize'][1]))
                        if num<10: num=10
                        if num>50: num=50
                        print num
                        if BGrng[1]<0: bg=self.movie.getBGold(cutoff=BGrng[1], num=num, spac=int(self.parameters['frames']/(num+1)), prerun=30, save=True,channel=self.parameters['channel'])
                        else: bg=self.movie.getBG(rng=BGrng, num=num, spac=int(self.parameters['frames']/(num+1)), prerun=30, save=True,channel=self.parameters['channel'])
                        self.sb.SetStatusText(self.moviefile, 1)
                self.images['Background']=self.movie.bg
            else:
                if type(self.images['Original']).__name__!='ndarray':
                    image=self.movie.getFrame(self.framenum)
                    self.images['Original']=image.copy()
                else: image=self.images['Original'].copy()
                if len(image.shape)>2:
                    image=image[:,:,self.parameters['channel']]
                    self.images['Single channel'] = image.copy()
                bgsub=image.astype(float)-self.movie.bg
                self.images['BG treated']=rt.mxContr(bgsub)*mask+255*(1-mask)
                thresh=rt.mxContr((self.images['BG treated']<self.parameters['threshold']).astype(int))
                if self.parameters['struct']>0:
                    thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
                if self.invCheck.GetValue(): thresh=255-thresh
                self.images['Threshold']=thresh.copy()
                #contours, hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                self.images['Particles']=self.images['Original'].copy()
                if self.imType=='Particles':
                    blobs,contours=rt.extract_blobs(thresh, -1, self.parameters['blobsize'], self.parameters['sphericity'], diskfit=True,returnCont=True, outpSpac=1) #TODO:why is diskfit hardcoded to True?
                    for b in range(len(blobs)):
                        if blobs[b][-2]==0:
                            if self.diskfitCheck.GetValue(): cv2.circle(self.images['Particles'],(np.int32(blobs[b][3]),np.int32(blobs[b][4])),np.int32(np.sqrt(blobs[b][2]/np.pi)),(255,120,0),2)
                            else:
                                print contours[b]
                                cv2.drawContours(self.images['Particles'],[contours[b]],-1,(0,255,120),2)
                        else:
                            if self.diskfitCheck.GetValue(): cv2.circle(self.images['Particles'],(np.int32(blobs[b][3]),np.int32(blobs[b][4])),np.int32(np.sqrt(blobs[b][2]/np.pi)),(0,255,120),2)
                            else:
                                print contours[b]
                                cv2.drawContours(self.images['Particles'],[contours[b]],-1,(0,255,120),2)
                    contcount=blobs.shape[0]
                    if self.sizeCheck.GetValue():
                        r1=np.ceil(np.sqrt(self.parameters['blobsize'][0]/np.pi))
                        r2=np.ceil(np.sqrt(self.parameters['blobsize'][1]/np.pi))
                        cv2.circle(self.images['Particles'],(np.int32(r2+5),np.int32(r2+5)),np.int32(r1),(255,0,0),-1)
                        cv2.circle(self.images['Particles'],(np.int32(3*r2+10),np.int32(r2+5)),np.int32(r2),(255,0,0),-1)
                    self.sb.SetStatusText("%d particles"%contcount, 0)
            self.scp.im.Redraw()
            try: self.HistoWin.Update(self, self.images[self.imType])
            except AttributeError: pass

    def ClImgDisplay(self,event=None):
        if self.movie.typ!='none':
            self.parameters['channel']=int(self.channelCB.GetValue())
            if self.maskCheck.GetValue():
                try:
                    im=np.array(Image.open(self.movie.datadir+'mask.png'))
                    if len(im.shape)==3: im=im[:,:,self.parameters['channel']]
                    mask=(im>0).astype(float)
                    if self.parameters['circle'][0]==0:
                        th=im.copy().astype(np.uint8)
                        contours, hierarchy=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        (xm,ym),rm=cv2.minEnclosingCircle(contours[0])
                        self.parameters['circle']=[xm,self.parameters['imsize'][1]-ym,rm]
                        print self.parameters['circle']
                except:
                    mask=np.zeros(self.movie.parameters['imsize'][::-1])+1.
            else:
                mask=np.zeros(self.movie.parameters['imsize'][::-1])+1.
            self.images['Mask']=mask.astype(np.uint8)*255
            for item in self.cImstate:
                if item.GetValue(): self.imType=item.GetLabelText()
            if type(self.images['Original']).__name__!='ndarray':
                image=self.movie.getFrame(self.framenum)
                self.images['Original']=image.copy()
            else: image=self.images['Original'].copy()
            if len(image.shape)>2:
                image=image[:,:,self.parameters['channel']]
                self.images['Single channel'] = image.copy()
            blur=rt.mxContr(image.copy())*mask+255*(1-mask)
            blur=cv2.GaussianBlur(blur,(self.parameters['blur'],self.parameters['blur']),0)
            self.images['Blur']=blur.copy()
            thresh=rt.mxContr((blur<self.parameters['threshold']).astype(int))
            if self.invCheck.GetValue(): thresh=255-thresh
            if self.parameters['struct']>0:
                thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
            if self.invCheck.GetValue(): thresh=255-thresh
            self.images['Threshold']=thresh.copy()
            self.images['Clusters']=self.images['Original'].copy()
            self.images['Voronoi']=self.images['Original'].copy()
            if self.imType=='Clusters':
                blobs,contours=rt.extract_blobs(thresh, -1, self.parameters['blobsize'], -1, diskfit=False,returnCont=True, outpSpac=1)
                count = 0
                contcount=blobs.shape[0]
                if self.clustNumCheck.GetValue() and contcount>0:
                    self.ShowParas(text=str(blobs[:,1:]))
                for b in range(len(blobs)):
                    if self.clustNumCheck.GetValue():
                        cv2.putText(self.images['Clusters'],str(count), (int(blobs[count,3]),int(blobs[count,4])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
                    count +=1
                    cv2.drawContours(self.images['Clusters'],[contours[b]],-1,(0,255,120),2)
                contcount=blobs.shape[0]
                self.sb.SetStatusText("%d clusters"%contcount, 0)
                if self.sizeCheck.GetValue():
                    r1=np.ceil(np.sqrt(self.parameters['blobsize'][0]/np.pi))
                    r2=np.ceil(np.sqrt(self.parameters['blobsize'][1]/np.pi))
                    cv2.circle(self.images['Clusters'],(np.int32(r2+5),np.int32(r2+5)),np.int32(r1),(255,0,0),-1)
                    cv2.circle(self.images['Clusters'],(np.int32(3*r2+10),np.int32(r2+5)),np.int32(r2),(255,0,0),-1)
            if self.imType=='Voronoi' and rt.vorflag:
                blobs=rt.extract_blobs(thresh, -1, self.parameters['blobsize'], -1, diskfit=False,returnCont=False, outpSpac=1)
                if blobs.shape[0]>1:
                    newpoints=[]
                    vor=rt.Voronoi(blobs[:,3:5])
                    circ=self.parameters['circle']
                    dists=np.sum((vor.vertices-np.array(circ[:2]))**2,axis=1)-circ[2]**2
                    extinds=[-1]+(dists>0).nonzero()[0]
                    for i in range(blobs.shape[0]):
                        r=vor.regions[vor.point_region[i]]
                        newpoints+=[rt.circle_invert(blobs[i,3:5],circ, integ=True)]
                    pts=np.vstack((blobs[:,3:5],np.array(newpoints)))
                    vor=rt.Voronoi(pts)
                    for i in range(blobs.shape[0]):
                        r=vor.regions[vor.point_region[i]]
                        col=tuple([int(255*c) for c in cm.jet(i*255/blobs.shape[0])])[:3]
                        cv2.polylines(self.images['Voronoi'], [(vor.vertices[r]).astype(np.int32)], True, col[:3], 2)
                        cv2.circle(self.images['Voronoi'], (int(circ[0]),int(circ[1])), int(circ[2]),(0,0,255),2)
            self.scp.im.Redraw()
            try: self.HistoWin.Update(self, self.images[self.imType])
            except AttributeError: pass


    def StImgDisplay(self,event=None):
        if self.movie.typ=='3D stack':
            self.parameters['channel']=int(self.channelCB.GetValue())
            for item in self.sImstate:
                if item.GetValue(): self.imType=item.GetLabelText()
            image=self.movie.getFrame(self.framenum)
            #print self.movie.crop
            if type(image).__name__=='ndarray':
                if image.shape[:2]!=(self.movie.parameters['crop'][2]-self.movie.parameters['crop'][0],self.movie.parameters['crop'][3]-self.movie.parameters['crop'][1]):
                    if len(image.shape)==2: image=image[self.movie.parameters['crop'][0]:self.movie.parameters['crop'][2],self.movie.parameters['crop'][1]:self.movie.parameters['crop'][3]]
                    if len(image.shape)==3: image=image[self.movie.parameters['crop'][0]:self.movie.parameters['crop'][2],self.movie.parameters['crop'][1]:self.movie.parameters['crop'][3],:]
                    self.images['Original']=image.copy()
                if len(image.shape)>2:
                    image=image[:,:,self.parameters['channel']]
                print image.shape, len(image.shape)
                if self.parameters['blur']>1:
                    image=cv2.GaussianBlur(image,(self.parameters['blur'],self.parameters['blur']),0)
                self.images['Single channel'] = image.copy()
                thresh=rt.mxContr((self.images['Single channel']<self.parameters['threshold']).astype(int))
                if self.parameters['struct']>0:
                    thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
                if self.invCheck.GetValue(): thresh=255-thresh
                self.images['Threshold']=thresh.copy()
                self.images['Particles']=self.images['Original'].copy()
                if self.imType=='Particles':
                    if np.amin(thresh)!=np.amax(thresh): blobs,contours=rt.extract_blobs(thresh, -1, self.parameters['blobsize'], self.parameters['sphericity'], diskfit=True,returnCont=True, outpSpac=1)
                    else: blobs,contours=np.array([]).reshape(0,8),[]
                    for b in range(len(blobs)):
                        if blobs[b][-2]==0:
                            if self.diskfitCheck.GetValue():
                                cv2.circle(self.images['Particles'],(np.int32(blobs[b][3]),np.int32(blobs[b][4])),np.int32(np.sqrt(blobs[b][2]/np.pi)),(255,120,0),2)
                            else:
                                #print contours[b]
                                cv2.drawContours(self.images['Particles'],[contours[b]],-1,(0,255,120),2)
                        else:
                            if self.diskfitCheck.GetValue():
                                cv2.circle(self.images['Particles'],(np.int32(blobs[b][3]),np.int32(blobs[b][4])),np.int32(np.sqrt(blobs[b][2]/np.pi)),(0,255,120),2)
                            else:
                                #print contours[b]
                                cv2.drawContours(self.images['Particles'],[contours[b]],-1,(0,255,120),2)
                    contcount=blobs.shape[0]
                    if self.sizeCheck.GetValue():
                        r1=np.ceil(np.sqrt(self.parameters['blobsize'][0]/np.pi))
                        r2=np.ceil(np.sqrt(self.parameters['blobsize'][1]/np.pi))
                        cv2.circle(self.images['Particles'],(np.int32(r2+5),np.int32(r2+5)),np.int32(r1),(255,0,0),-1)
                        cv2.circle(self.images['Particles'],(np.int32(3*r2+10),np.int32(r2+5)),np.int32(r2),(255,0,0),-1)
                    self.sb.SetStatusText("%d particles"%contcount, 0)
                self.scp.im.Redraw()
                try: self.HistoWin.Update(self, self.images[self.imType])
                except AttributeError: pass

    def ExportImage(self,event):
        dlg = wx.FileDialog(self, "Export current image to PNG", self.cdir, "",
        "PNG files (*.png)|*.png", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            Image.fromarray(self.images[self.imType]).convert('RGB').save(dlg.GetPath())


    def ReadParas(self,event):
        evID=event.GetId()
        if evID==200: #binary threshold
            thresh=int(self.threshContr.GetValue())
            if 0<=thresh<=255: self.parameters['threshold']=thresh
        if evID==201: #structuring element size
            self.parameters['struct']=int(self.strContr.GetValue())
            if self.parameters['struct']>0: self.kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.parameters['struct'],self.parameters['struct']))
            else: self.kernel=False
        #frame number
        if evID==202:  self.framenum=self.frameSldr.GetValue() #slider
        if evID==203: self.framenum+=1 #fwd Button
        if evID==204: self.framenum-=1 #back Button
        if evID==205: self.framenum=int(self.frameContr.GetValue()) #text field
        if evID==206:
            s=self.psizeContr.GetValue() #text field
            self.parameters['blobsize']=(int(s.split(',')[0]),int(s.split(',')[1]))
        if evID==207: self.parameters['channel']=int(self.channelCB.GetValue())
        if evID==208: self.parameters['sphericity']=float(self.sphericityContr.GetValue())
        if evID==209:
            self.parameters['blur']=int(self.blurContr.GetValue())
            if self.parameters['blur']%2==0: self.parameters['blur']+=1
        if evID==210:
            self.parameters['spacing']=int(self.frameSpacContr.GetValue())
        if evID in range(211,215):
            self.parameters['sizepreview']=self.sizeCheck.GetValue()
            self.parameters['mask']=self.maskCheck.GetValue()
            self.parameters['invert']=self.invCheck.GetValue()
            self.parameters['diskfit']=self.diskfitCheck.GetValue()
        if evID==651:
            if self.movie.typ=="3D stack":
                try:
                    self.movie.parameters['crop']=[int(i) for i in self.stCropContr.GetValue().split(',')]
                    self.parameters['crop']=self.movie.parameters['crop']
                except: raise
        self.frameContr.SetValue(str(self.framenum))
        self.frameSpacContr.SetValue(str(self.parameters['spacing']))
        self.frameSldr.SetValue(self.framenum)
        self.strContr.SetValue(str(self.parameters['struct']))
        self.threshContr.SetValue(str(self.parameters['threshold']))
        self.sphericityContr.SetValue(str(self.parameters['sphericity']))
        self.blurContr.SetValue(str(self.parameters['blur']))
        if self.movie.typ=="3D stack": self.stCropContr.SetValue(str(self.movie.parameters['crop'])[1:-1])
        self.images['Original']=self.movie.getFrame(self.framenum)
        try: self.infoWin.Update()
        except AttributeError:
            self.infoWin=InfoWin(self)
            self.infoWin.Show()
            self.infoWin.Update()
        if evID>0: self.infoWin.Raise()
        text=self.nb.GetPageText(self.nb.GetSelection())
        if text=='Particles':
            self.ImgDisplay()
        elif text=='Clusters':
            self.ClImgDisplay()
        elif text=='3D stack':
            self.StImgDisplay()
        #self.sSaveProject()

    def ReadParasFromFile(self,event=None, filename=''):
        if os.path.exists(self.movie.datadir): d=self.movie.datadir
        else: d=''
        if filename=='':
            dlg = wx.FileDialog(self, "Choose parameter file", d, 'para.txt',style=wx.OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                filename=dlg.GetPath()
        try:
            with open(filename,'r') as f: text=f.read()
            text=text.split('\n')
            for t in text:
                t=t.split(':')
                if t[0].strip() in ['struct','threshold','frames', 'channel','blur','spacing','imgspacing','maxdist','lossmargin','lenlim']:#integer parameters
                    self.parameters[t[0]]=int(t[1].strip())
                if t[0].strip() in ['blobsize','imsize', 'crop', 'framelim','circle']:#tuple parameters
                    tsplit=re.sub('[\s\[\]\(\)]','',t[1]).split(',')
                    self.parameters[t[0]]=tuple([int(float(it)) for it in tsplit]) #this is a bit of a hack, but strings with dots in them don't convert to int, apparently
                if t[0].strip() in ['framerate','sphericity','xscale','yscale','zscale']:#float parameters
                    self.parameters[t[0]]=float(t[1].strip())
                if t[0].strip() == 'channel':
                    self.channelCB.SetValue(t[1].strip())
                if t[0].strip() in ['sizepreview','mask','diskfit','invert']:#float parameters
                    self.parameters[t[0]]=rt.str_to_bool(t[1].strip())
            self.strContr.SetValue(str(self.parameters['struct']))
            if self.parameters['struct']>0: self.kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.parameters['struct'],self.parameters['struct']))
            else: self.kernel=False
            self.threshContr.SetValue(str(self.parameters['threshold']))
            self.psizeContr.SetValue(str(self.parameters['blobsize']).replace(' ','')[1:-1])
            self.sphericityContr.SetValue("%.2f"%self.parameters['sphericity'])
            self.blurContr.SetValue("%d"%self.parameters['blur'])
            self.frameSpacContr.SetValue("%d"%self.parameters['spacing'])
            self.stCropContr.SetValue(str(self.parameters['crop']).replace(' ','')[1:-1])
            self.maskCheck.SetValue(self.parameters['mask'])
            self.diskfitCheck.SetValue(self.parameters['diskfit'])
            self.sizeCheck.SetValue(self.parameters['sizepreview'])
            self.invCheck.SetValue(self.parameters['invert'])
            self.ShowParas()
        except:
            print "Ooops... Try a different file?"
            self.sb.SetStatusText("Ooops... Try a different file?",0)
            raise


    def GetCoordinates(self,event):
        self.getCoordB.Disable()
        self.sb.SetStatusText("Working... Running coordinate analysis",1)
        try:
            lim=self.frameMinMaxContr.GetValue()
            self.parameters['framelim']=(int(lim.split(',')[0]),int(lim.split(',')[1]))
        except:
            pass
        if self.maskCheck.GetValue():
            try: mask=self.movie.datadir+'mask.png'
            except: mask=False
        else: mask=False
        self.parameters['channel']=int(self.channelCB.GetValue())
        self.movie.extractCoords(framelim=self.parameters['framelim'], blobsize=self.parameters['blobsize'], threshold=self.parameters['threshold'],kernel=self.kernel, delete=True, mask=mask,channel=self.parameters['channel'], sphericity=self.parameters['sphericity'],diskfit=self.diskfitCheck.GetValue(), crop=self.parameters['crop'], invert=self.invCheck.GetValue())
        self.sb.SetStatusText(self.moviefile, 1)
        self.getCoordB.Enable()

    def GetClusters(self,event):
        print self.parameters
        self.getCluB.Disable()
        self.sb.SetStatusText("Working... Running coordinate analysis",1)
        try:
            lim=self.frameMinMaxContr.GetValue()
            self.parameters['framelim']=(int(lim.split(',')[0]),int(lim.split(',')[1]))
        except:
            pass
        if self.maskCheck.GetValue(): mask=self.movie.datadir+'mask.png'
        else: mask=False
        self.parameters['channel']=int(self.channelCB.GetValue())
        self.movie.getClusters(thresh=self.parameters['threshold'],gkern=self.parameters['blur'],clsize=self.parameters['blobsize'],channel=self.parameters['channel'],rng=self.parameters['framelim'],spacing=self.parameters['spacing'], maskfile=self.movie.datadir+'mask.png', circ=self.parameters['circle'],imgspacing=self.parameters['imgspacing'])
        self.sb.SetStatusText(self.moviefile, 1)
        self.getCluB.Enable()

    def ConvClustTraj(self,event):
        if os.path.exists(self.movie.datadir+'clusters.txt'):
            datafile=self.movie.datadir+"clusters.txt"
        else:
            dlg = wx.FileDialog(self, "Select cluster data file", self.cdir, style=wx.OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                datafile=dlg.GetPath()
            else:
                datafile=""
        if datafile!="":
            self.convTrajCluB.Disable()
            self.sb.SetStatusText("Working... Extracting trajectories from coordinates.",1)
            self.movie.CoordtoTraj(tempfile=datafile, lossmargin=1,maxdist=self.parameters['maxdist'],spacing=self.parameters['spacing'])
            self.sb.SetStatusText(self.moviefile, 1)
            self.convTrajCluB.Enable()

    def GetTrajectories(self,event):
        self.getTrajB.Disable()
        self.sb.SetStatusText("Working... Running trajectory analysis",1)
        dlg = wx.FileDialog(self, "Choose coordinate file", self.movie.datadir, style=wx.OPEN)
        flag=False
        fname=''
        if dlg.ShowModal() == wx.ID_OK:
            fname=dlg.GetPath()
            with open(fname,'r') as f: 
                fileheader=f.readline()
                if rt.COORDHEADER==fileheader: flag=True
        else:
            wx.MessageBox('Please provide/generate a coordinate file (via Get Coordinates).', 'Info', wx.OK | wx.ICON_INFORMATION)
        if flag:
            self.movie.CoordtoTraj(tempfile=fname,lenlim=self.parameters['lenlim'], delete=True, breakind=1e9, maxdist=self.parameters['maxdist'], lossmargin=self.parameters['lossmargin'], spacing=self.parameters['spacing'])
        else:
            wx.MessageBox('Please provide/generate a coordinate file (via Get Coordinates).', 'Info', wx.OK | wx.ICON_INFORMATION)
                
        self.getTrajB.Enable()


    def GetBG(self,event):
        s=self.BGrngContr.GetValue() #text field
        BGrng=(int(s.split(',')[0]),int(s.split(',')[1]))
        self.parameters['channel']=int(self.channelCB.GetValue())
        if BGrng[1]<0: bg=self.movie.getBGold(cutoff=BGrng[0], num=40, spac=int(self.parameters['frames']/51), prerun=100, save=True,channel=self.parameters['channel'])
        else: bg=self.movie.getBG(rng=BGrng, num=40, spac=int(self.parameters['frames']/51), prerun=100, save=True,channel=self.parameters['channel'])


    def ResetCrop(self,event):
        if self.movie.typ=='3D stack':
            self.movie.parameters['crop']=[0,0,self.movie.parameters['imsize'][0],self.movie.parameters['imsize'][1]]
            self.parameters['crop']=self.movie.parameters['crop']
            self.stCropContr.SetValue(str(self.movie.parameters['crop'])[1:-1])
            self.StImgDisplay()

    def PlotStack(self,event):
        self.stackwin=StackWin(self)
        self.stackwin.Show()
        self.stackwin.Raise()


    def OnClose(self,event):
        self.Destroy()

app=wx.App(redirect=False)

frame=MyFrame()
frame.Show()
app.MainLoop()
app.Destroy()
