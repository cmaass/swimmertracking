# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:52:06 2015

@author: corinna
"""
moduleDir = '/windows/D/goettingen/python/swimmertracking/'
wDir= '/media/corinna/Corinna2/SymposionGoe'
    
import sys
if moduleDir not in sys.path:
    sys.path.append(moduleDir)
import readtraces as rt
import os
os.chdir(wDir)
mov=rt.movie('7p5wtpcTTAB_40degC_4_short.mov')
mov.CoordtoTraj('temp', maxdist=200, lenlim=150,lossmargin=5)

