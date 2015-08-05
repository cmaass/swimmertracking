# -*- coding: utf-8 -*-
import readtraces as rt
import os
from matplotlib import cm
os.chdir('/media/cmdata/datagoe/')#hier Verzeichnisnamen angeben.
mov=rt.movie('single_droplet_1.avi')#hier Dateiname/relativer Pfad
mov.readParas() #er nimmt an, daß das paras.txt file im 'moviename-data' Verzeichnis ist.
mov.plotMovie(outname=None, decim=3, scale=2, crop=[0,0,0,0], mask='trajectory', frate=10, cmap=cm.jet, bounds=(0,1e8), tr=True, channel=0, lenlim=100)
#wenn outname none, nimmt er default (-traced.avi)
#decim: nur jeder n-te frame kommt in den Film
#scale: Film wird um Faktor n runterskaliert
#crop: Abschnitte links unten rechts oben (unbeschnitten, wenn alles 0)
#mask: sucht nach Trajektorienfiles mit diesem Schlagwort
#frate: Bildrate des Films
#cmap: colour map
#bounds: Framebereich/Filmausschnitt, obere/untere Bildnummer. (0,1e8) heißt, er nimmt alles
#tr: weiß ich auch nicht ganz, was ich damit wollte. Auf True lassen
#channel: Bildkanal
#lenlim: minimale Trajektorienlänge. Sortiert Kleinscheiß aus.p