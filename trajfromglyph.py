# -*- coding: utf-8 -*-
import fnmatch
import os
import readtraces as rt

matches = []
for root, dirnames, filenames in os.walk('<path>'):# hier pfad ändern!
  for filename in fnmatch.filter(filenames, '*.csv'):
    matches.append(os.path.join(root, filename))
    
for m in matches:
    mov=rt.nomovie(m[:-4]+'-data')
    mov.getGlyphTraj()