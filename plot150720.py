# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:26:10 2015

@author: cmaass
"""

import numpy as np
from matplotlib import pyplot as plt
theta=np.linspace(0,np.pi,100)
fig=plt.figure(figsize=[5,8])
beta=-.2
k=.03
plt.subplot(211)
plt.title(r"$\beta=-.2$, $v(\theta)=\sin(\theta)-\beta\sin(2\theta)$")
plt.plot(theta,k*theta, "--", label=r"$F_h\propto\theta$")
plt.plot(theta[:-1],np.diff(np.sin(theta)+beta*(np.sin(2*theta))), label=r"$F_s\propto\partial_\theta v (\theta)$")
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$F$')
x_tick = [0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
plt.gca().set_xticks(x_tick)
plt.gca().set_xticklabels(x_label)

k=.4
plt.subplot(212)
plt.plot(theta,k*theta, "--", label=r"$F_h\propto\theta$")
plt.plot(theta,np.sin(theta)+beta*(np.sin(2*theta)), label=r"$F_s\propto v(\theta)$")
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$F$')
x_tick = [0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
plt.gca().set_xticks(x_tick)
plt.gca().set_xticklabels(x_label)
plt.savefig('/home/cmaass/test.pdf')