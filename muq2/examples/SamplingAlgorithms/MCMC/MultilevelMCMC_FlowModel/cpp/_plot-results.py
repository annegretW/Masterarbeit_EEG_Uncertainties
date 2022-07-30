# linear algegbra package for python
import numpy as np

import h5py as h5

# import plotting packages and set default figure options
useserif = True # use a serif font with figures?
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
if useserif:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['text.usetex'] = True
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

filename = 'output.h5'
file = h5.File(filename, 'r')

truex = file['/truth/x'] [()].T [0]
soln = file['/truth/solution'] [()].T [0]
obsLoc = file['/data/locations'] [()].T [0]
data = file['/data/observations'] [()].T [0]

priorx = file['/prior information/x'] [()].T [0]
priorSamples = file['/prior information/samples'] [()]

fig = plt.figure()
ax = plt.gca()
ax.plot(truex, soln)
ax.plot(obsLoc, data, 'o')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel(r'Solution $u(x)$')
ax.set_xlabel(r'Space $x$')
plt.savefig('TrueSolution.png', format='png', bbox_inches='tight')

fig = plt.figure()
ax = plt.gca()
for samp in priorSamples:
    ax.plot(priorx, samp)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel(r'Log conductivity')
ax.set_xlabel(r'Space $x$')
plt.savefig('PriorSamples.png', format='png', bbox_inches='tight')
