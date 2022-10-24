import colorsys
from re import I
import numpy as np
import utility_functions
import meshio
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import math

def get_midpoints(mesh):
        nodes = mesh.points
        cells = mesh.cells_dict['triangle']
        centers = np.zeros((len(cells),2))
        for i in range(len(cells)):
                cell = cells[i]
                a = nodes[cell[0]]
                b = nodes[cell[1]]
                c = nodes[cell[2]]
                x = (a[0]+b[0]+c[0])/3
                y = (a[1]+b[1]+c[1])/3
                centers[i] = [x,y]
        return centers

def binning(centers,samples):
        bins = np.zeros(len(centers))
        samples = np.transpose(samples)
        for s in samples:
                bins[utility_functions.find_next_node(centers,s)] += 1
        return bins/len(samples)

def draw_model(mesh_path, conductivities_path, point, center, ax):
        mesh = meshio.read(mesh_path)
        f = open(conductivities_path,'r')
        conductivities = []

        with open(conductivities_path) as f:
                for line in f:
                        conductivities.append(float(line.strip()))
        f.close()

        tissues = mesh.cell_data['gmsh:physical'][0]
        values = [conductivities[t] for t in tissues]
        
        ax.tripcolor(mesh.points[:,0], mesh.points[:,1], triangles=mesh.cells_dict['triangle'], facecolors=values)
        
        orientation = utility_functions.get_dipole_orientation(point,center)
        ax.add_artist(
                pt.Arrow(point[0], point[1], 20*orientation[0], 20*orientation[1], width=5, facecolor="red"))


def draw_cells(mesh_path, samples, ax, vmin=None, vmax=None):
        mesh = meshio.read(mesh_path)
        centers = get_midpoints(mesh)
        colors = binning(centers,samples)

        if vmin==None:
                vmin = np.amin(colors)
        if vmax==None:
                vmax = np.amax(colors)
        
        im = ax.tripcolor(mesh.points[:,0], mesh.points[:,1], triangles=mesh.cells_dict['triangle'], facecolors=colors, vmin=vmin, vmax=vmax)
        return im, vmin, vmax

def draw_densities(samples, axis, vmin=None, vmax=None):
        contourf = {}
        if vmin!=None:
                contourf['vmin'] = vmin
        if vmax!=None:
                contourf['vmax'] = vmax

        im = az.plot_dist(np.concatenate([samples[0,:],np.array([35.,219.])]),
            np.concatenate([samples[1,:],np.array([35.,219.])]),
            textsize=18, ax=axis, contourf_kwargs={'vmin': vmin, 'vmax': vmax})
        return im

def set_ax(axis, color='None'):
        for r in [78,86,92]:
                axis.add_artist(
                        pt.Circle((127,127),r,edgecolor="white",facecolor=color))
        axis.set_xlabel('Samples X')
        axis.set_ylabel('Samples Y')
        axis.set_xlim(35, 219)
        axis.set_ylim(35,219)
        axis.set_aspect(1)

def draw_point(axis, point, alpha=1):
        axis.add_artist(
                pt.Circle(point,3,facecolor="red",alpha=alpha))

def diagnostics(samples):
        n = len(samples)
        az.plot_trace(samples)
        plt.xlabel('steps')
        az.plot_autocorr(samples, max_lag=100)
        plt.xlabel('lag')
        print("Effective sample size: " + str(az.ess(samples)/n))
        print("Autocorrelation time: " + str(n/az.ess(samples)))
        print("MCSE: " + str(az.mcse(samples)))