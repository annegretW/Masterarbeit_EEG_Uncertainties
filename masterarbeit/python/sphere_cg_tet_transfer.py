# Compute EEG leadfield using the standard (CG-) FEM approach,
# in a 4-layer spherical tetrahedral sphere model
# with different source models (Partial integration, St. Venant, Whitney and Subtraction)
# using the transfer matrix apporach

# I. Import libraries
import numpy as np
import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt
duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp
import os
import os

# II. Define input files
folder_input = '../data/'
folder_output = '../results/'
filename_grid = os.path.join(folder_input, 'tet_mesh.msh')
filename_tensors = os.path.join(folder_input, 'conductivities.txt')
#filename_dipoles = os.path.join(folder_input, 'sphere_dipoles.txt')
#filename_analytical = os.path.join(folder_input, 'sphere_eeg_analytical.txt')
filename_electrodes = os.path.join(folder_input, 'electrodes_1005.txt')

# III. Create MEEG driver
# We create the driver object which will read the mesh along with the conductivity tensors from the provided files
config = {
    'type' : 'fitted',
    'solver_type' : 'cg',
    'element_type' : 'tetrahedron',
    'volume_conductor' : {
        'grid.filename' : filename_grid,
        'tensors.filename' : filename_tensors
    }
}
driver = dp.MEEGDriver3d(config)

# IV. Read and set electrode positions
# When projecting the electrodes, we choose the closest nodes
electrodes = np.genfromtxt(filename_electrodes,delimiter=None) 
electrodes = [dp.FieldVector3D(t) for t in electrodes.tolist()]
electrode_config = {
    'type' : 'closest_subentity_center',
    'codims' : [3]
}
driver.setElectrodes(electrodes, electrode_config)

# V. Compute EEG leadfield
# Compute the transfer matrix
configTM = {
    'solver' : {
        'reduction' : 1e-10
    }
}
tm = driver.computeEEGTransferMatrix(configTM)
tm_eeg = np.array(tm[0])

# Create source model configurations (Partial integration St. Venant, Subtraction, Whitney)
source_model_configs = {
    'Partial integration' : {
        'type' : 'partial_integration'
    },
    'Venant' : {
        'type' : 'venant',
        'numberOfMoments' : 3,
        'referenceLength' : 20,
        'weightingExponent' : 1,
        'relaxationFactor' : 1e-6,
        'mixedMoments' : True,
        'restrict' : True,
        'initialization' : 'closest_vertex'
    },
    'Whitney' : {
        'type' : 'whitney',
        'referenceLength' : 20,
        'restricted' : True,
        'faceSources' : 'all',
        'edgeSources'  : 'all',
        'interpolation' : 'PBO'
    },
    'Subtraction' : {
        'type' : 'subtraction',
        'intorderadd' : 2,
        'intorderadd_lb' : 2
    }
}

# Read dipoles
dipoles = [dp.Dipole3d([140,140,140], [0.57735, 0.57735, 0.57735])]

# Apply the transfer matrix
solutions = dict()
for sm in source_model_configs:
    lf = driver.applyEEGTransfer(tm_eeg, dipoles, {
                    'source_model' : source_model_configs[sm],
                    'post_process' : True,
                    'subtract_mean' : True
                })
    solutions[sm] = np.array(lf[0])

print(solutions)
# VI. Comparison with (quasi-)analytical solution & visualization
# Load the analytical solution
#analytical = np.transpose(np.genfromtxt(filename_analytical,delimiter= ''))

# compute errors between numerical and analytical solution:
# RDM(u,v) = \left\|\frac{u}{\|u\|}-\frac{v}{\|v\|}\right\|
# MAG(u,v) = \frac{\|u\|}{\|v\|}
# lnMAG(u,v)= \ln(MAG(u,v))
#norm = lambda x : np.linalg.norm(x, axis=1)
#nnum = {sm : norm(solutions[sm]) for sm in solutions}
#nana = norm(analytical)
#lnmag = {sm: np.log(nnum[sm]/nana)  for sm in nnum}
#rdm = {sm: norm(solutions[sm]/nnum[sm][:, None]-analytical/nana[:, None]) for sm in solutions}

# Plot errors
'''
eccentricities = [np.transpose(np.round(np.linalg.norm(np.array(dip.position())-(127,127,127))/78,3)) for dip in dipoles]
df = pd.concat([pd.DataFrame({
        'eccentricity' : eccentricities,
        'RDM' : rdm[sm],
        'lnMAG' : lnmag[sm],
        'Source model' : sm,

    }) for sm in rdm])


plt.figure(figsize=(10,8))
sb.boxplot(y='RDM', x='eccentricity', 
                 data=df, 
                 palette="colorblind",
                 hue='Source model')
plt.savefig(os.path.join(folder_output, 'sphere_cg_tet_transfer_rdm.png'))


plt.figure(figsize=(10,8))
sb.boxplot(y='lnMAG', x='eccentricity', 
                 data=df, 
                 palette="colorblind",
                 hue='Source model')
plt.savefig(os.path.join(folder_output, 'sphere_cg_tet_transfer_mag.png'))


# Visualization of output: mesh, first dipole, resulting potential of this dipole at the electrodes
driver.write({
    'format' : 'vtk',
    'filename' : os.path.join(folder_output, 'sphere_cg_tet_transfer_headmodel')
})

pvtk = dp.PointVTKWriter3d(dipoles[0])
pvtk.write(os.path.join(folder_output, 'sphere_cg_tet_transfer_testdipole'))

pvtk = dp.PointVTKWriter3d(electrodes, True)
pvtk.addScalarData('potential', solutions['Venant'][0]) 
pvtk.write(os.path.join(folder_output, 'sphere_cg_tet_transfer_solution_venant'))

# Print a list of relevant publications
driver.print_citations()
'''