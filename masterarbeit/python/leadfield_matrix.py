#!/usr/bin/python3

###############################################################################
# Create and save the leadfield matrix                                        #
###############################################################################

import numpy as np
import time
import leadfield

duneuropy_path='/home/anne/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

mesh_path='../data/tet_mesh.msh'
tensors_path='../data/conductivities.txt'
electrodes_path='../data/electrodes.txt'
dipoles_path='../data/dipoles_ecc_0.99_radial.txt'

if __name__=="__main__":
    print("Create leadfield matrix")
    n = 100

    m = 20
    L = leadfield.create_leadfield_matrix(mesh_path,tensors_path,electrodes_path,dipoles_path,n,m)
    np.savez_compressed('../data/leadfield_matrix_100_L1.npz', L)

    m = 40
    L = leadfield.create_leadfield_matrix(mesh_path,tensors_path,electrodes_path,dipoles_path,n,m)
    np.savez_compressed('../data/leadfield_matrix_100_L2.npz', L)

    m = 70
    L = leadfield.create_leadfield_matrix(mesh_path,tensors_path,electrodes_path,dipoles_path,n,m)
    np.savez_compressed('../data/leadfield_matrix_100_L3.npz', L)