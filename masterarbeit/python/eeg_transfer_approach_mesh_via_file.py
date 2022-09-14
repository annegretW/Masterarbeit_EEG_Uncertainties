#!/bin/python3

import numpy as np
import time
import utility_functions

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

#optionally add simbiopy for analytical EEG solutions
simbiopy_path='/home/anne/Masterarbeit/duneuro/simbiosphere/build/src'
sys.path.append(simbiopy_path)
import simbiopy as sp

def analytical_solution(dipole, mesh_path, tensors_path, electrodes_path):
  # create driver
  volume_conductor_cfg = {
    'grid.filename' : mesh_path, 
    'tensors.filename' : tensors_path}
    
  driver_cfg = {
    'type' : 'fitted', 
    'solver_type' : 'cg', 
    'element_type' : 'tetrahedron', 
    'post_process' : 'true', 
    'subtract_mean' : 'true'}

  solver_cfg = {
    'reduction' : '1e-14', 
    'edge_norm_type' : 'houston', 
    'penalty' : '20', 
    'scheme' : 'sipg', 
    'weights' : 'tensorOnly'}
    
  #driver_cfg['solver'] = solver_cfg
  driver_cfg['volume_conductor'] = volume_conductor_cfg

  print('Creating driver')
  meeg_driver = dp.MEEGDriver3d(driver_cfg)
  print('Driver created')

  # set electrodes
  print('Setting electrodes')
  electrode_cfg = {'type' : 'closest_subentity_center', 'codims' : '3'}
  electrodes = dp.read_field_vectors_3d(electrodes_path)
  meeg_driver.setElectrodes(electrodes, electrode_cfg)
  print('Electodes set')

  ##################
  ## Optional : Compute analytical solutions via simbio
  ##################

  # compute EEG forward solutions analytically using simbiopy
  # mm
  print('Computing analytical EEG solutions')
  center = [127, 127, 127]
  # mm
  radii = [92, 86, 80, 78]
  # S/ mm
  conductivities = [0.00043, 0.00001, 0.00179, 0.00033]
  electrodes_simbio = [np.array(electrode).tolist() for electrode in electrodes]
  analytical_solution = sp.analytic_solution(radii, center, conductivities, electrodes_simbio, np.array(dipole.position()).tolist(), np.array(dipole.moment()).tolist())
  mean = sum(analytical_solution) / len(analytical_solution)
  analytical_solution = [x - mean for x in analytical_solution]
  print('Analytical EEG solutions computed')

  return analytical_solution