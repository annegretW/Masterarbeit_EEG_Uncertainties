#!/usr/bin/python3
import sys
import numpy as np
import utility_functions
import eeg_model
import json
import time
import math

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp


if __name__ == "__main__":
    # Get path to config file
    parameters_path = sys.argv[1]
    
    # Read config file
    file = open(parameters_path)
    config = json.load(file)
    file.close()

    # Set model (L - Use leadfield, T - Use transfer matrix)
    model = config["Setup"]["Matrix"]

    # Set noise
    relative_noise = config["ModelConfig"]["RelativeNoise"]

    # Set parameters
    center = (config["Geometry"]["Center"]["x"], config["Geometry"]["Center"]["y"])
    radii = config["Geometry"]["Conductivities"]
    conductivities = config["Geometry"]["Conductivities"]

    # Set type
    dipole_type = config["ModelConfig"]["Dipole"]["Type"]

    # Set dipole
    position = (config["ModelConfig"]["Dipole"]["Position"]["x"],config["ModelConfig"]["Dipole"]["Position"]["y"])

    if dipole_type == 'Radial':
        s_ref = utility_functions.get_radial_dipole(position,center)
    else:
        rho = config["ModelConfig"]["Dipole"]["Orientation"]["rho"]
        s_ref = utility_functions.get_dipole(position,center,rho)
    
    # Set paths
    path_electrodes = {}
    
    mesh_types = {}
    path_meshs = {}
    path_matrices = {}
    var_factor = {}
    b_ref = {} 
    sigma = {}

    general_level_config = config["GeneralLevelConfig"]

    levels = config["Sampling"]["Levels"]
    for level in levels:
        level_config = config[level]

        path_electrodes[level] = level_config["Electrodes"] if "Electrodes" in level_config else general_level_config["Electrodes"]     
        path_meshs[level] = level_config["Mesh"] if "Mesh" in level_config else general_level_config["Mesh"]   
        var_factor[level] = level_config["VarFactor"] if "VarFactor" in level_config else general_level_config["VarFactor"]   
        mesh_types[level] = level_config["MeshType"] if "MeshType" in level_config else general_level_config["MeshType"]   

        if model=='T':
            path_matrices[level] = level_config["TransferMatrix"] if "TransferMatrix" in level_config else general_level_config["TransferMatrix"]   
        elif model=='L':
            path_matrices[level] = level_config["LeadfieldMatrix"] if "LeadfieldMatrix" in level_config else general_level_config["LeadfieldMatrix"]   

        m = len(np.load(path_electrodes[level])["arr_0"])
        b_ref[level] = np.zeros(m)

        config_source = config[config[level]["SourceModel"]] if "SourceModel" in config[level] else config[config["GeneralLevelConfig"]["SourceModel"]]   
        if relative_noise==0:
            b_ref[level] = utility_functions.calc_sensor_values(s_ref, path_electrodes[level], mesh_types[level], path_meshs[level], conductivities, config_source)
            sigma_0 = 0.001*np.amax(np.absolute(b_ref[level]))
        else:
            b_ref[level], sigma_0 = utility_functions.calc_disturbed_sensor_values(s_ref, path_electrodes[level], mesh_types[level], path_meshs[level], conductivities, config_source, relative_noise)

        sigma[level] = var_factor[level]*sigma_0

    # Create EEG modell
    if model=='T':
        testmodel = eeg_model.EEGModelTransfer(config, levels, b_ref, sigma, path_matrices, mesh_types, path_meshs, conductivities, center, dipole_type)
    elif model == 'L':
        testmodel = eeg_model.EEGModelLeadfield(levels, b_ref, sigma, path_matrices, path_meshs)

    for level in levels:
        startTime = time.time()

        for i in range(100000):    
            theta = np.array([np.random.uniform(low=0, high=255),np.random.uniform(low=0, high=255),np.random.uniform(low=0.0, high=2*math.pi)])
            testmodel.posterior(theta, level)

        executionTime = (time.time() - startTime)
        print('Execution time on ' + level + ' in seconds: ' + str(executionTime))