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

def compute_costs(parameters_path):    
    ##### READ CONFIG #####
    # Get path to config file    
    # Read config file
    file = open(parameters_path)
    config = json.load(file)
    file.close()

    ##### SET GENERAL CONFIGS #####
    model = config["Setup"]["Matrix"]
    relative_noise = config["ModelConfig"]["RelativeNoise"]
    center = (config["Geometry"]["Center"]["x"], config["Geometry"]["Center"]["y"])
    conductivities = config["Geometry"]["Conductivities"]
    chains = config["Setup"]["Chains"]

    # Set type
    dipole_type = config["ModelConfig"]["DipoleType"]

    ##### SET DIPOLE #####
    # Dipole position is either read from the config or generated randomly
    s_ref = {}
    if config["Setup"]["Dipole"] == "Random":
        mesh_ref = np.load(config["GeneralLevelConfig"]["Reference"]["Mesh"] if "Reference" in config["GeneralLevelConfig"] else config[config["Sampling"]["Levels"][-1]]["Mesh"])
        while(True):
            position = (utility_functions.get_random(config["Geometry"]["Domain_x_Min"],config["Geometry"]["Domain_x_Max"]),
                    utility_functions.get_random(config["Geometry"]["Domain_y_Min"],config["Geometry"]["Domain_y_Max"]))
            center_ref = utility_functions.find_next_center(mesh_ref,'hex',position)
            if(mesh_ref['gray_probs'][int(center_ref[0]+mesh_ref['cells_per_dim']*center_ref[1])]>0.5):
                break
    else:
        position = (config["ModelConfig"]["Dipoles"][0][0],config["ModelConfig"]["Dipoles"][0][1])

    # Dipole orientation is either radial or given by the config or generated randomly
    if dipole_type == 'Radial':
        s_ref[0] = utility_functions.get_radial_dipole(position,center)
    else:
        if config["Setup"]["Dipole"] == "Random":
            rho = utility_functions.get_random(0,2*math.pi)
        else:
            rho = config["ModelConfig"]["Dipoles"][0][2]

        print("Dipole:")
        print(utility_functions.get_dipole(position,center,rho))
        s_ref[0] = utility_functions.get_dipole(position,center,rho)

    ##### COMPUTE REFERENCE SENSOR VALUES #####
    # Read configs yielding for all levels
    general_level_config = config["GeneralLevelConfig"]

    if "Reference" in general_level_config:
        b_ref_general = {}
        sigma_0_general = {}
        transfer_matrix = general_level_config["Reference"]["TransferMatrix"]
        mesh = general_level_config["Reference"]["Mesh"]
        source_model = general_level_config["Reference"]["SourceModel"]
        config_source = config[source_model]
        b_ref_general[0], sigma_0_general[0] = utility_functions.calc_disturbed_sensor_values(s_ref[0], transfer_matrix, mesh, conductivities, config_source, relative_noise)

    ##### SET LEVEL DEPENDENT CONFIGS #####
    # Initialize dictionairies
    path_electrodes = {}
    mesh_types = {}
    path_meshs = {}
    path_matrices = {}
    var_factor = {}
    b_ref = {} 
    sigma = {}

    # Iterate through all levels
    levels = config["Sampling"]["Levels"]
    for level in levels:
        # Read config yielding for the current level
        level_config = config[level]

        # Set paths
        path_electrodes[level] = level_config["Electrodes"] if "Electrodes" in level_config else general_level_config["Electrodes"]     
        path_meshs[level] = level_config["Mesh"] if "Mesh" in level_config else general_level_config["Mesh"]   
        var_factor[level] = level_config["VarFactor"] if "VarFactor" in level_config else general_level_config["VarFactor"]   
        mesh_types[level] = level_config["MeshType"] if "MeshType" in level_config else general_level_config["MeshType"]   

        if model=='T':
            path_matrices[level] = level_config["TransferMatrix"] if "TransferMatrix" in level_config else general_level_config["TransferMatrix"]   
        elif model=='L':
            path_matrices[level] = level_config["LeadfieldMatrix"] if "LeadfieldMatrix" in level_config else general_level_config["LeadfieldMatrix"]   

        m = len(np.load(path_electrodes[level])["arr_0"])

        # Compute reference sensor values for the level or use the same for each level
        if "Reference" in general_level_config:
            b_ref[level] = b_ref_general
            sigma[level] = sigma_0_general
        else:
            b_ref_c = {}
            sigma_c = {}
            config_source = config[config[level]["SourceModel"]]   
            b_ref_c[0], sigma_0 = utility_functions.calc_disturbed_sensor_values(s_ref[0], path_matrices[level], path_meshs[level], conductivities, config_source, relative_noise)
            sigma_c[0] = var_factor[level]*sigma_0
            b_ref[level] = b_ref_c
            sigma[level] = sigma_c

    ##### CREATE EEG MODEL #####
    # Either transfer matrix or leadfield matrix is used in the model
    print(b_ref) 

    if model=='T':
        testmodel = eeg_model.EEGModelTransfer(config, levels, b_ref, sigma, path_matrices, mesh_types, path_meshs, conductivities, center, dipole_type)
    elif model == 'L':
        testmodel = eeg_model.EEGModelLeadfield(config, levels, b_ref, sigma, path_matrices, mesh_types, path_meshs, conductivities, center, dipole_type)


    sec_per_sample = {}
    for level in levels:
        startTime = time.time()
        samples = 1000
        for i in range(samples):    
            theta = np.array([np.random.uniform(low=0, high=255),np.random.uniform(low=0, high=255),np.random.uniform(low=0.0, high=2*math.pi)])
            testmodel.posterior(theta, 0, level)

        executionTime = (time.time() - startTime)
        #print('Execution time on ' + level + ' in seconds: ' + str(executionTime))
        sec_per_sample[level] = executionTime/samples

    return sec_per_sample

if __name__ == "__main__":
    # Get path to config file
    parameters_path = sys.argv[1]

    sec_per_sample = compute_costs(parameters_path)
    print(sec_per_sample)

