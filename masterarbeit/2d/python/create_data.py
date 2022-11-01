from os.path import exists
import sys
import json
from utility_functions import get_electrodes, save_leadfield_matrix, save_transfer_matrix

if __name__ == "__main__":
    # Get path to config file
    parameters_path = sys.argv[1]
    
    # Read config file
    file = open(parameters_path)
    config = json.load(file)
    file.close()

    model = config["Setup"]["Matrix"]

    conductivities_path = config["Geometry"]["Conductivities"]

    general_level_config = config["GeneralLevelConfig"]
    levels = config["Sampling"]["Levels"]
    for level in levels:
        level_config = config[level]
        electrodes_path = level_config["Electrodes"] if "Electrodes" in level_config else general_level_config["Electrodes"]     
        mesh_path = level_config["Mesh"] if "Mesh" in level_config else general_level_config["Mesh"]   
        mesh_type = level_config["MeshType"] if "MeshType" in level_config else general_level_config["MeshType"]   


        # Generate electrode positions if not already existing  
        if not exists(electrodes_path):
            get_electrodes(electrodes_path,mesh_path)

        # Create leadfield matrices if not already existing
        if model=='L':
            matrix_path = level_config["LeadfieldMatrix"]
            if not exists(matrix_path):
                save_leadfield_matrix(
                    electrodes_path, 
                    conductivities_path, 
                    mesh_path, 
                    matrix_path)

        # Create transfer matrices if not already existing
        elif model == 'T':
            matrix_path = level_config["TransferMatrix"]
            if not exists(matrix_path):
                save_transfer_matrix(
                    electrodes_path, 
                    conductivities_path, 
                    mesh_type,
                    mesh_path, 
                    matrix_path)