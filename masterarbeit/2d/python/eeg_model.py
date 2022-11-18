#!/usr/bin/python3
import sys
import umbridge
import numpy as np
import utility_functions
import meshio
import math
import json
import structured_mesh as msh

from scipy.io import loadmat

duneuropy_path='/home/anne/Masterarbeit/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

#################################################################################################
# This is an EEG model, containing lists of meshes and the according leadfield                  #
# matrices - one per level                                                                      #   
#                                                                                               #
# Attributes:                                                                                   #
#   mesh                - list of meshs of different coarseness (one per level)                 #
#   leadfield_matrix    - list of ladfield matrices according to the meshs (one per level)      #   
#   b_ref               - reference values of sources                                           #
#   sigma               - posterior variance                                                    #
##################################################################################################
class EEGModelLeadfield(umbridge.Model):
    def __init__(self, levels, b_ref, sigma, leadfield_path_list, mesh_path_list):
        super(EEGModelLeadfield, self).__init__()

        # initialize lists
        self.mesh = {}
        self.leadfield_matrix = {}
        self.m = {}

        for l in levels:
            # read mesh
            mesh = meshio.read(mesh_path_list[l])
            self.mesh[l] = mesh
            #self.mesh.append(msh.StructuredMesh(center, radii, cells_per_dim[i]))
            self.leadfield_matrix[l] = np.load(leadfield_path_list[l])['arr_0']
            self.m = len(b_ref[l])

        # reference values of measurement values
        self.b_ref = b_ref

        # posterior variance
        self.sigma = sigma


    def get_input_sizes(self):
        return [2]

    def get_output_sizes(self):
        return [1]

    # Calculates the posterior probability of the source theta on a given level
    def posterior(self, theta, level):
        #if (math.sqrt((theta[0]-127)**2+(theta[1]-127)**2)>78):
        #    return -1e20

        # find next node to theta on the mesh and select the according leadfield
        points = np.array(self.mesh[level].points[:,0:2])
        index = utility_functions.find_next_node(points,theta[0:2])
        b = np.array(self.leadfield_matrix[i][index])

        posterior = ((1/(2*self.sigma[level]**2))**(self.m[level]/2))*np.exp(-(1/(2*self.sigma[level]**2))*(np.linalg.norm(np.array(self.b_ref[level])-np.array(b), 2)/np.linalg.norm(np.array(self.b_ref[level]), 2))**2)
        if posterior==0:
           return -1e20
        
        return np.log(posterior)
        
    def __call__(self, parameters, config={'level': 1}):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True


#################################################################################################
# This is an EEG model, containing lists of meshes and the according transfer                   #
# matrices - one per level                                                                      #   
#                                                                                               #
# Attributes:                                                                                   #
#   mesh                - list of meshs of different coarseness (one per level)                 #
#   transfer_matrix     - list of transfer matrices according to the meshs (one per level)      #   
#   b_ref               - reference values of sources                                           #
#   sigma               - posterior variance                                                    #
##################################################################################################
class EEGModelTransfer(umbridge.Model):
    def __init__(self, config, levels, b_ref, sigma, transfer_path_list, mesh_types, mesh_list, conductivities, center, mode):
        super(EEGModelTransfer, self).__init__()

        assert(mode in ['Radial','Arbitrary'])
        self.mode = mode

        # initialize lists
        self.mesh = {}
        self.mesh_type = {}
        self.transfer_matrix = {}
        self.meg_drivers = {}
        self.m = {}
        self.tissue_probs = {}
        self.config = {}

        # iterate through all levels
        for l in levels:
            self.transfer_matrix[l] = np.load(transfer_path_list[l])['arr_0']
            
            if mesh_list[l].split(".")[-1] == "msh":
                mesh = meshio.read(mesh_list[l])
                self.mesh[l] = mesh
                driver_config = {
                    'type' : 'fitted',
                    'solver_type' : 'cg',
                    'element_type' : 'tetrahedron',
                    'volume_conductor': {
                        'grid.filename' : mesh_list[l], 
                        'tensors.filename' : conductivities
                        },
                    'post_process' : 'true', 
                    'subtract_mean' : 'true'
                }

                #TODO: Compute tissue_probs for .msh file
                self.tissue_probs[l] = np.ones(len(mesh.cells))
            
            else:
                mesh = np.load(mesh_list[l])
                self.mesh[l] = mesh
                driver_config = {
                    'type' : 'fitted',
                    'solver_type' : 'cg',
                    'element_type' : 'hexahedron',
                    'volume_conductor' : {
                        'grid' : {
                            'elements' : mesh['elements'].tolist(),
                            'nodes' : mesh['nodes'].tolist()
                        },
                        'tensors' : {
                            'labels' : mesh['labels'].tolist(),
                            'conductivities' : conductivities
                        },
                    },
                    'post_process' : 'true', 
                    'subtract_mean' : 'true'
                }
                self.tissue_probs[l] = mesh['gray_probs']

            self.mesh_type[l] =  config[l]["MeshType"] if "MeshType" in config[l] else config["GeneralLevelConfig"]["MeshType"]   
            config_source = config[config[l]["SourceModel"]] if "SourceModel" in config[l] else config[config["GeneralLevelConfig"]["SourceModel"]]   

            if config_source["type"] =="venant":
                source_model_config = {
                    'type' : config_source["type"],
                    'numberOfMoments' : config_source["numberOfMoments"],
                    'referenceLength' : config_source["referenceLength"],
                    'weightingExponent' : config_source["weightingExponent"],
                    'relaxationFactor' : config_source["relaxationFactor"],
                    'mixedMoments' : bool(config_source["mixedMoments"]),
                    'restrict' : bool(config_source["restrict"]),
                    'initialization' : config_source["initialization"],
                }
            elif config_source["type"] =="subtraction":
                source_model_config = {
                    "type": "subtraction",
                    "intorderadd" : 2,
                    "intorderadd_lb" : 2
                }
            else:
                source_model_config = {
                    "type": "partial_integration"
                }

            self.config[l] = {
                'solver.reduction' : 1e-10,
                'source_model' : source_model_config,
                'post_process' : True,
                'subtract_mean' : True
            }

            meg_driver = dp.MEEGDriver2d(driver_config)
            self.meg_drivers[l] = meg_driver

            self.m[l] = len(b_ref[l])

        # reference values of measurement values
        self.b_ref = b_ref
        
        # posterior variance
        self.sigma = sigma

        self.center = center
        self.dim = config["Geometry"]["Dim"]

        self.domain_x_min = config["Geometry"]["Domain_x_Min"]
        self.domain_x_max = config["Geometry"]["Domain_x_Max"]
        self.domain_y_min = config["Geometry"]["Domain_y_Min"]
        self.domain_y_max = config["Geometry"]["Domain_y_Max"]

        #tissue_prob_map = loadmat('/home/anne/Masterarbeit/masterarbeit/2d//data/T1SliceAnne.mat')
        #self.gray_prob = tissue_prob_map['T1Slice']['gray'][0][0]

    def get_input_sizes(self):
        return [self.dim]

    def get_output_sizes(self):
        return [1]

    # Calculates the posterior probability of the source theta on a given level
    def posterior(self, theta, chain, level):
        #if (math.sqrt((theta[0]-127)**2+(theta[1]-127)**2)>78):
        #    return -1e20
        if(theta[0]<self.domain_x_min or theta[0]>self.domain_x_max or theta[1]<self.domain_y_min  or theta[1]>self.domain_y_max):
            return -1e20

        if self.mode=='Radial':
            next_dipole = utility_functions.get_radial_dipole(theta[0:2], self.center)

        else:
            next_dipole = utility_functions.get_dipole(theta[0:2], self.center, theta[2])

        b = self.meg_drivers[level].applyEEGTransfer(self.transfer_matrix[level],[next_dipole],self.config[level])[0]

        # calculate the posterior as a normal distribution
        c = utility_functions.find_next_center(self.mesh[level],self.mesh_type[level],theta)
        tissue_prob = self.tissue_probs[level][int(c[0]+self.mesh[level]['cells_per_dim']*c[1])]
        #tissue_prob = self.mesh[level]['gray_probs'][utility_functions.find_next_node(self.mesh[level]['centers'],theta[0:2])]
        w = 1e-3 # level dependent
        posterior = ((1-w)*tissue_prob+w)*((1/(2*self.sigma[level][chain]**2))**(self.m[level]/2))*np.exp(-(1/(2*self.sigma[level][chain]**2))*(np.linalg.norm(np.array(self.b_ref[level][chain])-np.array(b), 2)/np.linalg.norm(np.array(self.b_ref[level][chain]), 2))**2)

        if posterior==0:
           return -1e20

        return np.log(posterior)
        
    def __call__(self, parameters, config):
        return [[self.posterior(parameters[0],config["chain"],config["level"])]]

    def supports_evaluate(self):
        return True


if __name__ == "__main__":
    ##### READ CONFIG #####
    # Get path to config file
    parameters_path = sys.argv[1]
    
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
    dipole_type = config["ModelConfig"]["Dipole"]["Type"]

    ##### SET DIPOLE #####
    # Dipole position is either read from the config or generated randomly
    s_ref = {}
    for c in range(chains):
        if config["Setup"]["Dipole"] == "Random":
            mesh_ref = np.load(config["GeneralLevelConfig"]["Reference"]["Mesh"] if "Reference" in config["GeneralLevelConfig"] else config[config["Sampling"]["Levels"][-1]]["Reference"]["Mesh"])
            while(True):
                position = (utility_functions.get_random(config["Geometry"]["Domain_x_Min"],config["Geometry"]["Domain_x_Max"]),
                        utility_functions.get_random(config["Geometry"]["Domain_y_Min"],config["Geometry"]["Domain_y_Max"]))
                center_ref = utility_functions.find_next_center(mesh_ref,'hex',position)
                if(mesh_ref['gray_probs'][int(center_ref[0]+mesh_ref['cells_per_dim']*center_ref[1])]>0.5):
                    break
        else:
            position = (config["ModelConfig"]["Dipole"]["Position"]["x"],config["ModelConfig"]["Dipole"]["Position"]["y"])

        # Dipole orientation is either radial or given by the config or generated randomly
        if dipole_type == 'Radial':
            s_ref[c] = utility_functions.get_radial_dipole(position,center)
        else:
            if config["Setup"]["Dipole"] == "Random":
                rho = utility_functions.get_random(0,2*math.pi)
            else:
                rho = config["ModelConfig"]["Dipole"]["Orientation"]["rho"]

            print("Dipole:")
            print(utility_functions.get_dipole(position,center,rho))
            s_ref[c] = utility_functions.get_dipole(position,center,rho)

    ##### COMPUTE REFERENCE SENSOR VALUES #####
    # Read configs yielding for all levels
    general_level_config = config["GeneralLevelConfig"]

    if "Reference" in general_level_config:
        b_ref_general = {}
        sigma_0_general = {}
        for c in range(chains):
            transfer_matrix = general_level_config["Reference"]["TransferMatrix"]
            mesh = general_level_config["Reference"]["Mesh"]
            source_model = general_level_config["Reference"]["SourceModel"]
            config_source = config[source_model]

            b_ref_general[c], sigma_0_general[c] = utility_functions.calc_disturbed_sensor_values(s_ref[c], transfer_matrix, mesh, conductivities, config_source, relative_noise)

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
        if "Reference" in level_config:
            for c in range(chains):
                config_source = config[config[level]["SourceModel"]] if "SourceModel" in config[level] else config[config["GeneralLevelConfig"]["SourceModel"]]   
                b_ref[level][c], sigma_0 = utility_functions.calc_disturbed_sensor_values(s_ref[c], path_matrices[level], mesh_types[level], path_meshs[level], conductivities, config_source, relative_noise)
                sigma[level][c] = var_factor[level]*sigma_0
        else:
            b_ref[level] = b_ref_general
            sigma[level] = sigma_0_general

    ##### CREATE EEG MODEL #####
    # Either transfer matrix or leadfield matrix is used in the model
    if model=='T':
        testmodel = EEGModelTransfer(config, levels, b_ref, sigma, path_matrices, mesh_types, path_meshs, conductivities, center, dipole_type)
    elif model == 'L':
        testmodel = EEGModelLeadfield(levels, b_ref, sigma, path_matrices, path_meshs)

    # Send model via localhost
    umbridge.serve_model(testmodel, 4243)