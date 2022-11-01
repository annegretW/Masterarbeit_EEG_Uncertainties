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
        if (math.sqrt((theta[0]-127)**2+(theta[1]-127)**2)>78):
            return -1e20

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
        self.transfer_matrix = {}
        self.meg_drivers = {}
        self.m = {}

        self.tissue_probs = {}
        self.config = {}

        for l in levels:
            self.transfer_matrix[l] = np.load(transfer_path_list[l])['arr_0']
            
            if mesh_types[l] == "File":
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

            else:
                mesh = msh.StructuredMesh(mesh_list[l])
                self.mesh[l] = mesh
                driver_config = {
                    'type' : 'fitted',
                    'solver_type' : 'cg',
                    'element_type' : 'hexahedron',
                    'volume_conductor' : {
                        'grid' : {
                            'elements' : mesh.elements.tolist(),
                            'nodes' : mesh.nodes.tolist()
                        },
                        'tensors' : {
                            'labels' : mesh.labels.tolist(),
                            'conductivities' : conductivities
                        },
                    },
                    'post_process' : 'true', 
                    'subtract_mean' : 'true'
                }
                self.tissue_probs[l] = mesh.gray_probs

            config_source = config[config[level]["SourceModel"]] if "SourceModel" in config[level] else config[config["GeneralLevelConfig"]["SourceModel"]]   

            if config_source["type"] =="Venant":
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
            elif config_source["type"] =="Subtraction":
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
        self.dim = len(center)

        
        #tissue_prob_map = loadmat('/home/anne/Masterarbeit/masterarbeit/2d//data/T1SliceAnne.mat')
        #self.gray_prob = tissue_prob_map['T1Slice']['gray'][0][0]

    def get_input_sizes(self):
        if self.mode=='Radial':
            return [self.dim]
        else:
            return [self.dim+1]

    def get_output_sizes(self):
        return [1]

    # Calculates the posterior probability of the source theta on a given level
    def posterior(self, theta, level):
        if (math.sqrt((theta[0]-127)**2+(theta[1]-127)**2)>78):
            return -1e20

        if self.mode=='Radial':
            next_dipole = utility_functions.get_radial_dipole(theta[0:self.dim], self.center)

        else:
            next_dipole = utility_functions.get_dipole(theta[0:self.dim], self.center, theta[self.dim])

        #next_dipole = utility_functions.get_radial_dipole(theta, self.center)

        b = self.meg_drivers[level].applyEEGTransfer(self.transfer_matrix[level],[next_dipole],self.config[level])[0]

        # calculate the posterior as a normal distribution
        c = self.mesh[level].find_next_center(theta)
        tissue_prob = self.tissue_probs[level][int(c[0]+self.mesh[level].cells_per_dim*c[1])]
        #tissue_prob = self.mesh[level]['gray_probs'][utility_functions.find_next_node(self.mesh[level]['centers'],theta[0:2])]
        posterior = tissue_prob*((1/(2*self.sigma[level]**2))**(self.m[level]/2))*np.exp(-(1/(2*self.sigma[level]**2))*(np.linalg.norm(np.array(self.b_ref[level])-np.array(b), 2)/np.linalg.norm(np.array(self.b_ref[level]), 2))**2)

        if posterior==0:
           return -1e20

        '''if(level=="Level3"):
            print(theta)
            print(posterior)'''

        return np.log(posterior)
        
    def __call__(self, parameters, config):
        return [[self.posterior(parameters[0],config["level"])]]

    def supports_evaluate(self):
        return True

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

        if relative_noise==0:
            b_ref[level] = utility_functions.calc_sensor_values(s_ref, path_electrodes[level], mesh_types[level], path_meshs[level], conductivities)
            sigma_0 = 0.001*np.amax(np.absolute(b_ref[level]))
        else:
            b_ref[level], sigma_0 = utility_functions.calc_disturbed_sensor_values(s_ref, path_electrodes[level], mesh_types[level], path_meshs[level], conductivities, relative_noise)

        sigma[level] = var_factor[level]*sigma_0

    # Create EEG modell
    if model=='T':
        testmodel = EEGModelTransfer(config, levels, b_ref, sigma, path_matrices, mesh_types, path_meshs, conductivities, center, dipole_type)
    elif model == 'L':
        testmodel = EEGModelLeadfield(levels, b_ref, sigma, path_matrices, path_meshs)

    # send via localhost
    umbridge.serve_model(testmodel, 4243)