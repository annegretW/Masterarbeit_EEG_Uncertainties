#!/usr/bin/python3
import sys
import numpy as np
from scipy.io import loadmat
import json

def structured_mesh(N, path, x_min, x_max, y_min, y_max, blurr):
    width = x_max-x_min
    coarseness = int(256/N)
    cell_size = width/N

    nr_nodes = N + 1
    lower_left = np.array([x_min, y_min])
    upper_right = np.array([x_max, y_max])

    # create nodes array for bounding box
    nodes = np.meshgrid(*[np.linspace(ll,ur,num=N+1) for ll,ur in zip(lower_left, upper_right)], indexing='ij')
    nodes_array = np.reshape(np.transpose(np.array(nodes, order='F')),[nr_nodes**2, 2], order = 'C')
        
    # map elements to nodes in bounding box
    elements = np.empty(shape=(N**2,4))
    idx_nodes = np.arange(nodes_array.shape[0])
    idx_top =  idx_nodes[idx_nodes%nr_nodes**2  >= (nr_nodes)*(nr_nodes-1) ]
    idx_right = idx_nodes[idx_nodes%nr_nodes==nr_nodes-1]

    elements[:,0] = np.delete(idx_nodes, np.append(idx_right, idx_top))
    elements[:,1] = elements[:,0] + 1
    elements[:,[2,3]] = elements[:,[0,1]] + nr_nodes
        
    # create tissue labels for elements in bounding box
    tissue_prob_map = loadmat('/home/anne/Masterarbeit/masterarbeit/2d/data/TPMSliceAnne.mat')

    white_prob = np.zeros((N, N),float)
    gray_prob = np.zeros((N, N),float)
    csf_prob = np.zeros((N, N),float)
    skull_prob = np.zeros((N, N),float)
    scalp_prob = np.zeros((N, N),float)

    gray_probabilities = np.zeros((N,N), dtype=float)

    for i in range(N):
        for j in range(N):
            for k in range(coarseness):
                for l in range(coarseness):
                    white_prob[i,j] += tissue_prob_map['tpm']['white'][0][0][coarseness*i+k,coarseness*j+l]
                    gray_prob[i,j] += tissue_prob_map['tpm']['gray'][0][0][coarseness*i+k,coarseness*j+l]
                    csf_prob[i,j] += tissue_prob_map['tpm']['csf'][0][0][coarseness*i+k,coarseness*j+l]
                    skull_prob[i,j] += tissue_prob_map['tpm']['skull'][0][0][coarseness*i+k,coarseness*j+l]
                    scalp_prob[i,j] += tissue_prob_map['tpm']['scalp'][0][0][coarseness*i+k,coarseness*j+l]

                    if(tissue_prob_map['tpm']['gray'][0][0][coarseness*i+k,coarseness*j+l]>gray_probabilities[i,j]):
                        gray_probabilities[i,j] = tissue_prob_map['tpm']['gray'][0][0][coarseness*i+k,coarseness*j+l]
            white_prob[i,j] = white_prob[i,j]/(coarseness**2)
            gray_prob[i,j] = gray_prob[i,j]/(coarseness**2)
            csf_prob[i,j] = csf_prob[i,j]/(coarseness**2)
            skull_prob[i,j] = skull_prob[i,j]/(coarseness**2)
            scalp_prob[i,j] = scalp_prob[i,j]/(coarseness**2)

    gray_probs = np.zeros((N,N), dtype=float)
    labels = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            gray_probs[j,i] = min(1,max(0,np.random.normal(gray_probabilities[i,j], blurr)))
            if white_prob[i,j]>0.5: 
                labels[j,i]=1
            elif gray_prob[i,j]>0.5: 
                labels[j,i]=2
            elif csf_prob[i,j]>0.5: 
                labels[j,i]=3
            elif skull_prob[i,j]>0.5: 
                labels[j,i]=4
            elif scalp_prob[i,j]>0.5: 
                labels[j,i]=5
            else:
                labels[j,i]=0

    nodes = nodes_array.astype(int)
    elements = elements.astype(int)
    gray_probs = np.ravel(gray_probs, order='F')
    labels = np.ravel(labels, order='F').astype(int)

    centers = np.empty((N**2,2))
    for i in range(N**2):
        centers[i] = (nodes[elements[i,0]]+nodes[elements[i,1]]+nodes[elements[i,2]]+nodes[elements[i,3]])/4

    np.savez_compressed(path, cells_per_dim=N, cell_size=cell_size, elements=elements, centers=centers, nodes=nodes, labels=labels, gray_probs=gray_probs)

    print("\nCreated new mesh with \n " + str(len(nodes)) + " nodes \n " + str(len(elements)) + " elements\n")


if __name__ == "__main__":
    N = int(sys.argv[1])
    path = sys.argv[2]

    x_min = 0
    x_max = 256
    y_min = 0
    y_max = 256

    blurr = float(sys.argv[3])

    structured_mesh(N, path, x_min, x_max, y_min, y_max, blurr)