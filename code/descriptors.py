#------------------------------------------------------------------------------------------
#
#      Descriptors
#
#------------------------------------------------------------------------------------------
    

import os
import numpy as np
from sklearn.neighbors import KDTree
import argparse
from ply import read_ply, write_ply
from utils import get_paths



def PCA(points):
    N,dim=points.shape
    mean=np.mean(points, axis=0)
    m=points-mean[None,:]
    cov=m.T @ m/N
    val,vec=np.linalg.eigh(cov)


    eigenvalues = val
    eigenvectors = vec
    return eigenvalues, eigenvectors


def compute_local_PCA(query_points, cloud_points,tree, radius,k=None):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    
    all_eigenvalues, all_eigenvectors, neigh_len=[],[], []
    if k is None:
        neighbor_ids=tree.query_radius(query_points, r=radius)
    else:
        _,neighbor_ids =  tree.query(query_points, k=k)

    for neighbor_list in neighbor_ids:
        neigh_len.append(len(neighbor_list))
        neighborhood=cloud_points[neighbor_list]
        eigenvalues, eigenvectors=PCA(neighborhood)
        all_eigenvalues.append(eigenvalues), all_eigenvectors.append(eigenvectors)

    all_eigenvalues, all_eigenvectors = np.array(all_eigenvalues), np.array(all_eigenvectors)
    neigh_len=np.array(neigh_len)


    return all_eigenvalues, all_eigenvectors, neigh_len

# "static" neighborhood features: same neighborhood parameters for everyone
def compute_features_static(query_points, cloud_points,tree, radius,k=None):

    all_eigenvalues, all_eigenvectors, n_len=compute_local_PCA(query_points, cloud_points, tree,radius,k=k)
    normals = all_eigenvectors[:, :, 0]
    normals=normals/np.linalg.norm(normals, axis=-1)[:,None]

    eps=1e-8
    lambd3, lambd2, lambd1=all_eigenvalues[:,0], all_eigenvalues[:,1], all_eigenvalues[:,2]
    theta1,theta2,theta3=lambd1**0.5, lambd2**0.5, lambd3**0.5

    a1d = (theta1-theta2)/(theta1+eps)
    a2d = (theta2-theta3)/(theta1+eps)
    a3d = theta3/(theta1+eps)
    s=a1d+a2d+a3d
    a1d,a2d,a3d=a1d/s,a2d/s,a3d/s
    V=theta1*theta2*theta3

    return normals,a1d, a2d, a3d, V



# find optimal neighborhoods for each point, in terms of nb of neighbors (kNN)
def opt_neighborhoods_k(query_points, cloud_points, tree,k_vec):
    best_k=np.zeros(query_points.shape[0])
    best_normals=np.zeros((query_points.shape[0],3))
    best_a=np.zeros((query_points.shape[0],3))
    best_E=np.inf*np.ones(query_points.shape[0])
    best_V=np.zeros(query_points.shape[0])

    for k in k_vec:
        normals,a1d, a2d, a3d,V=compute_features_static(query_points, cloud_points, tree,radius=None,k=k)
        entropy= - a1d*np.log(a1d)- a2d*np.log(a2d)  - a3d*np.log(a3d)
        to_change=(entropy<best_E)
        best_k[to_change]=k
        best_normals[to_change]=normals[to_change]
        
        best_a[to_change]=np.array([a1d[to_change],a2d[to_change],a3d[to_change]]).T

        best_E[to_change]=entropy[to_change]
        best_V[to_change]=V[to_change]

    
    d=np.argmax(best_a,axis=-1).astype(float)+1.0


    # char vector: [r,n, a1D, a2D, a3Dd, Ef, V] for each point
    opt_char={idx:{'k':best_k[idx],'n':best_normals[idx],'a':best_a[idx],'E':best_E[idx],\
                   'V':best_V[idx], 'd': d[idx]} for idx in range(query_points.shape[0])}
        
    return opt_char



if __name__ == '__main__':

    # Compute optimal neighborhoods and geometrical descriptors 
    # **********************
    #

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset", type=str,default='buddha', choices=['home_at', 'buddha','dragon'])
    parser.add_argument("--static_r", type=float, default=None)
    parser.add_argument("--static_k", type=int, default=None)
    parser.add_argument("--find_optimal_k",action=argparse.BooleanOptionalAction ,default=False,\
                        help="to compute and save optimal neighborhoods (kNN) parameters, should be done once" )
    
    args = parser.parse_args()

    # Cloud paths
    ref_path,data_path =  get_paths(args)
    
    # Load clouds
    ref_ply = read_ply(ref_path)
    data_ply = read_ply(data_path)
    ref = np.vstack((ref_ply['x'], ref_ply['y'],ref_ply['z'])).T
    data = np.vstack((data_ply['x'], data_ply['y'], data_ply['z'])).T

    paths=[ref_path, data_path]
    clouds=[ref, data]

    for cloud,path in zip(clouds, paths):
        tree=KDTree(cloud, metric='euclidean')

        if (args.static_k is not None) or (args.static_r is not None):
            r,k=args.static_r, args.static_k
            

            # compute and save features
            normals,linearity, planarity, sphericity,n_len=compute_features_static(cloud, cloud,tree, r,k=k)
            write_ply(path, (cloud, normals, linearity, planarity, sphericity),['x', 'y', 'z', 'nx', 'ny', 'nz','l', 'p', 's' ])


        if args.find_optimal_k:
            N=cloud.shape[0]
            k_vec=np.linspace((1/2000)*N, (1/300)*N,num=5).astype(int)
            opt_char_dic=opt_neighborhoods_k(cloud, cloud, tree, k_vec)

            normals=np.array([list(opt_char_dic[idx]['n']) for idx in range(len(opt_char_dic))])
            linearity=np.array([opt_char_dic[idx]['a'][0] for idx in range(len(opt_char_dic))])
            planarity=np.array([opt_char_dic[idx]['a'][1] for idx in range(len(opt_char_dic))])
            sphericity=np.array([opt_char_dic[idx]['a'][2] for idx in range(len(opt_char_dic))])
            dimension=np.array([opt_char_dic[idx]['d'] for idx in range(len(opt_char_dic))])
            entropy=np.array([opt_char_dic[idx]['E'] for idx in range(len(opt_char_dic))])
            volume=np.array([opt_char_dic[idx]['V'] for idx in range(len(opt_char_dic))])
            write_ply(path, (cloud, normals, linearity, planarity, sphericity,dimension, volume \
                             ,entropy),['x', 'y', 'z', 'nx', 'ny', 'nz','l', 'p', 's','d','V', 'E' ])
            