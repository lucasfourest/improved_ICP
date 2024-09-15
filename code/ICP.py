#
#
#      0===================================0
#      |      Iterative Closest Point      |
#      0===================================0




import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from ply import write_ply, read_ply
from utils import *
import time
import sys




#  -------- Selection step  --------

def selection_entropy(E, threshold=0.5):
    valid_idx=np.where(E < threshold)[0]
    return valid_idx

def selection_dimension(d):
    valid_idx=np.where(d==2)[0]
    return valid_idx




#  -------- Matching step  --------

def match_omnivariance(data_aligned,tree,V_aligned, V_ref,k=5):

    k_closest_idx_in_ref=np.array(tree.query(data_aligned.T,k=k, return_distance=False))
    distV_neighbors= [[abs(V_aligned[i] - V_ref[idx_neighbor]) for idx_neighbor in k_closest_idx_in_ref[i]] for \
                  i in range(data_aligned.shape[-1])]
    distV_neighbors=np.array(distV_neighbors) 
    matched_idx= np.argmin(distV_neighbors, axis=-1)
    matched_neighbors = np.array([k_closest_idx_in_ref[i,matched_idx[i]] for i in range(data_aligned.shape[-1])])
    return matched_neighbors




#  -------- Weighting step  --------

def get_weights_euclidean(data_aligned, matched_ref):
    distances=np.sqrt(np.sum((data_aligned - matched_ref)**2, axis=0))
    weights=1- distances/np.max(distances)
    return weights

def get_weights_omnivariance(V_aligned, V_matched_ref):
    distances=np.abs(V_aligned - V_matched_ref)
    weights=1- distances/np.max(distances)
    return weights

def get_weights_normals(normals_aligned, normals_matched_ref):
    weights = np.abs(np.sum(normals_aligned * normals_matched_ref, axis=0))
    return weights


#  -------- Rejection step  --------

def rejection_euclidean(data_aligned, matched_ref, filter=0.5):
   
    keep_rate =1-filter
    distances=np.sum((data_aligned - matched_ref)**2, axis=0)
    ranked_ids = np.argsort(distances)
    to_keep=int(ranked_ids.shape[0]*keep_rate)
    return ranked_ids[:to_keep]


def rejection_omnivariance(V_aligned, V_matched_ref, filter=0.5):
    keep_rate =1-filter
    distances=np.abs(V_aligned - V_matched_ref)
    ranked_ids = np.argsort(distances)# in increasing order
    to_keep=int(ranked_ids.shape[0]*keep_rate)
    return ranked_ids[:to_keep]


def rejection_euclidean_std(data_aligned, matched_ref):
    distances=np.sqrt(np.sum((data_aligned - matched_ref)**2, axis=0))
    m, std= np.mean(distances), np.std(distances)
    ids_to_keep=np.where(distances-m< 2.5*std)[0]
    return ids_to_keep


def rejection_omnivariance_std(V_aligned, V_matched_ref):
    distances=np.abs(V_aligned - V_matched_ref)
    m, std= np.mean(distances), np.std(distances)
    ids_to_keep=np.where(distances-m < 2.5*std)[0]
    return ids_to_keep


#  -------- Fitting step  --------

def best_rigid_transform(data, ref, weights=None):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    if weights is None:
        p_m, p_m_p=np.mean(ref,axis=1), np.mean(data,axis=1)
        Q, Q_p = ref - p_m[:, None] , data - p_m_p[:, None]
        C=Q_p @ Q.T
    else:
        p_m, p_m_p=np.sum(weights[None,:]*ref, axis=-1)/np.sum(weights) ,\
                    np.sum(weights[None,:]*data, axis=-1)/np.sum(weights) 
        C=((weights[None,:]*data) @ ref.T)/np.sum(weights)  -  p_m_p[:,None] @ p_m[None,:]

    U, S, V_transp = np.linalg.svd(C)
    V=V_transp.T
    R = V @ U.T
    if np.linalg.det(R)<0:
        U[:,-1]=-U[:,-1]
        R=V@U.T
    T = p_m - R@p_m_p
    

    return R, T[:,None]

#  -------- Global ICP algo  --------

def ICP(data_dic, ref_dic,tree, max_iter, RMS_threshold,rej_omn=None, rej_eucl=None\
                       , w_omn=False, w_normals=False, w_eucl=False, m_omn=None):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        t_list = list of timestamps of successive steps (starting from 0)
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, math each point of data with a point in ref this obtain 
                a (1 x N_data) array of indices. This is the list of those arrays at each iteration
           
    '''

    # load dictionnaries of data 
    data,  normals_data, d_data, E_data, V_data = data_dic['points'], data_dic['n'], data_dic['d'], \
                            data_dic['E'], data_dic['V']
    ref,  normals_ref, d_ref, E_ref, V_ref = ref_dic['points'], ref_dic['n'], ref_dic['d'], \
                             ref_dic['E'], ref_dic['V']
    

    # Variable for aligned data
    data_aligned = np.copy(data)
    if w_normals:
        normals_data_aligned=np.copy(normals_data)

    

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []

    t0=time.time()
    idx0=np.array(tree.query(data_aligned.T,k=1, return_distance=False)).T
    matched_ref=ref[:,idx0.squeeze(0)]
    rms0=rmse(data_aligned, matched_ref)
    t1=time.time()

    RMS_list = [rms0]
    t_list = [t1]
    Rn_list=[]
    rms=np.inf
    iter=0

    while iter < max_iter and rms>RMS_threshold:

        # Matching
        if m_omn is not None:idx=match_omnivariance(data_aligned, tree, V_data, V_ref, k=m_omn)
        else:idx=np.array(tree.query(data_aligned.T,k=1, return_distance=False)).T.squeeze(0)
        neighbors_list.append(idx)
        matched_ref=ref[:,neighbors_list[-1]]


        # Weighting
        weights=None
        if w_omn:
            V_aligned, V_matched_ref = V_data, V_ref[neighbors_list[-1]]
            weights=get_weights_omnivariance(V_aligned, V_matched_ref)    
        else:
            if w_normals:
                normals_matched_ref= normals_ref[:,neighbors_list[-1]]
                weights=get_weights_normals(normals_data_aligned, normals_matched_ref)
            elif w_eucl:
                weights=get_weights_euclidean(data_aligned, matched_ref)  


        # Rejecting
        retain_ids=np.array([i for i in range(data_aligned.shape[-1])])
        if rej_omn is not None:
            V_aligned, V_matched_ref = V_data, V_ref[neighbors_list[-1]]
            retain_ids=rejection_omnivariance(V_aligned, V_matched_ref,filter=rej_omn)

        elif rej_eucl is not None:
            retain_ids=rejection_euclidean(data_aligned, matched_ref,filter=rej_eucl) 
     
        retained_data_aligned, retained_matched_ref = data_aligned[:,retain_ids],matched_ref[:,retain_ids]
        if weights is not None:
                weights=weights[retain_ids]


        # Fitting       
        R,T=best_rigid_transform(data=retained_data_aligned, ref=retained_matched_ref, weights=weights)
        if iter==0: R_aux, T_aux= R, T
        else:
            R_aux, T_aux= R@ R_aux, R@T_aux+T
        R_list.append(R_aux), T_list.append(T_aux)
        data_aligned=R@data_aligned + T
        if w_normals:
            normals_data_aligned=R@normals_data_aligned + T


        # Evaluating
        rms=rmse(data_aligned, matched_ref)
        t=time.time()
        t_list.append(t)

        RMS_list.append(rms)
        iter+=1
        
        
        

    t_list=[t_list[i]-t0 for i in range(len(t_list))]
    

    return t_list, R_list, T_list, neighbors_list, RMS_list





