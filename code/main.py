# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from utils import *
from ICP import *
import argparse

import sys


if __name__ == '__main__':
   

    # Test ICP and visualize
    # **********************
    #

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset", type=str,default='buddha', choices=['home_at', 'buddha','dragon'])
    parser.add_argument("--n_iter",type=int, default=30)
    parser.add_argument("--rms_threshold",type=float, default=1e-4)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction ,default=False,\
                        help="to enable showing ICP" )


    parser.add_argument("--s_d", action=argparse.BooleanOptionalAction ,default=False,\
                        help="to enable dimension based selection" )
    parser.add_argument("--s_e", type=float ,default=None,\
                        help="to enable and fix entropy based selection threshold" )
    
    parser.add_argument("--w_e",action=argparse.BooleanOptionalAction ,default=False,\
                        help="to enable euclidean weighting" )
    parser.add_argument("--w_n",action=argparse.BooleanOptionalAction ,default=False,\
                        help="to enable normals alignement weighting ")
    parser.add_argument("--w_o",action=argparse.BooleanOptionalAction ,default=False,\
                        help="to enable omnivariance weighting ")
    
    parser.add_argument("--r_e",type=float ,default=None,\
                        help="to enable and fix euclidean rejection filter" )
    parser.add_argument("--r_o",type=float ,default=None,\
                        help="to enable and fix omnivariance rejection filter")
    
    parser.add_argument("--m_o",type=int ,default=None,\
                        help="to enable omnivariance matching (over the k closest neighbors in ref)")
    
    
    args = parser.parse_args()

    print('\n############################################################################\n')

    #  -------- Cloud paths  -------- 
    ref_path,data_path =  get_paths(args)


    
    #  -------- Load clouds and descriptors  -------- 
    ref_ply = read_ply(ref_path)
    data_ply = read_ply(data_path)

    points_ref = np.vstack((ref_ply['x'], ref_ply['y'],ref_ply['z']))
    normals_ref=np.vstack((ref_ply['nx'], ref_ply['ny'],ref_ply['nz']))
    l_ref,p_ref,s_ref=ref_ply['l'],ref_ply['p'],ref_ply['s']
    d_ref,E_ref,V_ref= ref_ply['d'],ref_ply['E'],ref_ply['V']
    ref={'points':points_ref,'n':normals_ref, 'd':d_ref, 'E':E_ref, 'V':V_ref }

    points_data = np.vstack((data_ply['x'], data_ply['y'],data_ply['z']))        
    normals_data=np.vstack((data_ply['nx'], data_ply['ny'],data_ply['nz']))
    l_data,p_data,s_data=data_ply['l'],data_ply['p'],data_ply['s']
    d_data,E_data,V_data= data_ply['d'],data_ply['E'],data_ply['V']
    data={'points':points_data,'n':normals_data, 'd':d_data, 'E':E_data, 'V':V_data }

    # -------- Apply ICP --------

    # a) points selection
    sel_dim, sel_entropy=args.s_d, args.s_e
    idx_data, idx_ref = np.arange(points_data.shape[-1]), np.arange(points_ref.shape[-1])
    if sel_entropy is not None:
        idx_data= selection_entropy(E_data, threshold=sel_entropy)
        # idx_ref =selection_entropy(E_ref, threshold=sel_entropy)
        print(f'Entropy select° : data aligned: {points_data.shape[1]} ---> {len(idx_data)}, rate: {len(idx_data)/points_data.shape[1]}')
        print(f'\t \t ref: {points_ref.shape[1]} ---> {len(idx_ref)}, rate: {len(idx_ref)/points_ref.shape[1]}') 
    else:
        if sel_dim:
            idx_data = selection_dimension(d_data)
            # idx_ref = selection_dimension(d_ref)
            print(f'd=2 select° : data aligned: {points_data.shape[1]} ---> {len(idx_data)}, rate: {len(idx_data)/points_data.shape[1]}')
            print(f'\t\t ref: {points_ref.shape[1]} ---> {len(idx_ref)}, rate: {len(idx_ref)/points_ref.shape[1]}') 
    
    retained_data={key: value[..., idx_data] for key, value in data.items()}
    retained_ref={key: value[..., idx_ref] for key, value in ref.items()}

    # b) KD tree
    tree=KDTree(retained_ref['points'].T)

    # c) ICP
    w_omn, w_normals, w_eucl,m_omn = args.w_o, args.w_n, args.w_e,args.m_o
    if m_omn: print(f'Omnivariance matching, k={m_omn}')
    else: print('Default point to point matching')
    if w_omn:print('Omnivariance weighting')
    else:
        if w_normals: print('Normals alignement weighting')
        else:
            if w_eucl: print('Euclidean weighting')
            else: print('Constant weighting (w_i=1)')
    rej_omn,rej_eucl=args.r_o, args.r_e 
    if rej_omn is not None:
        print(f'Omnivariance reject°, reject° rate: {rej_omn}')
    elif rej_eucl is not None:
        print(f'Euclidean reject°, reject° rate: {rej_eucl}')
    t_list,R_list,T_list,neighbors_list,RMS_list = ICP(retained_data, retained_ref,tree, args.n_iter, args.rms_threshold,\
             rej_omn=rej_omn, rej_eucl=rej_eucl,w_omn=w_omn, w_normals=w_normals, w_eucl=w_eucl,m_omn=args.m_o)
    


    #  -------- Show ICP  --------
    if args.show:
        show_ICP(points_data, points_ref, R_list, T_list, neighbors_list)
    

    #  -------- Plot RMS  -------- 
    plt.plot(t_list,RMS_list)
    plt.show()

    # print('\n t : ',t_list)
    # print('RMSE : ',RMS_list)
    print(f'final t : {t_list[-1]}, final RMSE : {RMS_list[-1]}, best RMSE : {min(RMS_list)}')
    print('\n############################################################################\n')

   

