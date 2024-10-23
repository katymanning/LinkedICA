#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
import DOF_eigenspectrum as DOF
import flica_various as alb_various
from flica_various import flica_parseoptions
import numpy as np
import numpy.matlib as npm
import copy


def flica_init_params(Y,opts):
    
    K = len(Y)   #num kinds of data 
    L = np.int(opts['num_components'])    #num_components;
    R = Y[0].shape[1]    #num of subjects 
    default_list_of_arrays=[np.array(a).astype('float64') for a in range (0,K)] #list to save variables
    
    #set default options if not provided    
    opts=flica_parseoptions(R, opts) 
    
    #Compute degrees of freedom per voxel, if not provided
    if opts['dof_per_voxel']=='auto_eigenspectrum':
        opts['dof_per_voxel'] = np.ones(K);
        for k in range (0,K): 
            opts['dof_per_voxel'][k] = DOF.est_DOF_eigenspectrum(Y[k]) / (Y[k].shape[0])  # check alb.est_DOF_eigenspectrum ??             
    DD = opts['dof_per_voxel'] 


    # Multiply data by Virtual Decimation factor (often sqrt'd!) and Initialize <X> and <H> using PCA:
    N=np.zeros(K).astype('float64') #num of voxels per data type
    
    if opts['initH']=='PCA': 
        
        tmpY = copy.deepcopy(Y)       
        for k in range (0,K):     
            Y[k]=np.ascontiguousarray(Y[k],dtype='float64')
            N[k] = Y[k].shape[0]
            tmpY[k] = Y[k] * np.sqrt(DD[k]) # the     
        tmpYcat =np.vstack(tmpY) 
                
        
        [tmpU,tmpS,tmpV]=np.linalg.svd(tmpYcat,full_matrices=False);
        tmpV=np.transpose(tmpV)
        tmpS=np.diag(tmpS)
        tmpU = np.dot(tmpU[:,0:L],tmpS[0:L, 0:L])
        tmpV = np.divide(tmpV[:,0:L], 1./alb_various.rms(tmpU,0,[]));
        tmpU = np.divide(tmpU, alb_various.rms(tmpU,0,[]));
        
        del tmpYcat   
            
    elif opts['initH']=='PCAnew': #gong                
        tmpU=[np.array(a).astype('float64') for a in range (0,K)] #list to save variables    
        tmpV=np.zeros((R,L))
        for k in range(0,K):
            Y[k]=np.ascontiguousarray(Y[k],dtype='float64')
            N[k] = Y[k].shape[0]            
            [tmpU1,tmpS1,tmpV1]=np.linalg.svd(Y[k] * np.sqrt(DD[k]),full_matrices=False);
            tmpV1=np.transpose(tmpV1)
            tmpS1=np.diag(tmpS1)
            tmpU1 = np.dot(tmpU1[:,0:L],tmpS1[0:L, 0:L])
            tmpV1 = np.divide(tmpV1[:,0:L], 1./alb_various.rms(tmpU1,0,[]));
            tmpU1 = np.divide(tmpU1, alb_various.rms(tmpU1,0,[]));
            
            tmpU[k]=tmpU1
            tmpV=tmpV+tmpV1
            
        tmpU=np.vstack(tmpU)
        #may be we can use glm to get tmpV too!!
        tmpV=tmpV/K
        
    elif opts['initH']=='fixed': # you MUST provide a valid initialization mixing matrix as opts['initH']
        tmpV = opts['initH'].T
        tmpU = np.squeeze(np.linalg.lstsq(tmpV,tmpYcat.T)[0]).T 
 
        
    tmpV=tmpV.astype('float64')
    tmpU=tmpU.astype('float64') 
    #init subject courses matrix H
    H = np.divide(tmpV.T , np.sqrt(np.mean(DD)))
    
    #I FIX G TO BE ONE, Rgroups>1 not implemented yet
    G=np.ones(1,dtype='int') 
    Gmat =np.ones(R).astype('float64')
    H2Gmat = np.dot( np.square(H) , Gmat.T) # [LxG]
    H_colcov =np.dot(np.matlib.eye(L),pow(10,-12)).astype('float64')
    
    #define variables    
    X=copy.deepcopy(default_list_of_arrays)
    W=copy.deepcopy(default_list_of_arrays)
    W_rowcov=copy.deepcopy(default_list_of_arrays)
    WtW=copy.deepcopy(default_list_of_arrays)
    XtDX=copy.deepcopy(default_list_of_arrays)
    Y2D_sumN=copy.deepcopy(default_list_of_arrays)
    X2=copy.deepcopy(default_list_of_arrays)
    for  k in range (0,K): # De-concatenate to get X[k] estimates:
        if k==0:
            X[k] = tmpU[0:N.astype(int)[0],0:L]; # / sqrt(DD(k));
        else:
            X[k] = tmpU[ np.sum(N.astype(int)[0:k])  : np.sum(N.astype(int)[0:k+1])  , 0:L]         
        W[k] = np.multiply(np.ones(L).astype('float64') , np.sqrt(np.divide(np.mean(DD) ,DD[k])))  # so Y = X*diag(W)*H + noise;
        W_rowcov[k] = np.multiply(np.matlib.eye(L),pow(10,-12)).astype('float64') 
        prior_W_var = np.divide(np.ones(1).astype('float64'),DD[k])  
        WtW[k] = np.multiply(W[k][np.newaxis, :].T , W[k]) + W_rowcov[k];
        XtDX[k] = np.dot (np.dot(X[k].T , X[k]), DD[k]) # double prec.?
        Y2D_sumN[k] = np.multiply(DD[k] , np.sum(np.square(Y[k]),0) )    # double prec.?
        X2[k] = np.square(X[k])
    
    tmpU = None
    tmpV = None
    tmpS = None
    #Set up the models for P(X|params) and P(lambda):
    # define variables
    prior_pi_weights=copy.deepcopy(default_list_of_arrays)
    pi_weights=copy.deepcopy(default_list_of_arrays)
    pi_mean=copy.deepcopy(default_list_of_arrays)
    pi_log=copy.deepcopy(default_list_of_arrays)
    prior_beta_b=copy.deepcopy(default_list_of_arrays)
    prior_beta_c=copy.deepcopy(default_list_of_arrays)
    beta=copy.deepcopy(default_list_of_arrays)
    beta_log=copy.deepcopy(default_list_of_arrays)
    beta_c=copy.deepcopy(default_list_of_arrays)
    beta_binv=copy.deepcopy(default_list_of_arrays)
    mu = copy.deepcopy(default_list_of_arrays)
    mu2 =copy.deepcopy(default_list_of_arrays)
    mu_var =copy.deepcopy(default_list_of_arrays)    
    prior_lambda_b=copy.deepcopy(default_list_of_arrays)
    prior_lambda_c=copy.deepcopy(default_list_of_arrays)
    Lambda=copy.deepcopy(default_list_of_arrays) # I use capital in lambda from .m!!
    lambda_log=copy.deepcopy(default_list_of_arrays)
    lambda_c=copy.deepcopy(default_list_of_arrays)
    lambda_binv=copy.deepcopy(default_list_of_arrays)
    lambda_R=copy.deepcopy(default_list_of_arrays)
    lambda_log_R=copy.deepcopy(default_list_of_arrays)
    HlambdaHt=copy.deepcopy(default_list_of_arrays)
    sumN_Dq=np.ndarray([K,3,L]).astype('float64')
    sumN_DqXq=np.ndarray([K,3,L]).astype('float64')
    sumN_DqXq2=np.ndarray([K,3,L]).astype('float64')
    sumN_Dqlogq=np.ndarray([K,3,L]).astype('float64')
    qq=[np.ndarray(a).astype('float64') for a in range (0,K)]
    Xq_var=[np.ndarray(a).astype('float64') for a in range (0,K)]
    for  k in range (0,K):
        #Initialize pi_mean{k} [3xL]
        prior_pi_weights[k] = (N[k]*0.1 * np.ones([3, L])).astype('float64'); 
        pi_weights[k] = copy.deepcopy(prior_pi_weights[k]);
        dumm=np.divide( pi_weights[k], np.tile(np.sum(pi_weights[k],0),(pi_weights[k].shape[0], 1)) )
        pi_mean[k] = copy.deepcopy(dumm)
        pi_log[k] = np.log(pi_mean[k])
        # Initialize beta{k} [3xL]
        prior_beta_b[k] = np.tile([pow(10,3), 1, pow(10,3)],(L,1)).T.astype('float64')
        prior_beta_c[k] = np.tile([pow(10,-6), pow(10,6), pow(10,-6)],(L,1)).T.astype('float64') 
        beta[k] = np.tile(np.power([.1, 1000., 1.],-2), (L,1)).T.astype('float64')  #TEST??
        beta_log[k] = np.log(beta[k])        
        beta_c[k] = np.multiply(np.power(10,6),np.ones(beta[k].shape)).astype('float64')        
        beta_binv[k] = np.divide(np.power(10,6),beta[k]).astype('float64');
        #Initialize mu{k} [3xL]:
        prior_mu_mean = np.zeros(1).astype('float64') 
        prior_mu_var = pow(10,4)*np.ones(1).astype('float64') 
        mu[k] = prior_mu_mean + np.zeros([3,L]).astype('float64')
        mu2[k] = np.square(mu[k]) 
        mu_var[k] = (np.multiply(mu[k], 0)+pow(10,-12)).astype('float64') 
        #Initialize q{k} [NxLx3]:
        qq[k] = np.tile(pi_mean[k].T, (N[k].astype('int'), 1, 1))
        #Initialize X_q [NxLx3]:
        Xq_var[k] = np.multiply(pow(10,-12), np.ones([L,3])).astype('float64') #######################################################
        #Set up the model for lambda: 
        if opts['lambda_dims'] == 'R':
            prior_lambda_b[k] = np.multiply( pow(10,12), np.ones([R,1])).astype('float64') 
            prior_lambda_c[k] = np.multiply( pow(10,-12), np.ones([R,1])).astype('float64') 
            Lambda[k] = np.transpose(np.matrix(np.power(alb_various.rms(Y[k],0,[]),-2))).astype('float64')
            #% Note that any "missing data" scans should use Ga(b=1e-18, c=1e12)
            lambda_log[k] = np.log(Lambda[k]);
            lambda_c[k] = pow(10,12)*np.ones(1).astype('float64')
            lambda_binv[k] = np.divide(lambda_c[k],Lambda[k])
            lambda_R[k] = copy.deepcopy(Lambda[k])            
        elif opts['lambda_dims'] == 'G':
            print('not default, need to add?')            
        elif opts['lambda_dims'] == 'o': #the '' case in matlab
            prior_lambda_b[k] =  pow(10,12)*np.ones(1).astype('float64')
            prior_lambda_c[k] = pow(10,-12)*np.ones(1).astype('float64')
            Lambda[k] = np.transpose(np.matrix(np.power(alb_various.rms(Y[k],[],[]),-2))).astype('float64')
            lambda_log[k] = np.log(Lambda[k]);
            lambda_c[k] = pow(10,12)*np.ones(1).astype('float64')
            lambda_binv[k] = np.divide(lambda_c[k],Lambda[k]) 
            lambda_R[k] = np.tile(Lambda[k],(R,1))
    
    # Initialize eta [LxG]:Initial updates: eta H {lambda X|q,q,X}*2
    prior_eta_b = pow(10,6)*np.ones(1).astype('float64') #1e3 * 1000;
    prior_eta_c = pow(10,-3)*np.ones(1).astype('float64')#1e-3;
    eta = np.multiply(np.multiply(prior_eta_b,prior_eta_c), np.ones([L,np.int(1)]))  
    eta_log = np.log(eta)
    eta_c = copy.deepcopy(prior_eta_c)
    eta_binv = np.divide(np.ones(1).astype('float64'),prior_eta_b) 
    
    if opts['lambda_dims'] == 'R': 
        NH = R;         
    else: 
        NH = np.int(1) #G.astype('int'); # Which is 1...need to remove
        
    
    #gather for output as dictionary
    Posteriors={"X":X,"X2":X2,"XtDX": XtDX,"Xq_var":Xq_var,                
                "W":W,"W_rowcov":W_rowcov,"WtW":WtW,
                "H":H,"H2Gmat":H2Gmat,"H_colcov": H_colcov,"HlambdaHt":HlambdaHt,
                "mu": mu,"mu2":mu2,"mu_var":mu_var,
                "beta":beta,"beta_log":beta_log,"beta_c":beta_c,"beta_binv":beta_binv,
                "pi_weights":pi_weights,"pi_mean":pi_mean,"pi_log":pi_log,
                "Lambda":Lambda,"lambda_log":lambda_log,"lambda_c":lambda_c,"lambda_binv":lambda_binv,"lambda_R":lambda_R,"lambda_log_R":lambda_log_R,
                "eta":eta,"eta_log":eta_log,"eta_c":eta_c,"eta_binv":eta_binv,
                "Gmat":Gmat,"Y2D_sumN":Y2D_sumN,"sumN_Dq":sumN_Dq,"sumN_DqXq":sumN_DqXq,"sumN_DqXq2":sumN_DqXq2,"sumN_Dqlogq":sumN_Dqlogq,
                "qq":qq}
            
    Priors={"prior_pi_weights":prior_pi_weights,"prior_beta_b":prior_beta_b,"prior_beta_c":prior_beta_c,
            "prior_mu_mean":prior_mu_mean,"prior_mu_var":prior_mu_var,
            "prior_lambda_b":prior_lambda_b,"prior_lambda_c":prior_lambda_c,
            "prior_eta_b":prior_eta_b,"prior_eta_c":prior_eta_c, "prior_W_var":prior_W_var}
    
    
    Constants={"K":K,"L":L,"R":R ,"DD":DD,"N":N,"G":G,"NH":NH}
    
    return Priors, Posteriors, Constants 