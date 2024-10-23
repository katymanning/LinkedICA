##!/usr/bin/env python2
# -*- coding: utf-8 -*-
#from flica_small_updates import update_X
from flica_various import apply3_diag
from flica_various import inv_prescale
from flica_various import apply3_inv_prescale
from flica_various import apply3_diag2
from flica_various import sum_dims
from flica_various import logdet 
from flica_various import apply3_logdet  
 
import numpy as np
import scipy as sc
from numpy.lib.stride_tricks import as_strided
from pylab import size #* #for the find command or size
from pylab import identity
import copy
from pylab import trace
import time
#from numba import jit
#from flica_jit import better_dot

#from argparse import Namespace
#import scipy as sc
#from pylab import * 
#import copy
#from numpy.lib.stride_tricks import as_strided
#from scipy.linalg import get_blas_funcs #to accelerate np.dot

#@numba.jit(nopython=True, parallel=True)
#from numba import jit
#from numba import autojit, prange
#@jit(nopython=True)
#def better_dot(A,B):
#    output= np.dot(A,B) #A=A.astype('float64');B=B.astype('float64');
#    return output #np.dot(A,B)

#@jit(nopython=True, parallel=True)     
              

         

              

#@jit(nopython=True,parallel=True)
#@jit(nopython=True, parallel=True) 
#@autojit  
#from numba import autojit, prange
#@autojit

#@jit(nopython=True,parallel=True)
#@jit(nopython=True, parallel=True) 

#@autojit     
def update_X_k(input_dict):
    #if i==0:
    precalc_YlambdaHTW_NxL = np.dot(input_dict['Y_k'],np.multiply( np.dot(np.array(input_dict['lambda_R_k']),input_dict['W_k']),np.transpose(input_dict['H'])))
    
    
    for i in range(input_dict['L']):# (0,input_dict['L']): 
        #tt=time.time()                  
        input_dict['X_k'][:,i]=0  
        input_dict['X2_k'][:,i]=np.nan 
        
        
        #%% Update P'(X_i|q_i)            
        tmpM_N = precalc_YlambdaHTW_NxL[:,i] - np.dot(input_dict['X_k'], np.multiply(input_dict['WtW_k'][:,i], np.matrix(input_dict['HlambdaHt_k'][:,i]).T))
        #tt2=time.time() 
        tmpM_MxN =  np.add(tmpM_N,np.matrix(np.multiply(input_dict['beta_k'][:,i],input_dict['mu_k'][:,i])),order='F') 
        #print 'cost_1_4 = ' , time.time()-tt2        
        tmpL_M = np.multiply(input_dict['WtW_k'][i,i] , input_dict['HlambdaHt_k'][i,i]) + input_dict['beta_k'][:,i] 
        tmpVpost = np.divide(np.float64(1),tmpL_M)
        input_dict['Xq_var_k'][i,:] = copy.deepcopy(tmpVpost)#deep copy? In tmpLogQand xq2ki I use tmpVpost but its originaly Xq_var[k][i,:]
        Xqki = np.divide( tmpM_MxN, np.matrix(tmpL_M),dtype="float64")  # Xqki, Xqki_sq and tmpM_MxN are also large, vovelsx3          
        Xqki_sq=np.square(Xqki,order='F')                
        Xq2ki = np.add(Xqki_sq, tmpVpost, order='F') 
        #%% Update P'(q)  
        tmpLogQ = np.matrix( np.divide(np.add(np.log(tmpVpost,dtype="float64") , np.subtract(input_dict['beta_log_k'][:,i],
                               np.multiply(input_dict['beta_k'][:,i],input_dict['mu2_k'][:,i])),order='F') ,np.float64(2)) + 
                            np.squeeze(input_dict['pi_log_k'][:,i])) 
                                                         
        tmpLogQ = np.add(tmpLogQ,np.divide(Xqki_sq,np.matrix(np.multiply(np.float64(2),tmpVpost,order='F')),dtype='float64',order='F'),dtype='float64',order='F') #Xqki_sq is large  voxelsx3 so tmpLogQ also from here on                                 
 
        tmpLogQ = np.subtract(tmpLogQ,  np.amax(tmpLogQ,1),dtype='float64',order='F')

        qki = np.exp(tmpLogQ,dtype='float64',order='F')         
        qki = np.divide(qki, np.array(np.sum(qki,1,dtype="float64")),dtype='float64',order='F') #better sum float64       
        #print 'cost_2_4 = ' , time.time()-tt2         
        input_dict['sumN_Dq_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum(qki,0,dtype='float64')),dtype='float64')
        input_dict['sumN_DqXq_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum(np.multiply(qki,Xqki),0,dtype='float64')),dtype='float64') 
        input_dict['sumN_DqXq2_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum( np.multiply(qki,  Xq2ki,order='F' ),0,dtype='float64')).astype('float64'))          
        tmp_qlogq = np.multiply(qki,np.log(qki,dtype="float64"),order='F')
        tmp_qlogq[qki==0] = 0;  #% limit as q->0 of q*log(q) is 0.                
        input_dict['sumN_Dqlogq_k'][:,i] = np.multiply(input_dict['DD_k'] , np.matrix(np.sum(tmp_qlogq,0,dtype='float64')),
                                          order='F', dtype='float64')
        input_dict['X_k'][:,i] = np.squeeze(np.sum( np.multiply(Xqki, qki,order='K', dtype='float64'), 1))
        input_dict['X2_k'][:,i] = np.squeeze(np.sum( np.multiply(Xq2ki , qki,order='K', dtype='float64'), 1) )
        
    output_X_k_dict={'X_k':input_dict['X_k'],'X2_k':input_dict['X2_k'],'sumN_Dqlogq_k':input_dict['sumN_Dqlogq_k'],
                    'sumN_DqXq2_k':input_dict['sumN_DqXq2_k'],'sumN_DqXq_k':input_dict['sumN_DqXq_k'], 'sumN_Dq_k':input_dict['sumN_Dq_k'],
                    'Xq_var_k':input_dict['Xq_var_k']}
    return output_X_k_dict 




def update_mixmod(input_dict):

    for k in range(input_dict['K']):            
            input_dict['XtDX'][k] = np.dot(np.dot(np.transpose(input_dict['X'][k]), input_dict['X'][k]) , input_dict['DD'][k]) #% [LxL]
            np.fill_diagonal(input_dict['XtDX'][k],np.dot( np.sum( input_dict['X2'][k],0,dtype="float64") , input_dict['DD'][k]) ) # replace diagonal to include covariance
            #%% Update P'(pi_mean)
            input_dict['pi_weights'][k] = input_dict['prior_pi_weights'][k] + input_dict['sumN_Dq'][k] # [3xL]
            input_dict['pi_mean'][k] = np.divide(input_dict['pi_weights'][k], np.matrix(np.sum(input_dict['pi_weights'][k],0)) )
            input_dict['pi_log'][k] =np.subtract( sc.special.psi(input_dict['pi_weights'][k]) , np.matrix(sc.special.psi(np.sum(input_dict['pi_weights'][k],0))) ); #% [3xL]                
            #%% Update P'(beta)
            input_dict['beta_c'][k] = input_dict['prior_beta_c'][k] + (0.5*input_dict['sumN_Dq'][k]); #% [3xL]
            tmp = np.multiply(input_dict['sumN_Dq'][k] , input_dict['mu2'][k]) + input_dict['sumN_DqXq2'][k] - (2* np.multiply( input_dict['mu'][k] , input_dict['sumN_DqXq'][k])); #% [NxLx3]
            input_dict['beta_binv'][k] = np.float64(1)/input_dict['prior_beta_b'][k] + (tmp/np.float64(2)); #% [3xL]
            input_dict['beta'][k] = np.divide( input_dict['beta_c'][k] , input_dict['beta_binv'][k]); #% [3xL]
            input_dict['beta_log'][k] = sc.special.psi(input_dict['beta_c'][k]) - np.log(input_dict['beta_binv'][k])   #if ~(all(beta{k}(:) > 1e-10)), warning 'X getting awfully large', end %#ok<WNTAG>
            #%% Update P'(mu)
            tmp_L = (1./input_dict['prior_mu_var']) + np.multiply(input_dict['beta'][k] , input_dict['sumN_Dq'][k] );# % [3xL]
            tmp_M = np.divide( input_dict['prior_mu_mean'] , input_dict['prior_mu_var']) + np.multiply( input_dict['beta'][k] , input_dict['sumN_DqXq'][k]); #% [3xL]
            input_dict['mu'][k] = np.divide(tmp_M,tmp_L); #% [3xL]
            input_dict['mu_var'][k] = 1./tmp_L; #% [3xL]
            input_dict['mu2'][k] = np.square(input_dict['mu'][k]) + input_dict['mu_var'][k]
            
    output_mixmod_dict={'XtDX':input_dict['XtDX'],'pi_weights':input_dict['pi_weights'],'pi_log':input_dict['pi_log'],'pi_mean':input_dict['pi_mean'],
                    'beta_c':input_dict['beta_c'],'beta_binv':input_dict['beta_binv'],'beta':input_dict['beta'],'beta_log':input_dict['beta_log'],
                    'mu':input_dict['mu'],'mu_var':input_dict['mu_var'],'mu2':input_dict['mu2']}
                    
    return output_mixmod_dict 


def update_eta(input_dict):
    eta_binv =np.transpose(np.matrix( (np.float64(1)/input_dict['prior_eta_b']) + (input_dict['H2Gmat']/np.float64(2)) ).astype('float64')); 
    eta_c = input_dict['prior_eta_c'] + np.tile( (np.sum(input_dict['Gmat'],0)/2).astype('float64'),(input_dict['L'], 1)) ; 
    eta = np.divide(eta_c , eta_binv) 
    eta_log = sc.special.polygamma(0, eta_c) - np.log(eta_binv,dtype="float64") 
    output_eta_dict={'eta_binv':eta_binv,'eta_c':eta_c, 'eta':eta, 'eta_log':eta_log}
    return output_eta_dict

def update_H(input_dict):        
     if 'R' in input_dict['opts']['lambda_dims']: 
         aaa=np.expand_dims(np.dot(input_dict['eta'],np.matrix(input_dict['Gmat'])),axis=1) #[L=NumIcas 1 R=NumSubs]
         tmpVinv_LxLxNH = apply3_diag(aaa);
         tmp_R_to_NH = range(0,input_dict['R']) #1:R;
     else: 
        aaa=np.transpose( np.expand_dims(input_dict['eta'],axis=2),(0 ,2 , 1))
        tmpVinv_LxLxNH = np.squeeze(apply3_diag(aaa)) 
        tmp_R_to_NH = np.ones(input_dict['R']).astype(int)      
     
     tmpM = np.zeros([input_dict['L'],input_dict['R']]).astype('float64') 
     alb_dum=np.empty([input_dict['K'],input_dict['L']]).astype('float64')
     alb_dum[:]=np.nan;
     input_dict['H_PCs'] = np.vstack([alb_dum, np.transpose(input_dict['eta'])]) 
     for k in range (0,input_dict['K']): 
         input_dict['W'][k]=np.squeeze(np.array(input_dict['W'][k].T)).astype('float64')
         tmp_lambda_NH = input_dict['Lambda'][k] + np.zeros([input_dict['NH'],1],dtype='float64'); 
         aaa=np.array(np.dot( np.multiply(input_dict['WtW'][k].flatten(1), input_dict['XtDX'][k].flatten(1)).T, tmp_lambda_NH.T))#,order="F")
         aaa=as_strided(aaa,shape=(input_dict['L'],input_dict['L'],input_dict['NH']))
         if input_dict['NH'] ==1:
             aaa=aaa[:,:,0] #try to remove this loop 
         tmpVinv_LxLxNH = tmpVinv_LxLxNH + aaa; #del aaa 
         spm=np.dot(np.transpose(input_dict['X'][k]), input_dict['Y'][k])
         tmpM = tmpM + input_dict['DD'][k] * np.dot( np.dot( np.diag(input_dict['W'][k]) , spm) , np.diag(np.array(input_dict['lambda_R'][k].flatten())[0])  )
         input_dict['H_PCs'][k] = np.dot( np.multiply(np.diag(input_dict['WtW'][k]) , np.diag(input_dict['XtDX'][k]) )  , np.mean(input_dict['lambda_R'][k],dtype='float64'))                                  

     # Calculate H, H covariance, <H*Ht> and <H*lambda*Ht>         
     if input_dict['NH']==1:
        input_dict['H_colcov'] = inv_prescale(tmpVinv_LxLxNH)
        input_dict['H'] = np.dot(input_dict['H_colcov'], tmpM);
        alb_dum3=np.diag(input_dict['H_colcov']) 
     else: 
        input_dict['H_colcov'] =  apply3_inv_prescale(tmpVinv_LxLxNH)
        for rr in range (0,input_dict['R']):
            input_dict['H'][:,rr] = np.dot(input_dict['H_colcov'][:,:,tmp_R_to_NH[rr]] , tmpM[:,rr])     
        alb_dum3=apply3_diag2(input_dict['H_colcov']) 
        
     if input_dict['NH']==input_dict['R']:
        input_dict['H2Gmat'] = np.dot( np.square(input_dict['H']) ,input_dict['Gmat']) + np.dot(alb_dum3 ,input_dict['Gmat']) # [LxG]
     else: #not tested
        input_dict['H2Gmat'] = np.dot(np.square(input_dict['H']) , input_dict['Gmat']) + np.dot(alb_dum3 , np.dot( np.transpose(input_dict['Gmat']),input_dict['Gmat'])) # [LxG] 
        
     output_H_dict={'H':input_dict['H'],'H2Gmat':input_dict['H2Gmat'], 'H_colcov':input_dict['H_colcov'], 
                    'H_PCs':input_dict['H_PCs'], 'tmp_R_to_NH':tmp_R_to_NH, 'W':input_dict['W']}
     
     return output_H_dict    
 

def update_HlambdaHt_and_W(input_dict):            
    for k in range (0,input_dict['K']):
            input_dict['HlambdaHt'][k] = np.dot( np.dot(input_dict['H'] , np.diag(np.array(input_dict['lambda_R'][k].flatten())[0])    ) ,np.transpose(input_dict['H']))
            if size(input_dict['H_colcov'].shape)==2:
                ss=np.dot( np.transpose(input_dict['Gmat']) , input_dict['lambda_R'][k])
                input_dict['HlambdaHt'][k] = np.add(input_dict['HlambdaHt'][k] , np.multiply(input_dict['H_colcov'] , ss[0,0]) )  
            else:   
                if input_dict['H_colcov'].shape[2] == input_dict['R']:
                    for rr in range(0,input_dict['R']):
                        input_dict['HlambdaHt'][k] = np.add( input_dict['HlambdaHt'][k] , np.multiply(input_dict['H_colcov'][:,:,rr] , np.array(input_dict['lambda_R'][k][rr])) )
                else: # size(H_colcov,3) == G
                    for g in range (0,1):#G):
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + np.dot(input_dict['H_colcov'][:,:,g] , np.dot( np.transpose(input_dict['Gmat'][:,g]) , input_dict['lambda_R'][k])) 
                        
            #%% Update W 
            tmpL = np.multiply(input_dict['XtDX'][k] , input_dict['HlambdaHt'][k]) + ( (1./input_dict['prior_W_var']) * identity(input_dict['L'])).astype('float64') 
            tmpCov = inv_prescale(tmpL);
            input_dict['W_rowcov'][k] = (np.float64(0.5)*(tmpCov+np.transpose(tmpCov))).astype('float64') 
            spm=np.dot( np.transpose(input_dict['X'][k]) , input_dict['Y'][k]) 
            tmpM = np.diag(np.dot( np.dot( spm , np.diag(np.array(input_dict['lambda_R'][k].flatten())[0]) ) , np.transpose(input_dict['H']) )) * input_dict['DD'][k]
            input_dict['W'][k] = np.dot(np.matrix(tmpM) , input_dict['W_rowcov'][k])
            input_dict['WtW'][k] = np.matrix(np.dot(np.transpose(input_dict['W'][k]), input_dict['W'][k]) + input_dict['W_rowcov'][k]) 
                        
    output_HlamW_dict={'HlambdaHt':input_dict['HlambdaHt'],'W':input_dict['W'], 'WtW':input_dict['WtW'], 'W_rowcov':input_dict['W_rowcov']}
    return output_HlamW_dict 


def update_lambda(input_dict):
     for k in range (0,input_dict['K']):   
            tmp_diagHtWXtDXWH = np.sum( np.multiply(input_dict['H'] , np.dot(np.multiply(input_dict['WtW'][k],input_dict['XtDX'][k]),input_dict['H']))  , 0 )
            if size(input_dict['H_colcov'].shape)==2:
                tmp_diagHtWXtDXWH = tmp_diagHtWXtDXWH + np.dot( np.dot( np.multiply(input_dict['WtW'][k].flatten(1) ,np.matrix(input_dict['XtDX'][k].flatten(1))) , input_dict['H_colcov'].reshape(input_dict['L']*input_dict['L'],1
,order='F')), np.matrix(input_dict['Gmat']));
            else:
                if input_dict['H_colcov'].shape[2]==input_dict['R']:
                    tmp_diagHtWXtDXWH = tmp_diagHtWXtDXWH + np.dot( np.multiply(input_dict['WtW'][k].flatten(1) ,np.matrix(input_dict['XtDX'][k].flatten(1))) , input_dict['H_colcov'].reshape(input_dict['L']*input_dict['L'],input_dict['H_colcov'].shape[2]
,order='F'));
                else: # not tested !!!!!!!!!!!!!!!!!!!!!!
                    tmp_diagHtWXtDXWH = tmp_diagHtWXtDXWH + np.dot( np.dot( np.multiply(input_dict['WtW'][k].flatten(1) ,np.matrix(input_dict['XtDX'][k].flatten(1))) , input_dict['H_colcov'].reshape(input_dict['L']*input_dict['L'],input_dict['H_colcov'].shape[2]
,order='F')), np.transpose(input_dict['Gmat']));
            input_dict['lambda_c'][k] = (input_dict['DD'][k]*input_dict['N'][k]/2) * np.ones([input_dict['R'],1]); #% [Rx1]
            input_dict['lambda_binv'][k] = ((0.5*input_dict['DD'][k]* np.matrix(np.sum(np.square(input_dict['Y'][k]),0)) ) - (np.dot(np.multiply(np.dot(np.transpose(np.matrix(input_dict['X'][k])), input_dict['Y'][k]) ,input_dict['H']).T , input_dict['W'][k].flatten(1).T) * input_dict['DD'][k]).T + (0.5*tmp_diagHtWXtDXWH) ).T #% [Rx1]

            if input_dict['opts']['lambda_dims'] == 'R':
                1  #% OK!  lambda_c and lambda_binv are already Rx1
            elif input_dict['opts']['lambda_dims'] == 'G': # alb--> option G tested in python?
                input_dict['lambda_c'][k] = np.dot( np.matrix(input_dict['Gmat']).T , input_dict['lambda_c'][k]);
                input_dict['lambda_binv'][k] = np.dot( np.transpose(input_dict['Gmat']) , input_dict['lambda_binv'][k]);
            elif input_dict['opts']['lambda_dims'] == 'o':
                input_dict['lambda_c'][k] = np.sum(input_dict['lambda_c'][k]);
                input_dict['lambda_binv'][k] = np.sum(input_dict['lambda_binv'][k]);
            else:
                print('Unimpleneted')
            input_dict['lambda_c'][k] = input_dict['lambda_c'][k] + input_dict['prior_lambda_c'][k];
            input_dict['lambda_binv'][k] = input_dict['lambda_binv'][k] + (1./input_dict['prior_lambda_b'][k]);
            input_dict['Lambda'][k] = np.divide(input_dict['lambda_c'][k] , input_dict['lambda_binv'][k] ); #% [Rx1 or Gx1 or 1x1]#assert(all(lambda{k}>0))
            input_dict['lambda_log'][k] = sc.special.psi(input_dict['lambda_c'][k]) - np.log(input_dict['lambda_binv'][k])
            if input_dict['opts']['lambda_dims'] == 'R':
                input_dict['lambda_R'][k] = input_dict['Lambda'][k] + np.zeros([input_dict['R'],1]);
                input_dict['lambda_log_R'][k] = input_dict['lambda_log'][k] + np.zeros([input_dict['R'],1]);
            elif input_dict['opts']['lambda_dims'] == 'G': # alb--> G option not tested in python?
                input_dict['lambda_R'][k] = np.dot(input_dict['Gmat'] , input_dict['Lambda'][k]);
                input_dict['lambda_log_R'][k] = np.dot(input_dict['Gmat'] , input_dict['lambda_log'][k]);
            elif input_dict['opts']['lambda_dims'] == 'o': # same as case 'R"
                input_dict['lambda_R'][k] = np.matrix(input_dict['Lambda'][k] + np.zeros([input_dict['R'],1]));
                input_dict['lambda_log_R'][k] = np.matrix(input_dict['lambda_log'][k] + np.zeros([input_dict['R'],1]))
            else:
                print('Unimpleneted')
            #%% Calculate <H*lambda{k}*H'>
            input_dict['HlambdaHt'][k] = np.dot( np.dot(input_dict['H'], np.diag(np.array(input_dict['lambda_R'][k].flatten())[0])) , input_dict['H'].T)
            if size(input_dict['H_colcov'].shape)==2:
                        sss=np.dot(np.matrix(input_dict['Gmat']),input_dict['lambda_R'][k])
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + ( input_dict['H_colcov'] * sss[0,0]  )                
            else:
                if input_dict['H_colcov'].shape[2] == input_dict['R']:
                    for r in range (0,input_dict['R']):
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + (input_dict['H_colcov'][:,:,r] * np.array(input_dict['lambda_R'][k][r]))
                else:# alb--> next options are not tested in python
                    for g in range (0,1):#G):
                        input_dict['HlambdaHt'][k] = input_dict['HlambdaHt'][k] + np.dot( input_dict['H_colcov'][:,:,g], np.dot(np.transpose(np.matrix(input_dict['Gmat'][:,g])),input_dict['lambda_R'][k])) 
                        
     output_Lambda_dict={'Lambda':input_dict['Lambda'],'HlambdaHt':input_dict['HlambdaHt'],'lambda_log_R':input_dict['lambda_log_R'], 'lambda_R':input_dict['lambda_R'],
                         'lambda_log':input_dict['lambda_log'], 'lambda_binv':input_dict['lambda_binv'],'lambda_c':input_dict['lambda_c']}       
     return output_Lambda_dict 
    
def compute_F(input_dict): #NEED TO IMPROVE SUM_DIMS ...
              
    #for key,val in input_dict.items(): #load all
    #         exec(key + '=val')
    
    #ns=Namespace(**input_dict) #load all in python 3
    #locals().update(input_dict)
    
    #eta_log
    Fpart=input_dict['Fpart']         
    F = np.nan;  
    Fpart["Hprior"]=(sum_dims(np.dot(input_dict['eta_log'],np.matrix(input_dict['Gmat'])),[input_dict['L'], input_dict['R']])/2)- (np.log(2*np.pi)*input_dict['L']*input_dict['R']/2)- (sum_dims(np.multiply(input_dict['eta'],np.matrix(input_dict['H2Gmat']).T),[input_dict['L'],input_dict['G']])/2) 
    if size(input_dict['H_colcov'].shape)==2: #case lambda='o' 
        tmp1=logdet(input_dict['H_colcov'],'chol')
        Fpart["Hpost"] = 0.5*input_dict['L']*input_dict['R']*(1+2*np.pi) + 0.5*np.sum(input_dict['Gmat'])*tmp1;
    else: #case lambda='R' 
        tmp1= apply3_logdet(input_dict['H_colcov'],'chol')
        Fpart["Hpost"] = 0.5*input_dict['L']*input_dict['R']*(1+2*np.pi) + 0.5* sum_dims(tmp1,[1, 1, input_dict['R']])
    Fpart["etaPrior"] = -sum_dims(np.matrix(sc.special.gammaln(input_dict['prior_eta_c'])),[input_dict['L'], input_dict['G']]) +sum_dims(np.matrix(np.multiply(input_dict['prior_eta_c']-1,input_dict['eta_log'])),[input_dict['L'], input_dict['G']]) -sum_dims(np.matrix(input_dict['prior_eta_c']*np.log(input_dict['prior_eta_b'])),[input_dict['L'], input_dict['G']])  -sum_dims(np.matrix(input_dict['eta']/input_dict['prior_eta_b']),[input_dict['L'], input_dict['G']]);
    Fpart["etaPost"] = sum_dims(np.matrix(sc.special.gammaln(input_dict['eta_c'])),[input_dict['L'], input_dict['G']]) -sum_dims(np.multiply((input_dict['eta_c']-1),input_dict['eta_log']),[input_dict['L'], input_dict['G']])  +sum_dims(np.multiply(-input_dict['eta_c'],np.log(input_dict['eta_binv'])),[input_dict['L'], input_dict['G']]) +sum_dims(np.multiply(input_dict['eta'],input_dict['eta_binv']),[input_dict['L'], input_dict['G']]);            
    for kk in range(0,input_dict['K']):
        Fpart["Wprior"][kk] = sum_dims(np.matrix(np.log(1./input_dict['prior_W_var'],dtype="float64"),dtype="float64"),[1, input_dict['L']])/2 - np.log(2*np.pi,dtype="float64")*1*input_dict['L']/2 - trace(input_dict['WtW'][kk])/2/input_dict['prior_W_var'];
        Fpart["Wpost"][kk] = 0.5*1*input_dict['L']*(1+2*np.pi) + 0.5*logdet(input_dict['W_rowcov'][kk],'chol');
        Fpart["muPrior"][kk] = -0.5/input_dict['prior_mu_var']*sum_dims(np.matrix(input_dict['mu2'][kk]),[3, input_dict['L']])  +0.5*np.log(2*np.pi*input_dict['prior_mu_var'],dtype="float64") * 3*input_dict['L'];
        Fpart["muPost"][kk] = 0.5*(1+np.log(2*np.pi,dtype="float64"))*3*input_dict['L'] +0.5*sum_dims(np.matrix(np.log(input_dict['mu_var'][kk],dtype="float64")),[3, input_dict['L']]);
        Fpart["betaPrior"][kk] = -np.mean(np.mean(sc.special.gammaln(input_dict['prior_beta_c'][kk])))*3*input_dict['L'] +np.mean(np.mean( np.multiply( (input_dict['prior_beta_c'][kk]-1) , input_dict['beta_log'][kk])))*3*input_dict['L'] -  np.mean(np.mean(  np.multiply(input_dict['prior_beta_c'][kk],np.log(input_dict['prior_beta_b'][kk]))))*3*input_dict['L'] -np.mean(np.mean( np.multiply( 1./input_dict['prior_beta_b'][kk], input_dict['beta'][kk])))*3*input_dict['L'];
        Fpart["betaPost"][kk] = sum_dims(np.matrix(sc.special.gammaln(input_dict['beta_c'][kk])),[3, input_dict['L']]) -sum_dims(  np.matrix(np.multiply((input_dict['beta_c'][kk]-1),input_dict['beta_log'][kk])),[3, input_dict['L']]) +sum_dims(  np.matrix(np.multiply(input_dict['beta_c'][kk],-np.log(input_dict['beta_binv'][kk]))),[3, input_dict['L']]) +sum_dims(  np.matrix(np.multiply(input_dict['beta_binv'][kk],input_dict['beta'][kk])),[3, input_dict['L']]);            
        Fpart["piPrior"][kk] = sum_dims(np.matrix( sc.special.gammaln( sum_dims(np.matrix(input_dict['prior_pi_weights'][kk]),[3, 0]) )), [1, input_dict['L']] )  -sum_dims( np.matrix(sc.special.gammaln( input_dict['prior_pi_weights'][kk] )), [3, input_dict['L']]) +sum_dims( np.multiply( (input_dict['prior_pi_weights'][kk]-1) , input_dict['pi_log'][kk]), [3, input_dict['L']]);
        Fpart["piPost"][kk] = -sum_dims( np.matrix(sc.special.gammaln( sum_dims(np.matrix(input_dict['pi_weights'][kk]),[3, 0]) )), [1, input_dict['L']]) +sum_dims( np.matrix(sc.special.gammaln( input_dict['pi_weights'][kk] )), [3, input_dict['L']])  -sum_dims( np.matrix(np.multiply( (input_dict['pi_weights'][kk]-1) , input_dict['pi_log'][kk])), [3, input_dict['L']]);
        Fpart["qPrior"][kk] = sum_dims( np.matrix(np.multiply(input_dict['sumN_Dq'][kk] , input_dict['pi_log'][kk])), [3, input_dict['L']]);
        Fpart["qPost"][kk] = - sum_dims(np.matrix(input_dict['sumN_Dqlogq'][kk]), [3, input_dict['L']]);
        Fpart["Ylike1"][kk] = input_dict['N'][kk]*input_dict['DD'][kk]/2 * sum_dims(input_dict['lambda_log_R'][kk]-np.log(2*np.pi,dtype="float64"),[input_dict['R'], 1]);
        Fpart["Ylike2"][kk] = -0.5*input_dict['Y2D_sumN'][kk]*input_dict['lambda_R'][kk];
        Fpart["Ylike3"][kk] = input_dict['DD'][kk] * (np.dot( np.sum( np.multiply(input_dict['Y'][kk]  , np.dot(np.dot(input_dict['X'][kk],np.diagflat(input_dict['W'][kk])),input_dict['H'])),0) ,input_dict['lambda_R'][kk]));
        Fpart["Ylike4"][kk] = -0.5 * sum_dims(np.matrix( np.multiply(np.multiply( input_dict['XtDX'][kk] , input_dict['HlambdaHt'][kk]) , input_dict['WtW'][kk])), [input_dict['L'], input_dict['L']]);
        Fpart["lambdaPrior"][kk] = -np.sum(sc.special.gammaln(input_dict['prior_lambda_c'][kk])) +np.sum(np.multiply((input_dict['prior_lambda_c'][kk]-1),input_dict['lambda_log'][kk])) -np.sum(np.multiply(input_dict['prior_lambda_c'][kk],np.log(input_dict['prior_lambda_b'][kk]))) -np.sum(1./np.multiply(input_dict['prior_lambda_b'][kk],input_dict['Lambda'][kk]));
        Fpart["lambdaPost"][kk] = np.sum(sc.special.gammaln(input_dict['lambda_c'][kk])) -np.sum(np.multiply((input_dict['lambda_c'][kk]-1),input_dict['lambda_log'][kk])) -np.sum(np.multiply(input_dict['lambda_c'][kk],np.log(input_dict['lambda_binv'][kk]))) +np.sum(np.multiply(input_dict['lambda_binv'][kk],input_dict['Lambda'][kk]));
        Fpart["XPrior"][kk] = sum_dims( np.matrix((0.5 * np.multiply( (input_dict['beta_log'][kk]-np.log(2*np.pi,dtype="float64")) , input_dict['sumN_Dq'][kk])) - (0.5 * np.multiply(input_dict['beta'][kk] , input_dict['sumN_DqXq2'][kk])) + np.multiply( np.multiply(input_dict['beta'][kk] , input_dict['mu'][kk]) , input_dict['sumN_DqXq'][kk])  - (0.5* np.multiply( np.multiply( input_dict['beta'][kk] , input_dict['mu2'][kk]) , input_dict['sumN_Dq'][kk]))) , [3, input_dict['L']]);
        Fpart["XPost"][kk] = -sum_dims(np.matrix( -0.5*  np.multiply( input_dict['sumN_Dq'][kk], (1+np.log(2*np.pi,dtype="float64")+np.log(input_dict['Xq_var'][kk],dtype="float64")).T)), [3, input_dict['L']])
    #F = np.sum(np.sum(Fpart.values()),dtype="float64") #np.sum(sum([i for i in Fpart.values()])) #sum_carefully(Fpart); % add up all the bits 
    F = sum(sum(Fpart.values()))#,dtype="float64") #np.sum(sum([i for i in Fpart.values()])) #sum_carefully(Fpart); % add up all the bits 

    return F, Fpart