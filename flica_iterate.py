#MAIN LOOP OF LINKED ICA INFERENCE
#INPUT: DATA Y, OPTIONS OPTS, AND Priors, Posteriors, Constants BEING OUTPUT OF FLICA_INIT_PARAMS
#OUTPUT: A DICTIONARY CONTAINING THE RESULT OF THE FACTORIZATION (Y_k = X_k*W_k*H )       
#   most relevant outputs: Xk contains spatial maps, H contains 'subject scores' which are usually correlated with e.g behavioral or demographic measures 
  
import time

import numpy as np;

from flica_small_updates import update_eta
from flica_small_updates import update_H
from flica_small_updates import update_HlambdaHt_and_W
#from flica_small_updates import update_X_k_i
from flica_small_updates import update_X_k

from flica_small_updates import update_mixmod
from flica_small_updates import update_lambda
from flica_small_updates import compute_F

#to accelerate np.dot
#from scipy.linalg import get_blas_funcs 
#gemm = get_blas_funcs("gemm") # this is to use instead of np.dot(a,b) as gemm(1,a,b)
#from flica_jit import better_dot
#from numba import jit
#@jit
#def better_dot(A,B):
#    return np.dot(A,B)
def zeros32(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.zeros(*args, **kwargs)
    

def flica_iterate(Y,opts,Priors, Posteriors, Constants):               
    #define list to keep info for free energy
    
    # Fpart = {k: zeros32(Constants['k']) for k in [listofvals]}
        
    Fpart = {"Hprior":np.zeros(1),"Hpost":np.zeros(1),
             "etaPrior":np.zeros(1),"etaPost":np.zeros(1),
             "Wprior":np.zeros(Constants['K']),"Wpost":np.zeros(Constants['K']),
             "muPrior":np.zeros(Constants['K']),"muPost":np.zeros(Constants['K']),
             "betaPrior":np.zeros(Constants['K']), "betaPost":np.zeros(Constants['K']),
             "piPrior":np.zeros(Constants['K']), "piPost":np.zeros(Constants['K']), 
             "qPrior":np.zeros(Constants['K']), "qPost":np.zeros(Constants['K']), 
             "Ylike1":np.zeros(Constants['K']), "Ylike2":np.zeros(Constants['K']), 
             "Ylike3":np.zeros(Constants['K']), "Ylike4":np.zeros(Constants['K']), 
             "lambdaPrior":np.zeros(Constants['K']),"lambdaPost":np.zeros(Constants['K']) ,
             "XPrior":np.zeros(Constants['K']),"XPost":np.zeros(Constants['K'])} 
    
    F_history = [];
    change_H = np.array([]);
    convergence_flag=0
    its=-1
    FE_eval_space= np.sort(np.concatenate((np.concatenate((np.arange(2,opts['maxits'],50), np.arange(2,opts['maxits'],50)-1),axis=None),np.array([opts['maxits']+1,opts['maxits']])),axis=None))  -1    
    # iterate the updates
    flag2=0
    while convergence_flag == 0 : 
         its=its+1
         print('its = %s' % its)
         tt=time.time()
         
 ## Update eta  
         input_eta_update={'prior_eta_b': Priors['prior_eta_b'],'prior_eta_c': Priors['prior_eta_c'],
                           'H2Gmat':Posteriors['H2Gmat'],'Gmat':Posteriors['Gmat'],'L':Constants['L']}
         
         output_eta_dict = update_eta(input_eta_update)  
         
         Posteriors['eta_binv']=output_eta_dict['eta_binv']
         Posteriors['eta_c']=output_eta_dict['eta_c']
         Posteriors['eta']=output_eta_dict['eta']
         Posteriors['eta_log']=output_eta_dict['eta_log']

         
## Update H : depends on lamda_dims (R or ) and iterates over K
         old_H=Posteriors['H']
         input_H_update={'opts':opts,'Y':Y,'NH':Constants['NH'],
                         'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                         'eta':Posteriors['eta'],'Gmat':Posteriors['Gmat'],'Lambda':Posteriors['Lambda'],'lambda_R':Posteriors['lambda_R'],
                         'XtDX':Posteriors['XtDX'] ,'WtW':Posteriors['WtW'],
                         'K':Constants['K'],'R':Constants['R'], 'L':Constants['L'],'DD':Constants['DD']}
         
         output_H_dict = update_H(input_H_update)
         
         Posteriors['H']=output_H_dict['H']
         Posteriors['H2Gmat']=output_H_dict['H2Gmat']
         Posteriors['H_colcov']=output_H_dict['H_colcov']
         Posteriors['W']=output_H_dict['W']
         Posteriors['H_PCs']=output_H_dict['H_PCs']
         Posteriors['tmp_R_to_NH']=output_H_dict['tmp_R_to_NH']          

         
#update H*lambda{k}*H'> and also W : both iterate over K together; update W requires hugh matrix mult
         #tt2=time.time()
         input_HlamW_update={'Y':Y,'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                            'K':Constants['K'],'R':Constants['R'], 'L':Constants['L'],'DD':Constants['DD'],
                            'HlambdaHt':Posteriors['HlambdaHt'],'lambda_R':Posteriors['lambda_R'],'H_colcov':Posteriors['H_colcov'],
                            'Gmat':Posteriors['Gmat'],'XtDX':Posteriors['XtDX'] ,'WtW':Posteriors['WtW'],
                            'prior_W_var':Priors['prior_W_var'] ,'W_rowcov':Posteriors['W_rowcov']}
         
         output_HlamW_dict = update_HlambdaHt_and_W(input_HlamW_update)
         
         Posteriors['HlambdaHt']=output_HlamW_dict['HlambdaHt']
         Posteriors['W']=output_HlamW_dict['W']
         Posteriors['WtW']=output_HlamW_dict['WtW']
         Posteriors['W_rowcov']=output_HlamW_dict['W_rowcov']
         #print 'cost_HlambdaHt =', time.time()-tt2
         
         
         
         
#Update X: ITERATES OVER K AND OVER L
         
         tt2=time.time() 
         for k in range(Constants['K']):
             input_X_k_update={'X_k':Posteriors['X'][k],'H':Posteriors['H'], 'Y_k':Y[k],'L':Constants['L'],
                              'DD_k':Constants['DD'][k],'X2_k':Posteriors['X2'][k],'lambda_R_k':Posteriors['lambda_R'][k],
                              'W_k':Posteriors['W'][k],'WtW_k':Posteriors['WtW'][k],'HlambdaHt_k':Posteriors['HlambdaHt'][k],
                              'beta_k':Posteriors['beta'][k],'mu_k':Posteriors['mu'][k],
                              'Xq_var_k':Posteriors['Xq_var'][k],'sumN_Dq_k':Posteriors['sumN_Dq'][k,:,:],'sumN_DqXq_k':Posteriors['sumN_DqXq'][k,:,:],
                              'sumN_DqXq2_k':Posteriors['sumN_DqXq2'][k,:,:],'sumN_Dqlogq_k':Posteriors['sumN_Dqlogq'][k,:,:],
                              'beta_log_k':Posteriors['beta_log'][k],'mu2_k':Posteriors['mu2'][k],'pi_log_k':Posteriors['pi_log'][k]}

            ##@jit(nopython=True, parallel=True)
             output_X_k_dict= update_X_k(input_X_k_update)

             Posteriors['X'][k]=output_X_k_dict['X_k']
             Posteriors['X2'][k]=output_X_k_dict['X2_k']
             Posteriors['sumN_Dqlogq'][k]=output_X_k_dict['sumN_Dqlogq_k']
             Posteriors['sumN_DqXq2'][k]=output_X_k_dict['sumN_DqXq2_k']
             Posteriors['sumN_DqXq'][k]=output_X_k_dict['sumN_DqXq_k']
             Posteriors['sumN_Dq'][k]=output_X_k_dict['sumN_Dq_k']
             Posteriors['Xq_var'][k]=output_X_k_dict['Xq_var_k']
             
         print('cost X ',time.time()-tt2)    

         
         

#%% UPDATE THE MIXTURE MODELS         
         input_mixmod_update={'X':Posteriors['X'],'X2':Posteriors['X2'],'XtDX':Posteriors['XtDX'],
                     'K':Constants['K'], 'DD':Constants['DD'],
                     'pi_weights':Posteriors['pi_weights'],'prior_pi_weights':Priors['prior_pi_weights'],'sumN_Dq':Posteriors['sumN_Dq'], 
                     'pi_mean':Posteriors['pi_mean'],'pi_log':Posteriors['pi_log'],
                     'beta':Posteriors['beta'],'beta_log':Posteriors['beta_log'],'beta_c':Posteriors['beta_c'],'prior_beta_c':Priors['prior_beta_c'] ,
                     'beta_binv':Posteriors['beta_binv'],'prior_beta_b':Priors['prior_beta_b'], 
                     'mu':Posteriors['mu'],'mu2':Posteriors['mu2'],'sumN_DqXq':Posteriors['sumN_DqXq'],'sumN_DqXq2':Posteriors['sumN_DqXq2'],
                     'mu_var':Posteriors['mu_var'],'prior_mu_var':Priors['prior_mu_var'] ,'prior_mu_mean':Priors['prior_mu_mean'],
                         }  
         output_mixmod_dict=update_mixmod(input_mixmod_update)
         
         Posteriors['XtDX']=output_mixmod_dict['XtDX']
         Posteriors['pi_weights']=output_mixmod_dict['pi_weights']
         Posteriors['pi_log']=output_mixmod_dict['pi_log']
         Posteriors['pi_mean']=output_mixmod_dict['pi_mean']
         Posteriors['beta_c']=output_mixmod_dict['beta_c']
         Posteriors['beta_binv']=output_mixmod_dict['beta_binv']
         Posteriors['beta']=output_mixmod_dict['beta']
         Posteriors['beta_log']=output_mixmod_dict['beta_log']
         Posteriors['mu']=output_mixmod_dict['mu']
         Posteriors['mu_var']=output_mixmod_dict['mu_var']
         Posteriors['mu2']=output_mixmod_dict['mu2']

         
#%% Update P'(lambda)
         input_lambda_update={'opts':opts,'Y':Y,'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                         'K':Constants['K'], 'L':Constants['L'],'DD':Constants['DD'],'R':Constants['R'],'N':Constants['N'],                        
                         'WtW':Posteriors['WtW'],'XtDX':Posteriors['XtDX'], 'H_colcov':Posteriors['H_colcov'],'Gmat':Posteriors['Gmat'],
                         'lambda_c':Posteriors['lambda_c'],'lambda_binv':Posteriors['lambda_binv'],
                         'Lambda':Posteriors['Lambda'],'lambda_log':Posteriors['lambda_log'],
                         'lambda_log_R':Posteriors['lambda_log_R'],'lambda_R':Posteriors['lambda_R'],'HlambdaHt':Posteriors['HlambdaHt'],
                         'prior_lambda_c':Priors['prior_lambda_c'],'prior_lambda_b':Priors['prior_lambda_b']}                         
                                                 
         #tt2=time.time()
         output_lambda_dict = update_lambda(input_lambda_update)
         
         Posteriors['Lambda']=output_lambda_dict['Lambda']
         Posteriors['HlambdaHt']=output_lambda_dict['HlambdaHt']
         Posteriors['lambda_log_R']=output_lambda_dict['lambda_log_R']
         Posteriors['lambda_R']=output_lambda_dict['lambda_R']
         Posteriors['lambda_log']=output_lambda_dict['lambda_log']
         Posteriors['lambda_binv']=output_lambda_dict['lambda_binv']
         Posteriors['lambda_c']=output_lambda_dict['lambda_c']
         #print 'cost_lambda =', time.time()-tt2
                           
#%% Compute F, if desired
         if its in FE_eval_space:
             input_FE_computation={'Fpart':Fpart, 'Y':Y,'X':Posteriors['X'],'H':Posteriors['H'],'W':Posteriors['W'],
                             'K':Constants['K'], 'L':Constants['L'],'DD':Constants['DD'],'R':Constants['R'],'N':Constants['N'],'G':Constants['G'], 
                             'H_colcov':Posteriors['H_colcov'],'H2Gmat':Posteriors['H2Gmat'],'W_rowcov':Posteriors['W_rowcov'],
                             'WtW':Posteriors['WtW'],'mu2':Posteriors['mu2'],'mu_var':Posteriors['mu_var'],'Gmat':Posteriors['Gmat'],
                             'beta':Posteriors['beta'],'beta_log':Posteriors['beta_log'],'beta_c':Posteriors['beta_c'],'beta_binv':Posteriors['beta_binv'],
                             'pi_log':Posteriors['pi_log'],'pi_weights':Posteriors['pi_weights'],                         
                             'eta':Posteriors['eta'],'eta_c':Posteriors['eta_c'],'eta_binv':Posteriors['eta_binv'],'eta_log':Posteriors['eta_log'],
                             'sumN_Dqlogq':Posteriors['sumN_Dqlogq'],'lambda_log_R':Posteriors['lambda_log_R'],
                             'Y2D_sumN':Posteriors['Y2D_sumN'],'lambda_R':Posteriors['lambda_R'],'lambda_log':Posteriors['lambda_log'],
                             'lambda_binv':Posteriors['lambda_binv'],'lambda_c':Posteriors['lambda_c'],'Lambda':Posteriors['Lambda'],
                             'HlambdaHt':Posteriors['HlambdaHt'],'sumN_Dq':Posteriors['sumN_Dq'],'XtDX':Posteriors['XtDX'],
                             'mu':Posteriors['mu'],'sumN_DqXq':Posteriors['sumN_DqXq'],'sumN_DqXq2':Posteriors['sumN_DqXq2'],'Xq_var':Posteriors['Xq_var'],
                             'prior_eta_b': Priors['prior_eta_b'],'prior_eta_c': Priors['prior_eta_c'],
                             'prior_W_var':Priors['prior_W_var'] ,'prior_mu_var':Priors['prior_mu_var'] ,
                             'prior_beta_c':Priors['prior_beta_c'],'prior_beta_b':Priors['prior_beta_b'],
                             'prior_pi_weights':Priors['prior_pi_weights'],'prior_lambda_c':Priors['prior_lambda_c'],
                             'prior_lambda_b':Priors['prior_lambda_b']}
                                                     
             #tt2=time.time()
             F, Fpart = compute_F(input_FE_computation)
             F_history.append(F);
             #print 'cost_F =', time.time()-tt2
             #print 'F =', F
             
         if its>0:    
            #dH=np.divide(np.linalg.norm(old_H-Posteriors['H']),np.prod(old_H.shape))
            dH=np.max(np.absolute(old_H-Posteriors['H']))
            change_H=np.append(change_H,dH)
            #1-diag(abs(Htnew*Hold))
            print(change_H[-1])
            
            if (change_H[-1] < opts['tol']) & (flag2==0):
                flag2=1
                FE_eval_space=np.sort(np.concatenate((FE_eval_space,np.array([its+1,its+2])),axis=None))
     
            #import pdb;pdb.set_trace()
            if (its > (opts['maxits']-1) ) or  (flag2==1): #| (change_H[-1] < opts['tol']): #| (dF<
                convergence_flag=1
                dF=np.divide(F_history[-1]-F_history[-2],F_history[-1])
                #GATHER OUTPUT
                FLICA_OUTPUT_DICT ={"H":Posteriors['H'],"lambda":Posteriors['Lambda'],"W":Posteriors['W'],"beta":Posteriors['beta'],"mu":Posteriors['mu'],
                "pi":Posteriors['pi_mean'],"X":Posteriors['X'], "H_PCs":Posteriors['H_PCs'],"F":F,"F_history":F_history,
                                    "DD":Constants['DD'],"change_H":change_H,"opts":opts}              
         print('cost_it =',time.time()-tt)
         
    return FLICA_OUTPUT_DICT # the old M 






