
from pylab import find, tile
import copy
import numpy as np
import flica_various as alb_various

def flica_reorder(M):
    K=len(M['X'])
    R=M['H'].shape[1]
    for k in range(0,K):
        #M.X{k} * diag(M.W{k}.*sqrt( M.H.^2 * makesize(M.lambda{k},[R 1]) * M.DD(k))')]; %#ok<AGROW>
        if np.matrix(M['lambda'][k]).shape[0]==R:
            tmp=np.dot(np.square(M['H']),M['lambda'][k])
        else:
            tmp=np.dot(np.square(M['H']),tile(M['lambda'][k],[R,1]))
        tmp2=np.sqrt(np.dot(tmp,M['DD'][k]))
        tmp3=np.diag(np.multiply(M['W'][k],tmp2))
        tmp4=np.dot(M['X'][k],np.diag(tmp3))
        if k==0:
            Xcat=copy.deepcopy(tmp4)
        else:     
            Xcat=np.concatenate((Xcat,tmp4))
            
    weight = np.sum(np.square(Xcat),0)      
    order=np.argsort(weight)
    order=order[::-1]
    weight=weight[order]
    order[find(weight==0)]=[]
    weight[find(weight==0)]=[];
    polarity = np.sign(np.sum(np.multiply(np.square(Xcat),(Xcat>0)),0)  - np.sum(np.multiply(np.square(Xcat),(Xcat<0)),0)  ) 
    polarity = polarity[order].astype('int');
    
    #assert(all(weight>0) && all(abs(polarity)==1))
    # Do some tidy rescaling as well, while we're at it
    # H should have unit RMS
    rescaleH = np.divide(polarity, alb_various.rms(M['H'][order,:],1,[])) #'; ## ./
    rescaleW=[np.array(a) for a in range (0,K)]
    rescaleX=[np.array(a) for a in range (0,K)]
    for k in range(0,K):
        # W should all equal 1
        rescaleW[k] = 1./M['W'][k][:,order]
        # X should take up the remaining scaling
        rescaleX[k] = 1./rescaleH/rescaleW[k];
        #assertalmostequal(rescaleH.*rescaleW{k}.*rescaleX{k}, ones(size(polarity)));
        
    fieldnames= M.keys()
    for f in fieldnames:
         
         if f is 'H': # first dimension needs rescaleH
              M['H'] = np.dot(np.diag(rescaleH),M['H'][order,:])
              
         if f is 'H_PCs': # reorder dim2
              M['H_PCs'] = M['H_PCs'][:,order]
             
         if f is 'W': #second dimension needs rescaleW[k]
              for k in range(0,K):
                   M['W'][k] = np.dot(M['W'][k][:,order],np.diag(np.array(rescaleW[k])[0]))
                   
         if f is 'X': #second dimension needs rescaleX[k]
              for k in range(0,K):
                   M['X'][k] = np.dot(M['X'][k][:,order],np.diag(np.array(rescaleX[k])[0]))  
                   
         if f is 'mu': #second dimension needs rescaleX[k]
              for k in range(0,K):
                   M['mu'][k] = np.dot(M['mu'][k][:,order],np.diag(np.array(rescaleX[k])[0])) 
          
         #I ignore this         
         #case {'X','Xq','mu'} % second dimension needs rescaleX{k}; apply to each slice.
         #  for k=1:K
         #      M.(fn){k} = apply3(@(Z) Z*diag(rescaleX{k}), M.(fn){k}(:,order));
         #  end   
          
         
         if f is 'beta': #second dimension needs rescaleX[k]^-2
              for k in range(0,K):
                   M['beta'][k] = np.dot(M['beta'][k][:,order],np.diag(np.power(np.array(rescaleX[k])[0], -2)) )
                   
         if f is 'pi': #reorder dim 2
              for k in range(0,K):
                   M['pi'][k] = M['pi'][k][:,order] 
                   
                   
         if f is 'pi_mean': #reorder dim 2
              for k in range(0,K):
                   M['pi_mean'][k] = M['pi_mean'][k][:,order] 
                   
    for k in range(0,K):               
        if k==0:
            tmp=np.array(M['W'][0])[0]
        else:     
            tmp=np.concatenate((tmp,np.array(M['W'][k])[0]))
            
    #check=alb_various.rms(tmp-1,[],[])
    #if check < np.exp(-12)
                   
                   
                   
                   
    
    return M