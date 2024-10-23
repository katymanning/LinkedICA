import numpy as np;
import flica_various as various
#from numba import jit

def flica_parseoptions(R, opts={"num_components":10,"maxits":2000,"dof_per_voxel":"auto_eigenspectrum",
                                "lambda_dims":"R","initH":"PCA",'fs_path':'/Applications/freesurfer'}):
#set default options if not provided and do some small checks
#to do: add here more controls and maybe getting info for later on saving the data in right format etc....
    
    # Check any options that need to refer to the data:
    
    #R = Y[0].shape[1]; #number of subjects
    if  opts['num_components'] > R/4:
        print('Consider using more subjects??')#, opts['initH'],'check me'
        
    if type(opts['initH'])!=str:
	#import pdb; pdb.set_trace()
        if opts['initH'].shape != np.ones([opts['num_components'], R]).shape:
              print('The shape of the given H matrix does not have the right dimensions')
              print('Returning to fully unsupervised, ignoring your H ....')
              opts['initH']='PCA'
                            
    return opts

def logdet(M,ignorezeros):
    if ignorezeros=='chol': 
        ld = 2*np.sum(np.log(np.diag(np.linalg.cholesky(M)),dtype="float32")) 
    else:
       print('not implemented, not used in .m?')
       
    return ld
      
def apply3_logdet(X,ignorezeros):
    out=np.zeros([1,1,X.shape[2]]).astype('float32')
    for i in range (0,X.shape[2]):	
        out[:,:,i]=various.logdet(X[:,:,i],ignorezeros)    
    #test=np.apply_along_axis(various.logdet, 2, X,ignorezeros)#, *args, **kwargs)
    return out   
   


def sum_dims(M,dims):
#% Sum a matrix in various dimensions
#% BUT be prepared for the fact that the input matrix might be smaller than
#% it should be.
#% e.g. M is a 5x4 matrix and N is a 5x1 matrix, but conceptually they're
#% both the same size...
#%   sum_dims(M,[5 0]) = sum(M,1)
#%   sum_dims(N,[5 0]) = 5*sum(N,1) = 5*N
#%   sum_dims(M,[0 4]) = sum(M,2)
#%   sum_dims(N,[0 4]) = sum(N,2)
#%   sum_dims(M,[5 4]) = sum(M(:))
#%   sum_dims(N,[5 4]) = sum(N)*5.
#% An error will result if there's a size mismatch, e.g. sum_dims(M,[6 0]).  
    for d in range (0,len(dims)):
        if dims[d]==0:
            1
        elif dims[d]==M.shape[d]:
            M=np.sum(M,d,dtype="float64").astype("float32")
            if dims[d]==1: # added to match sum from matlab
                M=np.expand_dims(M,axis=d)
        elif (dims[d]>0) & (M.shape[d]==1):
            #M = M*dims[d] its correct
            M = np.multiply(M,dims[d],dtype="float32")
        else:# dims[d]>1 & M.shape[d]>1:
            print("some error, check .m")
    return M[0,0]

#from numba import jit
#@jit
#def better_dot(A,B):
#    return np.dot(A,B) 
#@jit(nopython=True,parallel=True, nogil=True)

#@jit(nopython=True)
#def better_dot(A,B):
#    return np.dot(A,B) #,np.ndarray([A.shape[0],B.shape[1]]).astype('float32'))    
    


def apply3_diag(X):
    out=np.zeros([X.shape[0],X.shape[0],X.shape[2]]).astype('float32');
    for i in range (0,X.shape[2]):	
		#out[:,:,i]=np.diagflat(X[:,:,i])
        out[:,:,i]=np.diagflat(X[:,:,i])
    return out

def apply3_diag2(X):
	out=np.zeros([X.shape[0],X.shape[2]]).astype('float32');
	for i in range (0,X.shape[2]):	
		#out[:,:,i]=np.diagflat(X[:,:,i])
		out[:,i]=np.diag(X[:,:,i])
	return out

def inv_prescale(inp):
	prescale = np.diag(np.power(np.diag(inp),np.float32(-.5)))
	out = np.dot( np.dot( prescale,  np.linalg.inv( np.dot(np.dot(prescale,inp),prescale)) )   ,prescale)
	return out

def apply3_inv_prescale(X):
	out=np.zeros([X.shape[0],X.shape[0],X.shape[2]]).astype('float32');
	for i in range (0,X.shape[2]):	
		out[:,:,i]=various.inv_prescale(X[:,:,i])

	return out

def rms(IN, dim, options):
	if dim==[]:  #I use only this case USED FLICA LOAD!!
		out = np.sqrt(np.sum(np.square(IN))/IN.size)# dumm = alb_various.rms(Y[k],0,[])
	else:
		out = np.sqrt(np.divide(np.sum(np.square(IN),dim),IN.shape[dim]))# dumm = alb_various.rms(Y[k],0,[])
	return out
