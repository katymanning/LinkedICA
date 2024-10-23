from pylab import find #* #for the find command or size
import nibabel as nib
import os 
#import copy
import numpy as np
#import flica_various as alb_various
import matplotlib.pyplot as plt
import pickle
#import json


def flica_save_everything(output_dir,M,fileinfo):


    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    
    f = open(output_dir + '/order_of_loaded_data.txt','w')
    
    for i in range(len(fileinfo['names'])):        
        f.write(fileinfo['names'][i] + fileinfo['filetype'][i])
        f.write('\n')
        
    f.close() 
        
    pickle.dump( M, open( os.path.join(output_dir, "M.p"), "wb" ) )
    # M=pickle.load(open('M.p','rb'))
    
    #fix to save save to json
    #json = json.dumps(M) 
    #f = open("M.json","w") 
    #f.write(json) 
    #f.close()
    
    K=len(M['X'])
    R=M['H'].shape[1]
    L=M['H'].shape[0]
   
    #Save H matrix to a text file. H CONTAINS THE SUBJECT COURSES, RELEVANT FOR LATER CORRELATION
    np.savetxt(output_dir+'/subjectCoursesOut.txt',M['H'])
        
    #Save lambda matrices -- useful for diagnosis actually,OPTS.LAMDA_DIMS='R'
    tmp=[np.array(a) for a in range (0,K)] # I us
    for k in range(0,K):
        tmp[k]= np.power(M['lambda'][k],-0.5)/fileinfo['scaling_data_transform'][k].astype(float)
        #tmp(:,k) = makesize(M.lambda{k}.^-0.5, [R 1]) / fileinfo.transforms{k,3}; check this in matlab
    
    #np.savetxt(output_dir+'/noiseStdevOut.txt',tmp)
    tmp2=np.squeeze(np.asarray(tmp))
    np.savetxt(output_dir+'/noiseStdevOut.txt',tmp2)


    #Plot the lambdas
    if np.max(np.matrix(tmp[k]).shape)==1:
        for k in range(0,K):
            plt.plot(range(0,R),np.tile(tmp[k],[R,1]))
    else:   
        for k in range(0,K):
            plt.plot(range(0,R),tmp[k])
        
    plt.title('Noise estimates by subject')   
    plt.ylabel('Noise standard deviation')
    plt.xlabel('Subject index')
    #plt.show()
    plt.savefig(output_dir+'/noiseStdev.png')
    plt.close()
    #f.show()
    #raw_input()
    
    # Save summary information to appropriate files
    ## M.H_PCs (absolute/relative info from each modality) or weight (total variance explained)
    #desome=np.round(np.divide(M['H_PCs'],np.tile(np.sum(M['H_PCs'],0),[M['H_PCs'].shape[0], 1] )  )*100).T 
    #desome=np.array(desome.astype(int))
    #desome=np.hstack((np.matrix(range(0,L)).T,desome.astype(int)))
    ##np.concatenate((np.matrix(range(0,L)).T,desome),0)
    #np.savetxt(output_dir+'/subjectCoursesFractions.txt',desome)
    
    #plt.close()
    
    
    
    pc=np.divide(M['H_PCs'][0:-1][:],np.tile(np.sum(M['H_PCs'][0:-1][:],0),[M['H_PCs'].shape[0]-1, 1] )) 
    np.savetxt(output_dir+'/Modality_contributions.txt',np.squeeze(np.asarray(pc)))

    #import matplotlib.pyplot as plt2  
    import pandas as pd   
    df2 = pd.DataFrame(pc.T)#, columns=['a', 'b','c'])
    df2.plot.barh(stacked=True);
    plt.savefig(output_dir+'/PCbars.png')
    plt.title('Relative weight of modalities in each component')   
    plt.ylabel('Component index')
    plt.xlabel('Fraction of weight')
    plt.close()
    #plt.show()
    
    #import matplotlib.pyplot as plt3  
    subjDom = np.divide( np.max(np.power(M['H'].T, 2),0)  , np.sum(np.power(M['H'].T, 2),0) )
    plt.plot(range(0,L),subjDom)
    null = np.random.randn(R,1000);
    null = np.divide(np.max(np.power(null,2),0) ,  np.sum(np.power(null,2),0));
    plt.axhline(y=np.percentile(null,95), xmin=0, xmax=L, linewidth=2, color = 'k')
    plt.axhline(y=np.percentile(null,95/L), xmin=0, xmax=L, linewidth=2, color = 'k')
    plt.title('Components dominated by a single subject')   
    plt.xlabel('Component index')
    plt.ylabel('Fraction of energy from max subject')
    plt.savefig(output_dir+'/subjectDominance.png')
    plt.close()

    
    #%% Save free energy plot
    #clear tmp*
    #tmp = [1:length(M.F_history); M.F_history];
    tmp=np.zeros([3,len(M['F_history'])])
    tmp[0:2,:]=np.matrix( [range(1,len(M['F_history'])+1) , M['F_history']])
    #tmp(:,isnan(tmp(2,:))) = [];
    tmp[2,1:] = np.divide(np.diff(tmp[1,:]) , np.diff(tmp[0,:]) )
    np.savetxt(output_dir+'/convergenceRate.txt',tmp)
    #loglog(tmp(1,:), tmp(3,:), '.-')
    #clear tmp*

    # Save H-course correlations plot
    #include?
    
    # Convert X*W*rms(H)*sqrt(lambda) into pseudo-Z-stats
    #Z = flica_Zmaps(M);
    #% Make spatial Z-stat maps
    Z=[np.array(a) for a in range (0,K)]
    for k in range(0,K):
        lambda_R = np.multiply(M['lambda'][k], np.ones([R,1])) #.*
        weight_L = np.sqrt( np.dot(np.power(M['H'],2) , lambda_R));
        if M['W'] == []: #~isempty(M.W)  FIXXXXXXX THISSSSSS I REMOVED PART OF CODE MAKING THIS EMPTY
            weight_L = np.multiply(weight_L , M['W'][k].T);
        #end      
        Z[k] = np.dot(M['X'][k] , np.diag(np.squeeze(np.asarray(weight_L))));

    # Convert Z matrices back into input-sized files and save them
    #flica_save(Z, fileinfo, output_dir);
    #if nargin<4, suffix=''; end
    for k in range(0,len(Z)):
        file_extension = fileinfo['filetype'][k]
        #new_folder_str= ('/Thresholded_Spatial_maps_modality_%s' % k)
        #os.mkdir(output_dir + new_folder_str)
        #if iscell(ft) && isequal(ft{1},'NIFTI')
        if file_extension == '.gz':
        # Save 4D NIFTI file:
            tmp_mask=fileinfo['masks'][k]
            tmp_vectorized_mask = np.reshape(tmp_mask, np.shape(tmp_mask)[0]*np.shape(tmp_mask)[1]*np.shape(tmp_mask)[2] ,order='F')
            Non_zero_voxels=find(tmp_vectorized_mask !=0)
            
            out = np.zeros([np.shape(tmp_mask)[0]* np.shape(tmp_mask)[1]*  np.shape(tmp_mask)[2], np.shape(Z[k])[1]])
            #tmp = np.zeros(np.shape(fileinfo['masks'][k]));
            for i in range(0,np.shape(Z[k])[1]):
                out[Non_zero_voxels,i] = Z[k][:,i];
                #out[:,:,:,i] = tmp;
            
            out=np.reshape(out,[np.shape(tmp_mask)[0], np.shape(tmp_mask)[1],  np.shape(tmp_mask)[2], np.shape(Z[k])[1]],order="F" )
            img2save=nib.Nifti1Image(out,fileinfo['affine'][k],fileinfo['header'][k])
            #str=("/Spatial_maps_Modality_%s.nii.gz" % (k+1))
            str=("/niftiOut_mi%s.nii.gz" % (k+1))
            outname=output_dir+ str
            nib.save(img2save,outname)
            print('Saving results ....')#'Saving "%s"...', outname
            del tmp_mask            
        
        if file_extension == '.mgh':
            tmp_mask=fileinfo['masks'][k]
            non_zero_vertex=find(tmp_mask!=0)
            out=np.tile(np.matrix(tmp_mask).T,Z[k].shape[1])
            out[non_zero_vertex,:]=Z[k]
            N_vertex_per_side=tmp_mask.shape[0]/2
            
            out_left = np.expand_dims(np.expand_dims(out[range(int(N_vertex_per_side)),:],1),1) #expand dims to can save to .mgh
            img2save=nib.freesurfer.mghformat.MGHImage(out_left, affine=fileinfo['affine'][k], header=fileinfo['header'][k], extra=None, file_map=None)
            #str=("/lh_Spatial_maps_Modality_%s.mgh" % (k+1))
            str=("/lh_niftiOut_mi%s.mgh" % (k+1))
            outname=output_dir+ str
            nib.save(img2save,outname)
            
            out_right = np.expand_dims(np.expand_dims(out[int(N_vertex_per_side):int(2*N_vertex_per_side),:],1),1) #expand dims to can save to .mgh
            img2save=nib.freesurfer.mghformat.MGHImage(out_right, affine=fileinfo['affine'][k], header=fileinfo['header'][k], extra=None, file_map=None)
            #str=("/rh_Spatial_maps_Modality_%s.mgh" % (k+1))
            str=("/rh_niftiOut_mi%s.mgh" % (k+1))
            outname=output_dir+ str
            nib.save(img2save,outname) 
          

        if file_extension == '.txt':
            str=("/niftiOut_mi%s.txt" % (k+1))
            outname=output_dir+ str
            np.savetxt(outname,Z[k])       
            
    
          
                   
    
    return M
