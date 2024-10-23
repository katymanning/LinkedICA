from __future__ import print_function
from copy import deepcopy
import numpy as np
from pylab import size, find 
import nibabel as nib
#from numpy import genfromtxt
import os as os
import flica_various as alb_various

def flica_load(opts):
    data_directories=[os.path.join(opts["brain_data_main_folder"],d) for d in os.listdir(opts["brain_data_main_folder"]) if os.path.isdir( os.path.join(opts["brain_data_main_folder"],d))]
    #f = open(opts['output_dir'] + '/order_of_loaded_data.txt','w')
    #for i in range(len(data_directories)):        
    #    f.write(data_directories[i] + ' \n')
    #f.close()    
        
    paths2data=[a for a in range(size(data_directories))]
    for folders in range(size(data_directories)):
        sub_files=os.listdir(data_directories[folders])
        for files in range(size(sub_files)):
            fileName, fileExtension = os.path.splitext(sub_files[files])
            if (fileExtension=='.gz') or (fileExtension=='.mgh') or (fileExtension=='.txt'):
                paths2data[folders]=os.path.join(data_directories[folders],sub_files[files])
                                    
    #paths2data = opts["paths2data"]
    #Change to a namedtuple. InputData = namedtuple('Modalities', 'Mod1 Mod2 Mod3 Mod4')
    # Point = namedtuple('Point', 'x y') ; origin = Point(x=0, y=0) ; origin.x or origin[0] 
    list_of_arrays=[np.array(a) for a in range (0,size(paths2data))]
    Data_Modality = deepcopy(list_of_arrays)
    mask = deepcopy(list_of_arrays)
    filetypes=deepcopy(list_of_arrays)
    names=deepcopy(list_of_arrays)
    affine=deepcopy(list_of_arrays)
    header=deepcopy(list_of_arrays)    
    #Non_zero_voxels= deepcopy(list_of_arrays)
        
    for i in range (0,size(paths2data)):
        fileName, fileExtension = os.path.splitext(paths2data[i])
        print('loading data from modality number',i+1, '=', fileName)

        filetypes[i]=fileExtension
        names[i]=fileName
        if fileExtension == '.gz': # should be .nii.gz, check ....
            
            #load the data
            img = nib.load(paths2data[i])        
            Data_Modality[i] = img.get_data()
            shape = Data_Modality[i].shape
            #gather info needed for saving
            affine[i]=img.affine
            header[i]=img.header

            #make 2 d
            Data_Modality[i]=np.reshape(Data_Modality[i],[shape[0]* shape[1]* shape[2],shape[3]], order='F')
            shape2d=Data_Modality[i].shape
            #Non_zero_voxels= ~np.all(data2d == 0, axis=1)
            Non_zero_voxels= ~np.all(Data_Modality[i] == 0, axis=1)
            
            #Data_Modality[i]=data2d[~np.all(data2d == 0, axis=1)]
            Data_Modality[i]=Data_Modality[i][~np.all(Data_Modality[i] == 0, axis=1)]
            #masking
            tmp_mask=np.zeros(shape2d[0])
            #del data2d
            #tmp_mask[find(Non_zero_voxels[i])]=1
            tmp_mask[find(Non_zero_voxels)]=1
            tmp_mask=np.reshape(tmp_mask, [shape[0], shape[1], shape[2]], order='F')
            mask[i]=tmp_mask
              
                        
        elif fileExtension == '.txt':
            Data_Modality[i]=np.loadtxt(paths2data[i])
           
        elif fileExtension == '.mgh':
            #.mgh oly one l or r must enter hier and both are sides loaded. first letter of name is chagned by l and r.
            #so from command line enter just the name of ONE side starting by lh_ or rh_
            #for the gui i use the 'forced input folder structure' and load both sides here
            fs_path=opts['fs_path'] #= '/Applications/freesurfer'

            #gather names to load , substitute ? by r and l.
            direct=os.path.dirname(paths2data[i])
            infile=os.path.basename(paths2data[i])            
            new_name=list(infile)
            new_name[0]='r'
            right_side= direct+'/'+''.join(new_name)  #os.path.join(direct,new_name) #
            new_name[0]='l'
            left_side=direct+'/'+''.join(new_name)
            
            #load both hemispheres data
            img1=nib.load(left_side)
            
            img2=nib.load(right_side)
            
            affine[i]=img1.affine #img1 and img2 have same affine and header
            header[i]=img1.header
            
            #concatenate in spatial dimension and make 2d
            vol=np.concatenate((img1.get_data(),img2.get_data()),0)
            
            data2d=np.reshape(vol,[vol.shape[0],vol.shape[3]], order='F')
            
            del vol
            
            #mask out median wall
            NvoxPerHemi = data2d.shape[0]/2
            fs_path=opts['fs_path'] #= '/Applications/freesurfer'
           
            if NvoxPerHemi == 2562:
                labelSrcDir = fs_path+'/subjects/fsaverage4/label/'
                
            if NvoxPerHemi == 10242:
                labelSrcDir = fs_path+'/subjects/fsaverage5/label/'
                
            if NvoxPerHemi == 40962:
                labelSrcDir = fs_path+'/subjects/fsaverage6/label/' 
                   
            if NvoxPerHemi == 163842:
                labelSrcDir = fs_path+'/subjects/fsaverage/label/' 
            
            mask[i]=np.ones(np.shape(data2d)[0])
            needed_labels=['lh.cortex.label','lh.Medial_wall.label','rh.cortex.label','rh.Medial_wall.label']            
            tmp=[np.array(a) for a in range(4)]
            for fi in range(0,4): #mask out ??? 
                tmp[fi]= nib.freesurfer.io.read_label(labelSrcDir+needed_labels[fi],read_scalars=True)
                tmp[fi] = tmp[fi][0] + 1 + ((needed_labels[fi][0]=='r')*NvoxPerHemi)
                
            mask[i][tmp[1].astype('int')] = 0
            mask[i][tmp[3].astype('int')] = 0
            
            
            Data_Modality[i]=data2d[find(mask[i][:] != 0),:]
            del data2d
            
            
    # Apply transformations to rawdata to get Y=input data for flica inference
    #transforms = repmat({'nop','auto2','rms','double'},len(paths2data),1);
    scaling_data_transform = [np.ndarray(a) for a in range (len(paths2data))]
   
    for k in range (len(paths2data)):
        # Implemented only original standard tansforms 
        #  Possibilities to add:
        #  Check for "missing data" volumes (missing subjects in some data modality): 
        #  implement log transform in some data modality? check .m for soft log : this can  be done also before input to flica
        
        # de-mean       
        Data_Modality[k] = Data_Modality[k]- (np.matrix(np.mean(Data_Modality[k],1)).T)    #(Y[k].T - tile(dumm,(Y[k].shape[1],1))).T
        
        # third data transform, De-scaling? need to save for saving results in flica save everything...
        scaling_data_transform[k] = alb_various.rms(Data_Modality[k],[],[])       
        Data_Modality[k] = np.divide(Data_Modality[k],scaling_data_transform[k])
        
    #Gather info for latter mapping of each modality spatial maps to their respective 3-d spaces
    fileinfo = {"data_directories":data_directories,"masks":mask,"scaling_data_transform":scaling_data_transform,
                "filetype":filetypes,"names":names,"affine":affine,"header":header}
    
    return Data_Modality,fileinfo;

