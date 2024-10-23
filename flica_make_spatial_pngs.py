#!/usr/bin/env python

"""call as:
     
./flica_make_spatial_pngs.py -fsl_dir path2_fsl_directory -fs_dir path2_free_surfer_directory -low_th 2 -high_th 5 flica_output_directory 

"""
from __future__ import print_function
import argparse

def cli_parser():
    # Create a parser. It'll print the description on the screen.
    parser = argparse.ArgumentParser(description=__doc__)
    # Add a positional argument
    parser.add_argument('flica_output_directory', help='Folder where output of Linked ICA factorization where saved')
    # Add optional arguments
    parser.add_argument('-low_th', help='Lower absolute value to threshold maps',default= '2')
    parser.add_argument('-high_th', help='Higher absolute value to threshold maps',default= '5')
    
    parser.add_argument('-fsl_dir', help='Path to FSL installation',default= '/usr/local/fsl')
    parser.add_argument('-fs_dir', help='Path to FreeSurfer installation',default= '/Applications/freesurfer')
    
    return parser


import subprocess # this seems to be nicer than os for executing command line things
import os
import nibabel as nib
from surfer import Brain
#print(__doc__)

import time
#import os
import numpy as np
import matplotlib.pyplot as plt

def main():     
    parser = cli_parser()
    # Get the inputs from the command line:
    args = vars(parser.parse_args())
    #import pdb;pdb.set_trace()
    flica_make_spatial_pngs(**args)


def flica_make_spatial_pngs(flica_output_dir=None,fsl_dir='/usr/local/fsl',fs_dir='/Applications/freesurfer',low_th=2,high_th=5):

    all_files=os.listdir(flica_output_dir)     
    for f in range(len(all_files)):
            fileName, fileExtension = os.path.splitext(all_files[f])
            
            if (fileExtension=='.gz' or fileExtension=='.mgh' or fileExtension=='.txt') and (fileName != 'Modality_contributions' and fileName != 'convergenceRate' and fileName!='noiseStdevOut' and fileName!='order_of_loaded_data' and fileName!='subjectCoursesFractions' and fileName!='subjectCoursesOut'):
                
                file_name_without_extension=os.path.splitext(all_files[f])[0]
                
                if fileExtension=='.gz':
                    tmp_save_dir=os.path.join(flica_output_dir,file_name_without_extension[0:-4])#[0:-4] is to remove the anloying .nii 
                else:
                    tmp_save_dir=os.path.join(flica_output_dir,file_name_without_extension)

                if not os.path.isdir(tmp_save_dir):
                    os.mkdir(tmp_save_dir)
                    
                if fileExtension!='.txt':
                    img=nib.load(os.path.join(flica_output_dir,all_files[f]))
                    affine1=img.affine 
                    header1=img.header
                    vol=img.get_data()
                elif fileExtension =='.txt':  
                    img=np.loadtxt(os.path.join(flica_output_dir,all_files[f]))
                                    
                if fileExtension=='.gz':
                    for ica_num in range(vol.shape[3]):
                        1
                        #tmp_img=vol[:,:,:,ica_num]
                        ##RENORMALIZE
                        #tmp_mean=np.mean(tmp_img[tmp_img!=0])
                        #tmp_std=(tmp_img[tmp_img!=0]).std()
                        #tmp_img[tmp_img!=0]=np.divide(tmp_img[tmp_img!=0]-tmp_mean,tmp_std)
                        ##plt.hist(tmp_img[tmp_img!=0],200);
                        ##plt.xlim([-3,3]);
                        ##plt.show()
                        
                        ##Threshold image
                        #tmp_img2=tmp_img;tmp_img2[tmp_img2<low_th];
                        #tmp_img3=tmp_img;tmp_img3[tmp_img3>-low_th]=0;
                        #tmp_img=tmp_img2+tmp_img3;
                        
                        #img2save=nib.Nifti1Image(tmp_img,affine=affine1,header=header1)
                        #str1='_ICA_%s.nii.gz' % (ica_num)
                        #str2= '/' + file_name_without_extension[0:-4] +  str1 #[0:-4] is to remove the anloying .nii      
                        #outname = tmp_save_dir + str2
                        #nib.save(img2save,outname)
                        
                        ##reference_T1_MNI='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
                        ##put in mni 1mm for high-res visualization
                        ##subprocess.call(['flirt','-applyisoxfm','1' ,'-in', outname,'-ref', reference_T1_MNI,'-out', outname])                        
                        
                        ##overlay a brain image reference and threshold OVERLAY GOES WRONG....
                        ##overlay 0 0 -c /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz 3000 8000 /Users/alblle/Dropbox/POSTDOC/FLICA2PYTHON/progress12_mac/Flica_Output/niftiOut_mi1/niftiOut_mi1_ICA_0.nii.gz 2 5 /Users/alblle/Dropbox/POSTDOC/FLICA2PYTHON/progress12_mac/Flica_Output/niftiOut_mi1/niftiOut_mi1_ICA_0.nii.gz -2 -5 tmp
                        ##reference_brain_image_visual=reference_T1_MNI #reference_brain_image_visual SHOULD BE AN OPTION TO CAN PLOT DTI IN SKELETON 
                        ##cmd=['fslview_deprecated',reference_brain_image_visual]#,'-b', str(2.3),',',str(6)]
                        ##cmd=['overlay','0','0','-c',reference_T1_MNI,str(3000),str(8000),outname,str(low_th),str(high_th),'tmp.nii.gz']#,outname,str(-1*low_th),str(-1*high_th),'tmp']
                        ##subprocess.call(cmd)
                        
                        ##cmd=['fslroi',outname,outname,'0','-1','0','-1','30','120']        
                        ##subprocess.call(cmd) #10 70 12 85 20 50
                        ##subprocess.call(['slicer', outname, '-n', '-u', '-i', '-0.5', '0.5','-S', '15', '100000', tmp_save_dir + '/tmp.ppm'] )
                        #subprocess.call(['slicer', outname, '-i', '-0.5', '0.5','-S', '15', '100000', tmp_save_dir + '/tmp.ppm'] )

                        ##subprocess.call(['slicer', outname, '-l','fsldir'+'/etc/luts/renderhot.lut','-n', '-u', '-i', '-0.5', '0.5','-S', '15', '100000', 'tmp.ppm'] )
                        
                        #subprocess.call(['convert', 'tmp.ppm', "%s.png" % outname])                        
                        
                        #subprocess.call(['rm', 'tmp.ppm'])
                        
                        #subprocess.call(['rm', outname])   
                    
                    
                elif fileExtension=='.mgh':                        
                    name=list(file_name_without_extension)
                    if name[0]=='r':
                        hemi = 'rh'
                    elif name[0]=='l':
                        hemi = 'lh'    

                    data2d=np.reshape(vol,[vol.shape[0],vol.shape[3]], order='F')
                    #brain = Brain(subject_id, hemi, surf, subjects_dir = fs_dir + '/subjects')
                    #brain.add_overlay(input_image, min=2, max=5, sign="abs")
                    #brain.save_image_sequence(0, 'whatever', use_abs_idx=True, row=-1, col=-1, montage='single', border_size=15, colorbar='auto', interpolation='quadratic')                                                
                    for ica_num in range(data2d.shape[1]):
                        tmp_img=np.expand_dims(np.expand_dims(np.squeeze(data2d[:,ica_num]),1),1)
                        img2save=nib.freesurfer.mghformat.MGHImage(tmp_img, affine=affine1)#, header=header1, extra=None, file_map=None)
                        str1="_ICA_%s.mgh" % (ica_num)
                        str= '/' + file_name_without_extension +  str1      
                        outname = tmp_save_dir + str
                        nib.save(img2save,outname)     
                        
                        subject_id = 'fsaverage'
                        ##hemi = 'lh'
                        surf = 'inflated'
                        if ica_num ==0: #do one dummy image because first does not get saved...
                            brain = Brain(subject_id, hemi, surf, subjects_dir = fs_dir + '/subjects')
                            time.sleep(1)
                            overlay_file =outname
                            brain.add_overlay(overlay_file, min=2, max=5, sign="abs")
                            time.sleep(.1)    
                            
                        brain = Brain(subject_id, hemi, surf, subjects_dir = fs_dir + '/subjects')
                        time.sleep(.1)
                        overlay_file =outname
                        brain.add_overlay(overlay_file, min=low_th, max=high_th, sign="abs")
                        brain.show_view('medial')#('lat')
                        brain.save_image("%s.png" % outname)
                        subprocess.call(['rm', outname]) #os.remove(outname)
                        
                elif fileExtension=='.txt': 
                    plt.imshow(img, aspect='auto')
                    plt.title(file_name_without_extension)   
                    plt.ylabel('Dimensions')
                    plt.xlabel('ICAs')
                    #plt.show()
                    plt.savefig(tmp_save_dir+'/'+file_name_without_extension + '.png')
                    plt.close()                       

if __name__ == '__main__':
    main()
