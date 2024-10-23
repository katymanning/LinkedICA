#!/usr/bin/env python

"""call as:
     
./flica.py -nICAs 2 -maxits 10 -lambda_dims 'R' -output_dir ~/Desktop/Flica_Output -fs_path /Applications/freesurfer /Users/alblle/Dropbox/POSTDOC/FLICA2PYTHON/brain_data_4subs

"""
from __future__ import print_function
import argparse
import warnings
warnings.filterwarnings("ignore")

def cli_parser():
    # Create a parser. It'll print the description on the screen.
    parser = argparse.ArgumentParser(description=__doc__)
    # Add a positional argument
    parser.add_argument('Infolder', help='Input folder containig a subfolder per input modality. Each subfolder must contain ONE .nii.gz file, or TWO .mgh files. In the case of MGH files, their names MUST start by lh_ and rh_ relating to left and right hemispheres respectively.')
    # Add optional arguments
    parser.add_argument('-nICAs', help='Number of Icas to compute',default=2,type=int)
    parser.add_argument('-lambda_dims', help='Spatial noise estimation dimensions (default, per modality)',default='o')
    parser.add_argument('-maxits', help='Maximum number of iterations',default=1000,type=int)
    parser.add_argument('-output_dir', help='Output directory',default='~/Desktop/Flica_Output')
    parser.add_argument('-fs_path', help='Path to FreeSurfer installation',default= '/Applications/freesurfer')
    parser.add_argument('-fsl_path', help='Path to FSL installation',default= '/usr/local/fsl')
    parser.add_argument('-tol', help='Tolerance for convergence',default=0.000001)
    parser.add_argument('-initH', help='Initial Mixing matrix (subject courses, default PCA)', default='PCA')
    parser.add_argument('-dof_per_voxel', help='Degrees of freedom per voxel, default auto_eigenspectrum)', default='auto_eigenspectrum')
    
    return parser

from flica_load import flica_load
from flica_init_params import flica_init_params
from flica_iterate import flica_iterate
from flica_reorder import flica_reorder
from flica_save_everything import flica_save_everything
#from flica_make_spatial_pngs import flica_make_spatial_pngs 
import os

def main():     
    parser = cli_parser()
    # Get the inputs from the command line:
    args = vars(parser.parse_args())
    #import pdb;pdb.set_trace()
    flica_partA(**args)
    
def flica_partA(Infolder=None, output_dir = os.path.join(os.getcwd(), "Flica_Output"),
        nICAs = 2, maxits = 5 , tol=0.000001, lambda_dims = 'o', fs_path = '/Applications/freesurfer',
        fsl_path = '/usr/local/fsl', initH = "PCA", dof_per_voxel = 'auto_eigenspectrum'):    
    
    opts = {"brain_data_main_folder":Infolder, 'output_dir':output_dir,
    "num_components":nICAs,"maxits":maxits,'lambda_dims':lambda_dims, 'fsl_path':fsl_path,
    "initH":initH,"dof_per_voxel":dof_per_voxel,"tol":tol} 
    
    # Run FLICA
    Y,images_info=flica_load(opts)
    
    Priors, Posteriors, Constants=flica_init_params(Y,opts)

    dict_Y={"Y": Y};
    dict_images_info={"images_info":images_info}
    dict_Priors={"Priors":Priors}
    dict_Posteriors={"Posteriors":Posteriors}
    dict_Constants={"Constants",Constants}
    
    handle=open('dict_Y.pickle','wb')
    pickle.dump(dict_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle1=open('dict_images_info.pickle','wb')
    pickle.dump(dict_images, handle1, protocol=pickle.HIGHEST_PROTOCOL)
    handle2=open('dict_Priors.pickle','wb')
    pickle.dump(dict_Priors, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    handle3=open('dict_Posteriors.pickle','wb')
    pickle.dump(dict_Posteriors, handle3, protocol=pickle.HIGHEST_PROTOCOL)
    handle4=open('dict_Constants.pickle','wb')
    pickle.dump(dict_Constants, handle4, protocol=pickle.HIGHEST_PROTOCOL)

    return 1

if __name__ == '__main__':
    main()

