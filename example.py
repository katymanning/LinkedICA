
import os 
import sys

#modify next line to provide the  path to your fsl_flica directory
toolbox_path = "/Users/alblle/allera_version_controlled_code/fsl_flica"

#Modify next line to identify where you want to save output:
output_dir='/Users/alblle/Desktop/Flica_Output'

#If you want to use your own data then modify next line to direct to your data folder
brain_data_main_folder= os.path.join(os.path.abspath(toolbox_path),"flica/data/brain_data_4subs2")
path_2_beh_data_file=os.path.join(os.path.abspath(toolbox_path),"flica/data/demogr_4sbj.txt")




sys.path.append(os.path.join(os.path.abspath(toolbox_path),"flica")) 
from flica import flica 

num_components=2  #number of ICAS to extract (recommend but not necesarily < Number of Subjects/4 (ask me if you want more explanation) )
maxits=30  # set to 3000 or somethng high, you can check convergence looking at Convergence rate.txt file saved in results.
lambda_dims='o' # 'R' or 'o'  # 'o' encodes modality-wise noise, 'R' encodes modality and subject wise (standard 'o')
fsl_path=os.environ['FSLDIR']
fs_path='/Applications/freesurfer'
tol=0.0001#0.000001 #tolerance for convergence. 

#run flica
#dum=flica(brain_data_main_folder, output_dir , num_components, maxits, tol, lambda_dims , fs_path, fsl_path,"PCAnew")
dum=flica(brain_data_main_folder, output_dir , num_components, maxits, tol, lambda_dims , fs_path, fsl_path,"PCA")



#perform linear correlations to other subject measures (behavioural/demographics/whatever....)
from flica_posthoc_correlations import flica_posthoc_correlations

end=flica_posthoc_correlations(path_2_beh_data_file,output_dir)
1
1

