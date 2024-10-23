# LinkedICA
Linked independent component analysis (beta version) developed with Dr. Alberto Llera and the FSL group.


*COMMAND LINE USE

You can call flica/flica.py and flica/flica_posthoc_correlations.py from the command line.
Type ./flica/flica.py -h or ./flica/flica_posthoc_correlations.py -h to see how to call each one. 

*USE FORM WITHIN PYTHON

Check the example script (flica/example.py)


*GRAPHICAL USER INTERFACE (GUI)

In flica folder you can find FLICA_GUI.py

Run it from terminal by typing 

fslpython flica/FLICA_GUI.py

——————————————————————————————————

*INPUT SPECIFICATIONS

-Data modalities input to Linked factorization

The code accepts .nii.gz, .mgh (freesurfer) or .txt as input for the factorisation.
In all this cases, the last dimension of the file encodes the subject index. 
For example, the 4-th dimension of the nii.gz files must be the subject number or the second dimension of the .txt must be the number of subjects. 

Further, for simplicity of use, both from command line or from GUI, I decided to provide as input to the factorisation an unique folder. This folder MUST be organised in such a way that the code understands it and knows where to find things. This folder MUST contain a subfolder for each input ‘modality’ and that folder MUST contain an unique file for the data of that modality for .nii.gz and .txt files (check the folder named data), and 2 files for the .mgh type; in the case of .mgh, the file names MUST start by lh_ or rh_ to encode left and right hemispheres respectively.

-Number of components

Usually < Number of subjects /4. Nevertheless, exceptions occur where high dimensional factorizations are very interesting.


-Maximum number of iterations

This is a bit tricky. I set it to around 3000 thousands when using real data.There is a threshold for tolerance in convergence (relative free energy change between two consecutive iterations) but, for this model, such tolerance is very much dependent in data and consequently is difficult to set to a fixed value. Convergence can be observed a posteriory looking at F_history in the M.npy

-Paths to free surfer and fsl installations

Are mostly not needed but is added since I will need it later on. 

-USING TYHE GUI

From the GUI you MUST create the folder to save the results before hand.
From the GUI you MUST define all input values shown there, including the paths to fslview and free surfer. You might need to use cmd+alt+g to get to hidden folders. If you do not have such installations just set the paths to random paths and let me know if you get issues, I do not think so (for this version). 


-POST-HOC correlations input data

The data input for post-hoc correlation analyses must be a .TXT file and in this case the first dimension encodes the subject index. Although this is the opposite as the input for 
Factorisation I consider it easier since .csv files or other formats in which we usually find the behavioural data are ordered in such way. Nevertheless if inputed in the wrong dimension the code will take care of the issue for you (as long as the number of subjects is different from the number of behavioural measures). 

