#!/usr/bin/env python

"""call as:   
./flica_posthoc_correlations path_2_beh_data flica_output_directory    
"""
##The text here will be available within the code as a __doc__ variable.
#See the argparse documentation and examples at
#https://docs.python.org/3.6/library/argparse.html
#
#Don't forget the shebang statement on the first line of the file if you plan to
#call the file directly as a script.
import argparse


def cli_parser():

    # Create a parser. It'll print the description on the screen.
    parser = argparse.ArgumentParser(description=__doc__)

    # Add a positional argument
    parser.add_argument('path_2_beh_data', help='path to .txt file containing behavioural data. the file size must be mxn with n=number of subjetcs')
    parser.add_argument('flica_output_directory', help='path to folder output of a flica factorization')

    #parser.add_argument_group('Infiles','group1 description')

    # Add an optional argument ("-m" instead of simply "m")
    #parser.add_argument('-flica_output_directory', help='path to folder output of a flica factorization')
    
    return parser


import os 
#import copy
import numpy as np
#import flica_various as alb_various
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def main(): 
    
    parser = cli_parser()

    # Get the inputs from the command line:
    args = vars(parser.parse_args())
    #import pdb;pdb.set_trace()
    flica_posthoc_correlations(**args)

    # Print the arguments, for example  print(args.Infolder)


def flica_posthoc_correlations(path_2_beh_data=None,flica_output_directory=None):


    output_dir=flica_output_directory
    path_2_beh_data=path_2_beh_data
    
    H = np.loadtxt(output_dir + '/subjectCoursesOut.txt')
    NIcas=H.shape[0]
    Nsubs=H.shape[1]
    fileName, fileExtension = os.path.splitext(path_2_beh_data)
    if fileExtension == '.txt':
        beh_data=np.loadtxt(path_2_beh_data)
    else:# if fileExtension == '.csv':
        print('your input for behavioural data is not suported, please use .txt') 
        
    if beh_data.shape[0]==Nsubs:
        1
    else:
        beh_data=beh_data.T
        
    Nbeh_measures=beh_data.shape[1]
    
    f = open(output_dir + '/significantComponents.txt','w') 
    
    for i in range(Nbeh_measures):
        bi=beh_data[:,i]
        f.write('measure_'+ str(i+1)+':  ')
        isSig = np.empty([NIcas,2])
        isSig[:]=np.nan
        
        corrString = np.chararray([NIcas,2], itemsize=20) #cell(NIcas,2);
        #corrString[:]=''
        
        for jj in range(NIcas):
            
            Hj=H[jj,:]
            
            # Plot outputs
            outplot = output_dir + '/correlation_beh_'+ str(i+1)+ '_ica_'+ str(jj+1)+ '_.png'
            plt.scatter(bi, Hj,  color='black')
            xlim=np.asarray(plt.xlim())
            #plt.plot((xlim[0], xlim[1]), (0, 0), 'k--',linewidth=1)
            plt.title('measure_'+ str(i+1) + ' vs component_'+str(jj+1))   
            plt.xlabel('measure_'+ str(i+1))
            plt.ylabel('component_'+str(jj+1))
           
            #plt.savefig(outplot)
            #plt.close()
            #plt.show()
            
            #PERFORM linear GLM ANALYSES
            
            from scipy import stats
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(bi,Hj)
            #slope and intercept are the betas from matlab code, p-vals are matched
            
            corrString[jj][0] = 'p='+ '{0:.6f}'.format(p_value)
            if p_value<1:#0.05:
                
                if p_value<0.05/(Nbeh_measures*NIcas):
                    lw=2
                    col='r'
                    isSig[jj,0] = p_value
                    corrString[jj,0] = '<b>' + corrString[jj,0] + '**</b>'
                elif p_value<0.05/(Nbeh_measures):
                    lw=1
                    col='r'
                    
                else:
                    lw=1
                    col='k'
                        
                plt.plot(xlim, (xlim*slope)+intercept,linewidth=lw, color = col)#(xlim, xlim*beta(2)+beta(1))
                    
            
            #ADD linear + QUADRATIC GLM ANALYSES AND PLOT.....

            #slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(np.column_stack((bi,np.power(bi,2))).T,np.matrix(Hj).T)
            #corrString[jj][1] = 'p='+ '{0:.6f}'.format(p_value2)
            corrString[jj][1] = 1 #corrString[jj][0] 
            #if p_value2<0.05:           
                #if p_value2<0.05/(Nbeh_measures*NIcas):
                #    lw=2
                #    col='r'
                #    isSig[jj,1] = p_value2
                #    corrString[jj,1] = '<b>' + corrString[jj,1] + '**</b>'
                #elif p_value2<0.05/(Nbeh_measures):
                #    lw=1
                #    col='r'
                #    
                #else:
                #    lw=1
                #    col='k'
                
            #plot(ans, ans*beta(2)+beta(1)+ans.^2*beta(3), style{:})

            
            #import statsmodels.api as sm
            #model = sm.GLM(bi, Hj, family=sm.families.Gaussian())
            #results = model.fit()
            #print(resultsresults.summary())
            ##est = sm.OLS(bi, Hj)
            ##res=est.fit()
            ##print(est.fit().f_pvalue)

            
            #from sklearn import linear_model
            #reg = linear_model.LinearRegression()
            #reg.fit(np.matrix(bi),np.matrix(Hj))
            ##LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
            ## The coefficients
            #print('Coefficients: \n', reg.coef_)
            
            
            plt.savefig(outplot)
            plt.close()
            
        isSig
        isSig[np.isnan(isSig)]=np.inf
        isSig = np.min(isSig,1)
        idx=np.where(np.isfinite(isSig))[0]
        #np.sort(isSig)
        for k in range(idx.shape[0]):
            isSig[idx[k]]
            f.write('#'+ str(idx[k]) + '(' + corrString[jj][0] + ' , ' + corrString[jj][0] + '), ' )
        
        f.write('\n')
                    


            
            

            

    

      
    f.close()
    print('Ended computing posthoc correlations')

    return 1

if __name__ == '__main__':
    main()