#! /usr/bin/env python
#import imp


from __future__ import print_function
#from future import standard_library
#standard_library.install_aliases()

import warnings
warnings.filterwarnings("ignore")
import sys
import os
#path1 = "/Users/alblle/Dropbox/POSTDOC/FLICA2PYTHON/"
#scriptpath1 = path1+"progress12_mac/FLICA_dependences"
flica_toolbox_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(flica_toolbox_path))


#import fun_gui_call_flica
#from flica import *
#from flica_reorder import *

#http://poquitopicante.blogspot.nl/2013/06/blog-post.html
#sys.path.append("/home/allera/anaconda/lib/python2.7/lib-tk")

#! /usr/bin/env python
 
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import *

#from tkFileDialog import *

#print scriptpath1
#from flica import *
#~~~~ FUNCTIONS~~~~

def data_folder():
    global folder_path
    folder_path=askdirectory()
    entry.delete(0, END)
    entry.insert(0, folder_path)
    return folder_path

def def_out_dir():
    global out_dir
    out_dir = askdirectory()
    entry5_2.delete(0, END)
    entry5_2.insert(0, out_dir)
    return out_dir

#def def_fslpath():
#    global fslpath
#    #filename = askopenfilename()
#    fslpath = askdirectory()
#    entry5_3.delete(0, END)
#    entry5_3.insert(0, fslpath)
#    return fslpath

def def_fspath():
    global fspath
    fspath = askdirectory()
    entry5_4.delete(0, END)
    entry5_4.insert(0, fspath)
    return fspath

def def_fslpath():
    global fslpath
    fslpath = askdirectory()
    entry5_4_2.delete(0, END)
    entry5_4_2.insert(0, fslpath)
    return fslpath

def beh_data_file():
    global beh_data
    beh_data = askopenfilename(filetypes=[("Text files","*.txt")])                     

    entry5.delete(0, END)
    entry5.insert(0, beh_data)
    return beh_data

def callback3():
    global numICAs
    numICAs = entry3.get()
    entry3.delete(0,END)
    entry3.insert(0,numICAs)
    return numICAs


def callback4():
    global maxIts
    maxIts = entry4.get()
    maxIts=maxIts
    entry4.delete(0,END)
    entry4.insert(0,maxIts)
    return maxIts

def gui_calls_VBflica():
    #import pdb;pdb.set_trace()
    os.system(flica_toolbox_path+"/flica.py -nICAs " + numICAs + " -maxits " + maxIts + " -lambda_dims " + noise + " -output_dir " + out_dir+ " -fs_path " + fspath+ " -fsl_path " + fslpath  +" "+ folder_path)
    return 1
    
def noise_est_opts():
    global noise
    noise='o'
    return noise

def noise_est_opts2():
    global noise
    noise='R'
    return noise

def gui_calls_posthoc_correlations():
    os.system(flica_toolbox_path+"/flica_posthoc_correlations.py " + beh_data + " " + out_dir) 
    return 1

def gui_calls_web_report():
    #print(flica_toolbox_path+"/call_script_flica_report_ALB.py " +  flica_toolbox_path+ " "  + out_dir + " " + fspath)
    os.system(flica_toolbox_path+"/call_script_flica_report_ALB.py " +  flica_toolbox_path+ " "  + out_dir + " " + fspath)
    return 1

root = Tk() # create a top-level window
 
master = Frame(root, name='master') # create Frame in "root"
master.pack(fill=BOTH) # fill both sides of the parent
 
root.title('FSL - Linked ICA') # title for top-level window
# quit if the window is deleted
root.protocol("WM_DELETE_WINDOW", master.quit)
 
nb = Notebook(master, name='nb') # create Notebook in "master"
nb.pack(fill=BOTH, padx=2, pady=3) # fill "master" but pad sides
 
#-->INPUT DATA TAB
f1 = Frame(nb, width=600, height=250)
f1.pack(fill=X)
nb.add(f1, text="Input options & run ICA") # add tab to Notebook

#folder_path = StringVar
Label(f1,text="Select input (brain) data directory").grid(row=0, column=0, sticky='e')
entry = Entry(f1, width=50)#, textvariable=folder_path)
entry.grid(row=0,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=data_folder).grid(row=0, column=27, sticky='ew', padx=8, pady=4)

Label(f1,text="Select output directory").grid(row=1, column=0, sticky='e')
entry5_2 = Entry(f1, width=50)#, textvariable=folder_path)
entry5_2.grid(row=1,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=def_out_dir).grid(row=1, column=27, sticky='ew', padx=8, pady=4)


Label(f1,text="Number of ICAs").grid(row=2, column=0, sticky='e')
entry3 = Entry(f1, width=10)#, textvariable=dum_str)
entry3.grid(row=2,column=1,padx=2,pady=2,sticky='we',columnspan=10)
Button(f1, text="save", command=callback3).grid(row=2, column=27, sticky='ew', padx=8, pady=4)

Label(f1,text="max number of VB iterations").grid(row=3, column=0, sticky='e')
entry4 = Entry(f1, width=10)#, textvariable=dum_str)
entry4.grid(row=3,column=1,padx=2,pady=2,sticky='we',columnspan=10)
Button(f1, text="save", command=callback4).grid(row=3, column=27, sticky='ew', padx=8, pady=4)


Label(f1,text="Noise estimation").grid(row=4, column=0, sticky='e')
Button(f1, text="Modality wise", command=noise_est_opts).grid(row=4, column=10, sticky='ew', padx=8, pady=4)
Button(f1, text="Modality & subject wise", command=noise_est_opts2).grid(row=4, column=20, sticky='ew', padx=8, pady=4)


Label(f1,text="FS path").grid(row=5, column=0, sticky='e')
entry5_4 = Entry(f1, width=50)#, textvariable=folder_path)
entry5_4.grid(row=5,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=def_fspath).grid(row=5, column=27, sticky='ew', padx=8, pady=4)

Label(f1,text="FSL path").grid(row=6, column=0, sticky='e')
entry5_4_2 = Entry(f1, width=50)#, textvariable=folder_path)
entry5_4_2.grid(row=6,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=def_fslpath).grid(row=6, column=27, sticky='ew', padx=8, pady=4)



Button(f1, text="RUN ICA", command=gui_calls_VBflica).grid(row=8, column=14, sticky='ew', padx=8, pady=4)
Button(f1, text="KILL GUI", command=master.quit).grid(row=10, column=14, sticky='ew', padx=8, pady=4)


#<--INPUT DATA TAB




#Label(f1,text="FSL path").grid(row=5, column=0, sticky='e')
#entry5_3 = Entry(f1, width=50)#, textvariable=folder_path)
#entry5_3.grid(row=5,column=1,padx=2,pady=2,sticky='we',columnspan=25)
#Button(f1, text="Browse", command=def_fslpath).grid(row=5, column=27, sticky='ew', padx=8, pady=4)





# Posthoc correlations tab
f2 = Frame(master, name='master-post_processing')
nb.add(f2, text="posthoc correlations")


Label(f2,text="Select behavioural data file").grid(row=0, column=0, sticky='e')
entry5 = Entry(f2, width=50)#, textvariable=folder_path)
entry5.grid(row=0,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f2, text="Browse", command=beh_data_file).grid(row=0, column=27, sticky='ew', padx=8, pady=4)

#Button(f4, text="post-processing", command=master.quit).grid(row=0, column=0, sticky='ew', padx=8, pady=4)
Button(f2, text="Run posthoc correlations", command=gui_calls_posthoc_correlations).grid(row=1, column=14, sticky='ew', padx=8, pady=4)


# Web report tab
f3 = Frame(master, name='master-webreport')
nb.add(f3, text="Web report")
Button(f3, text="Create orientative web report", command=gui_calls_web_report).grid(row=0, column=0, sticky='ew', padx=8, pady=4)


 
# start the app
if __name__ == "__main__":
    master.mainloop() # call master's Frame.mainloop() method.
    #root.destroy() # if mainloop quits, destroy window

