from IPython import display

import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
#import tensorflow_probability as tfp
import time

'''
    networkname='EPINET'  
    directory_ckp="epinet_checkpoints/%s_ckp"% (networkname)     
    if not os.path.exists(directory_ckp):
        os.makedirs(directory_ckp)
        
    if not os.path.exists('epinet_output/'):
        os.makedirs('epinet_output/')   
    directory_t='epinet_output/%s' % (networkname)    
    if not os.path.exists(directory_t):
        os.makedirs(directory_t)     
        
    txt_name='epinet_checkpoints/lf_%s.txt' % (networkname) 
	
	dir_LFimages=[
#            'additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
#            'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',    
#            'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table', 
            'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl' ]
'''


for directories in os.listdir(os.getcwd()): 
    dir = os.path.join('/home/user/workspace', directories)
    os.chdir(dir)
    current = os.path.dirname(dir)
    new = str(current).split("-")[0]
    print(new)
    #traindata_all,traindata_label=load_LFdata(dir_LFimages)