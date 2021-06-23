# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 15:54:01 2018

@author: shinyonsei2
"""

import numpy as np
#import imageio
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    
    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line
    
    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data
    


def load_LFdata2(dir_LFimages,r): 

    traindata=torch.FloatTensor(3, r, r, 9, 9).zero_()   
    EPI_h=torch.FloatTensor(3, r, r, 9).zero_() 
    EPI_v=torch.FloatTensor(3, r, r, 9).zero_() 
    LF_dir = random.choice(dir_LFimages)
    #print(dir_LFimages)
    h_index = 0
    v_index = 0
    for i in range(81):
        try:
            filename = LF_dir+'/input_Cam0%.2d.png' % i
            tmp = Image.open(filename)

        except:
            print(LF_dir+'/input_Cam0%.2d.png..does not exist' % i )
        u = i//9
        v = i-9*(i//9)
        #print('u:%.2d' % u )
        #print('v:%.2d' % v )
        tmp = transforms.Scale(r)(tmp)
        tmp = transforms.ToTensor()(tmp)
        traindata[:,:,:,u,v]=tmp 
        
        if v == 4: 
            EPI_h[:,:,:,v_index]=tmp
            v_index = v_index + 1

        elif u == 4:
            EPI_v[:,:,:,h_index]=tmp
            h_index = h_index + 1

        del tmp

    try:       
        #label = 'gt0' + LF_dir[3:6] + '.png'   
        label = '../gt/'+LF_dir+'/input_Cam000.png'
        #print('label:' + label)
        tmp = Image.open(label)
    except:
        print('can not load label'+label)  
    tmp = transforms.Scale(r)(tmp).convert('RGB')
    traindata_gt = transforms.ToTensor()(tmp)
    #traindata_gt= tmp.convert('RGB')
    del tmp

    #image_id=image_id+1
    return traindata, traindata_gt, EPI_h, EPI_v

def load_LFdata(dir_LFimages,r): 

    #traindata=torch.FloatTensor(3, r, r, 9, 9).zero_()   
    EPI_h=torch.FloatTensor(400,3, r, r, 9).zero_() 
    EPI_v=torch.FloatTensor(400,3, r, r, 9).zero_() 
    traindata_gt=torch.FloatTensor(400,3, r, r).zero_() 
    #LF_dir = random.choice(dir_LFimages)
    image_id=0
    for dir_LFimage in dir_LFimages:
    #print(dir_LFimage)
        h_index = 0
        v_index = 0
        for i in range(81):
            try:
                filename = dir_LFimage+'/input_Cam0%.2d.png' % i
                tmp = Image.open(filename)

            except:
                print(dir_LFimage+'/input_Cam0%.2d.png..does not exist' % i )
            u = i//9
            v = i-9*(i//9)
            #print('u:%.2d' % u )
            #print('v:%.2d' % v )
            tmp = transforms.Scale(r)(tmp)
            tmp = transforms.ToTensor()(tmp)
            #traindata[:,:,:,u,v]=tmp 
        
            if v == 4: 
                EPI_h[image_id,:,:,:,v_index]=tmp
                v_index = v_index + 1

            elif u == 4:
                EPI_v[image_id,:,:,:,h_index]=tmp
                h_index = h_index + 1

            del tmp

        try:       
            #label = 'gt0' + dir_LFimage[3:6] + '.png'   
            label = '../gt/'+dir_LFimage+'/input_Cam000.png'
            print('label:' + label)
            tmp = Image.open(label)
        except:
            print('can not load label'+label)  
        tmp = transforms.Scale(r)(tmp).convert('RGB')
        tmp = transforms.ToTensor()(tmp)
        traindata_gt[image_id,:,:,:]=tmp
        #traindata_gt= tmp.convert('RGB')
        del tmp

        image_id=image_id+1
    #return traindata, traindata_gt, EPI_h, EPI_v
    return traindata_gt, EPI_h, EPI_v


def load_LFdata_all(r): 
    iframes = 0
    os.chdir('/scratch/tg3/xc0957/sequence_weak/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_1,EPI_h_1,EPI_v_1=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)

    os.chdir('/scratch/tg3/xc0957/Olivia/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_2,EPI_h_2,EPI_v_2=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    '''
    os.chdir('/scratch/tg3/xc0957/aqua/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_3,EPI_h_3,EPI_v_3=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    
    os.chdir('/scratch/tg3/xc0957/pigeons/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_4,EPI_h_4,EPI_v_4=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    '''
    os.chdir('/scratch/tg3/xc0957/peacock/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_3,EPI_h_3,EPI_v_3=load_LFdata(all_subdirs,r)    
    iframes = iframes + len(all_subdirs)


    os.chdir('/scratch/tg3/xc0957/snow3/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_4,EPI_h_4,EPI_v_4=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)

    os.chdir('/scratch/tg3/xc0957/snow4/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_5,EPI_h_5,EPI_v_5=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    print('iframes',iframes)

    traindata_gt = torch.cat((traindata_gt_1,traindata_gt_2,traindata_gt_3,traindata_gt_4,traindata_gt_5),0)
    EPI_h = torch.cat((EPI_h_1,EPI_h_2,EPI_h_3,EPI_h_4,EPI_h_5),0)
    EPI_v = torch.cat((EPI_v_1,EPI_v_2,EPI_v_3,EPI_v_4,EPI_v_5),0)
    #os.chdir('/scratch/tg3/xc0957/sequence/')
    return traindata_gt, EPI_h, EPI_v,iframes

def load_LFdata_all2(r): 
    iframes = 0
    os.chdir('/scratch/tg3/xc0957/')
    
    traindata_gt_1 = torch.load('traindata_gt_1.pt')
    EPI_h_1 = torch.load('EPI_h_1.pt')
    EPI_v_1 = torch.load('EPI_v_1.pt')
    iframes = iframes + 400

    traindata_gt_2 = torch.load('traindata_gt_2.pt')
    EPI_h_2 = torch.load('EPI_h_2.pt')
    EPI_v_2 = torch.load('EPI_v_2.pt')
    iframes = iframes + 400

    traindata_gt_3 = torch.load('traindata_gt_3.pt')
    EPI_h_3 = torch.load('EPI_h_3.pt')
    EPI_v_3 = torch.load('EPI_v_3.pt')
    iframes = iframes + 400

    traindata_gt_4 = torch.load('traindata_gt_4.pt')
    EPI_h_4 = torch.load('EPI_h_4.pt')
    EPI_v_4 = torch.load('EPI_v_4.pt')
    iframes = iframes + 400

    traindata_gt_5 = torch.load('traindata_gt_5.pt')
    EPI_h_5 = torch.load('EPI_h_5.pt')
    EPI_v_5 = torch.load('EPI_v_5.pt')
    iframes = iframes + 400

    traindata_gt_6 = torch.load('traindata_gt_6.pt')
    EPI_h_6 = torch.load('EPI_h_6.pt')
    EPI_v_6 = torch.load('EPI_v_6.pt')
    iframes = iframes + 400

    traindata_gt_7 = torch.load('traindata_gt_7.pt')
    EPI_h_7 = torch.load('EPI_h_7.pt')
    EPI_v_7 = torch.load('EPI_v_7.pt')
    iframes = iframes + 400

    traindata_gt_8 = torch.load('traindata_gt_8.pt')
    EPI_h_8 = torch.load('EPI_h_8.pt')
    EPI_v_8 = torch.load('EPI_v_8.pt')
    iframes = iframes + 400

    
    print('iframes',iframes)

    traindata_gt = torch.cat((traindata_gt_1,traindata_gt_2,traindata_gt_3,traindata_gt_4,traindata_gt_5,traindata_gt_6,traindata_gt_7,traindata_gt_8),0)
    EPI_h = torch.cat((EPI_h_1,EPI_h_2,EPI_h_3,EPI_h_4,EPI_h_5,EPI_h_6,EPI_h_7,EPI_h_8),0)
    EPI_v = torch.cat((EPI_v_1,EPI_v_2,EPI_v_3,EPI_v_4,EPI_v_5,EPI_v_6,EPI_v_7,EPI_v_8),0)
    #os.chdir('/scratch/tg3/xc0957/sequence/')
    return traindata_gt, EPI_h, EPI_v,iframes

def prep_lf(r):
#torch.save(tensor, 'file.pt') and torch.load('file.pt')
    iframes = 0
    '''
    os.chdir('/scratch/tg3/xc0957/sequence_weak/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_1,EPI_h_1,EPI_v_1=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    torch.save(traindata_gt_1, 'traindata_gt_1.pt')
    torch.save(EPI_h_1, 'EPI_h_1.pt')
    torch.save(EPI_v_1, 'EPI_v_1.pt')

    os.chdir('/scratch/tg3/xc0957/Olivia/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_2,EPI_h_2,EPI_v_2=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    torch.save(traindata_gt_2, 'traindata_gt_2.pt')
    torch.save(EPI_h_2, 'EPI_h_2.pt')
    torch.save(EPI_v_2, 'EPI_v_2.pt')

    os.chdir('/scratch/tg3/xc0957/sequence_high/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_3,EPI_h_3,EPI_v_3=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    torch.save(traindata_gt_3, 'traindata_gt_3.pt')
    torch.save(EPI_h_3, 'EPI_h_3.pt')
    torch.save(EPI_v_3, 'EPI_v_3.pt')
    
    os.chdir('/scratch/tg3/xc0957/aqua/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_6,EPI_h_6,EPI_v_6=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    torch.save(traindata_gt_6, 'traindata_gt_6.pt')
    torch.save(EPI_h_6, 'EPI_h_6.pt')
    torch.save(EPI_v_6, 'EPI_v_6.pt')

    '''
    os.chdir('/scratch/tg3/xc0957/pigeons/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_7,EPI_h_7,EPI_v_7=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    torch.save(traindata_gt_7, 'traindata_gt_7.pt')
    torch.save(EPI_h_7, 'EPI_h_7.pt')
    torch.save(EPI_v_7, 'EPI_v_7.pt')

    os.chdir('/scratch/tg3/xc0957/peacock/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_8,EPI_h_8,EPI_v_8=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    torch.save(traindata_gt_8, 'traindata_gt_8.pt')
    torch.save(EPI_h_8, 'EPI_h_8.pt')
    torch.save(EPI_v_8, 'EPI_v_8.pt')    
    '''    

    os.chdir('/scratch/tg3/xc0957/snow3/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_4,EPI_h_4,EPI_v_4=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    torch.save(traindata_gt_4, 'traindata_gt_4.pt')
    torch.save(EPI_h_4, 'EPI_h_4.pt')
    torch.save(EPI_v_4, 'EPI_v_4.pt')
    '''
    os.chdir('/scratch/tg3/xc0957/snow4/sequence/')
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print(all_subdirs)
    traindata_gt_5,EPI_h_5,EPI_v_5=load_LFdata(all_subdirs,r)
    iframes = iframes + len(all_subdirs)
    print('iframes',iframes)
    torch.save(traindata_gt_5, 'traindata_gt_5.pt')
    torch.save(EPI_h_5, 'EPI_h_5.pt')
    torch.save(EPI_v_5, 'EPI_v_5.pt')

    traindata_gt = torch.cat((traindata_gt_1,traindata_gt_2,traindata_gt_3,traindata_gt_4,traindata_gt_5),0)
    EPI_h = torch.cat((EPI_h_1,EPI_h_2,EPI_h_3,EPI_h_4,EPI_h_5),0)
    EPI_v = torch.cat((EPI_v_1,EPI_v_2,EPI_v_3,EPI_v_4,EPI_v_5),0)
    #os.chdir('/scratch/tg3/xc0957/sequence/')
    return traindata_gt, EPI_h, EPI_v
