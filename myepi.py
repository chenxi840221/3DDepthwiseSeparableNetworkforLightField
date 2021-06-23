#from IPython import display

import glob
#import imageio
#import matplotlib.pyplot as plt
import numpy as np
#import PIL
#import tensorflow as tf
#import tensorflow_probability as tfp
import time

from util import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# for reading and displaying images
#from skimage.io import imread
import matplotlib.pyplot as plt

# PyTorch libraries and modules
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import *
#import h5py

#from plot3D import *
#from apex import amp
#from apex.fp16_utils import *
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import os
import random
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

if torch.cuda.is_available():

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    else:    
        print('single GPU!')
    device = torch.device("cuda")      
else:
    device = torch.device("cpu")  
    print('CPU!')

#device = torch.device("cpu") 

def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]


def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
    return np.asarray(data_t, dtype=np.float32)


# LF encoder model
class LFE(nn.Module):
    def __init__(self, synth=0):
        super(LFE, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 128, 2, 1)
        
        #self.conv_layer2 = self._conv_layer_set(128, 256, 2, 2)

        self.conv_layer2 = self._conv_layer_set(128, 256, 1, 1)

        self.conv_layer3 = self._conv_layer_set(256, 512, 2, 2)

        self.conv_layer4 = self._conv_layer_set(512, 512, 1, 1)

        self.conv_layer5 = self._conv_layer_set(512, 1024, 2, 2)

        self.conv_layer6 = self._conv_layer_set(1024, 1024, 1, 1)

        self.conv_layer7 = self._conv_layer_set(1024, 2048, 2, 2)

        #self.t_conv_layer1 = nn.ConvTranspose2d(512, 512, 2)  # kernel_size=3 to get to a 10x10 image output

        self.t_conv_layer1 = self.t_conv_layer_set(4096, 2048)

        self.t_conv_layer2 = self.t_conv_layer_set2(2048, 1024)

        self.t_conv_layer3 = self.t_conv_layer_set(1024, 512)

        self.t_conv_layer4 = self.t_conv_layer_set2(512, 256)

        self.t_conv_layer5 = self.t_conv_layer_set(256, 128)

        #self.t_conv_layer6 = self.t_conv_layer_set(256, 128)

        self.t_conv_layer7 = self.t_conv_layer_set(128, 3)
        #self.fc1 = nn.Linear(2**3*64, 128)
        #self.fc2 = nn.Linear(128, 2)
        #self.relu = nn.LeakyReLU()
        #self.batch=nn.BatchNorm3d(32)
        #self.drop=nn.Dropout(p=0.15)        

    def _conv_layer_set(self, in_c, out_c, pool_xy, pool_uv):
        conv_layer = nn.Sequential(
        #nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.Conv3d(in_c, in_c, kernel_size=(3, 3, 3), padding=(1,1,1)),
        nn.BatchNorm3d(in_c),
        nn.LeakyReLU(),
        #nn.ReLU(inplace=True),
        nn.Conv3d(in_c, in_c, kernel_size=(1, 1, 1), padding=(0,0,0)),
        nn.BatchNorm3d(in_c),
        nn.LeakyReLU(),
        nn.Conv3d(in_c, in_c, kernel_size=(3, 3, 3), padding=(1,1,1)),
        nn.BatchNorm3d(in_c),
        nn.LeakyReLU(),
        #nn.ReLU(inplace=True),
        nn.Conv3d(in_c, out_c, kernel_size=(1, 1, 1), padding=(0,0,0)),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        #nn.MaxPool3d((2, 2, pool), return_indices=True),
        nn.MaxPool3d((pool_xy, pool_xy, pool_uv)),
        )
        return conv_layer

    def t_conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        #nn.MaxUnpool3d((2, 2, pool)),
        #nn.ConvTranspose2d(in_c, out_c, kernel_size=(1, 1), stride=2, padding=(0,0)),
        #nn.ConvTranspose2d(out_c, out_c, kernel_size=(3, 3), padding=(0,0)),
        #nn.ConvTranspose2d(in_c, in_c, kernel_size=(1, 1), padding=(0,0)),
        #nn.LeakyReLU(),
        nn.ConvTranspose2d(in_c, out_c, kernel_size=(3, 3), stride=2, padding=(1,1),output_padding=(1,1)),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(),
        #nn.MaxPool2d((2, 2)),
        )
        return conv_layer

    def t_conv_layer_set2(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(in_c),
        nn.LeakyReLU(),
        nn.Conv2d(in_c, in_c, kernel_size=(1, 1), padding=(0, 0)),
        nn.BatchNorm2d(in_c),
        nn.LeakyReLU(),
        nn.Conv2d(in_c, in_c, kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(in_c),
        nn.LeakyReLU(),
        nn.Conv2d(in_c, out_c, kernel_size=(1, 1), padding=(0, 0)),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(),
        #nn.MaxPool2d((2, 2)),
        )
        return conv_layer

    def forward(self, h, v):
        # Set 1
        #[h_e,i1] = self.conv_layer1(h)

        #[h_e2,i2] = self.conv_layer2(h_e)
        '''
        h
        '''
        h_e = self.conv_layer1(h)

        h_e2 = self.conv_layer2(h_e)

        h_e3 = self.conv_layer3(h_e2)

        h_e4 = self.conv_layer4(h_e3)

        h_e5 = self.conv_layer5(h_e4)

        h_e6 = self.conv_layer6(h_e5)

        h_e7 = self.conv_layer7(h_e6)

        h_vec = torch.squeeze(h_e7, 4)

        '''
        v
        '''

        v_e = self.conv_layer1(v)

        v_e2 = self.conv_layer2(v_e)

        v_e3 = self.conv_layer3(h_e2) 

        v_e4 = self.conv_layer4(v_e3)

        v_e5 = self.conv_layer5(h_e4)

        v_e6 = self.conv_layer6(v_e5)

        v_e7 = self.conv_layer7(v_e6)

        v_vec = torch.squeeze(v_e7, 4)

        c = torch.cat((h_vec,v_vec),1)

        c1 = self.t_conv_layer1(c)

        c2 = self.t_conv_layer2(c1)

        c3 = self.t_conv_layer3(c2)

        c4 = self.t_conv_layer4(c3)

        c5 = self.t_conv_layer5(c4)
        
        #c6 = self.t_conv_layer6(c5)

        out = F.sigmoid(self.t_conv_layer7(c5))
        #out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        #h_e = self.relu(h_e)
        #h_e = self.batch(h_e)
        #h_e = self.drop(h_e)
        #out = self.fc2(out)

        #v_e = self.conv_layer1(v)
        #v_e = self.relu(v_e)
        #v_e = self.batch(v_e)
        #v_e = self.drop(v_e)    

        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = F.relu(self.t_conv1(x))
        # x = F.sigmoid(self.t_conv2(x)) 

        return out


#os.chdir('D:\\workspace\\myepinet\\hci_dataset\\additional')
#os.chdir('D:\\workspace\\myepinet\\hci_dataset\\test')
#os.chdir('E:\\users\\xichen\\myepinet\\hci_dataset\\test')
#os.chdir('/scratch/tg3/xc0957/sequence/')
os.chdir('/scratch/tg3/xc0957/save_2/')
model = LFE(0)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
if os.path.isfile('model.ckpt'):
    model.load_state_dict(torch.load('model.ckpt'))        
    print('model.ckpt loaded!')

learning_rate = 0.0001
#criterion = nn.MSELoss() 
#criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss() 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
#model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
print(model)

#os.chdir('E:\\users\\xichen\\underwater\\sequence_low_resolution\\underwater\\sequence')
#os.chdir('/scratch/tg3/xc0957/sequence/')
#all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
#print(all_subdirs)
#for i in range(len(all_subdirs)):
#for subdir in all_subdirs:
r = 128
traindata_gt_all,EPI_h_all,EPI_v_all,iframes=load_LFdata_all2(256)

os.chdir('/scratch/tg3/xc0957/save_2/')
scaler = GradScaler()
batch_size = 24
#fig, (ax1, ax2) = plt.subplots(1, 2)
fig, axs = plt.subplots(2,3)
for epoch in range(300000):
    #for i in range(len(all_subdirs)):
    #for i in range(200):
        optimizer.zero_grad()
        traindata_gt_p=torch.FloatTensor(batch_size,3, r, r)
        EPI_h_p=torch.FloatTensor(batch_size, 3, r, r, 9)
        EPI_v_p=torch.FloatTensor(batch_size, 3, r, r, 9)

        for bb in range(batch_size):

            #traindata_gt_p=torch.FloatTensor(batch_size,3, r, r)
            #EPI_h_p=torch.FloatTensor(batch_size, 3, r, r, 9)
            #EPI_v_p=torch.FloatTensor(batch_size, 3, r, r, 9)

            i = random.randrange(iframes)
            u_i = random.randrange(128)
            v_i = random.randrange(128)

            traindata_gt_p[bb,:,:,:]=traindata_gt_all[i,:,u_i:u_i+128,v_i:v_i+128]
            EPI_h_p[bb,:,:,:,:]=EPI_h_all[i,:,u_i:u_i+128,v_i:v_i+128,:]
            EPI_v_p[bb,:,:,:,:]=EPI_v_all[i,:,u_i:u_i+128,v_i:v_i+128,:]
        
        print('load done')

        #EPI_h = EPI_h.unsqueeze(0)
        #EPI_v = EPI_v.unsqueeze(0)
        #traindata_gt = traindata_gt.unsqueeze(0)

        EPI_h_p = EPI_h_p.to(device)
        EPI_v_p = EPI_v_p.to(device)
        traindata_gt_p = traindata_gt_p.to(device)
        mask = torch.zeros_like(traindata_gt_p)
        mask[:,:,14:114,14:114] = 1.0
        with autocast():
            out = model(EPI_h_p,EPI_v_p)
            masked_out = mask * out
            masked_traindata_gt_p = mask * traindata_gt_p
            loss = criterion(out,traindata_gt_p)
            #loss = criterion(masked_out,masked_traindata_gt_p)
            print('loss:',loss,'\n')

        #o1,o2,o3,o5,o7,v,c,c1,c2,c3,c4,c5,out = model(EPI_h_p,EPI_v_p)
        #loss = criterion(out,traindata_gt_p)
        #print('loss:',loss,'\n')
        # out = o12.squeeze(0)
        # img = out.detach().numpy()
        # img = img.transpose(1, 2, 0)
        # cv2.imshow('out',img)


        #input = torch.randn(3, 5, requires_grad=True)
        #target = torch.empty(3, dtype=torch.long).random_(5)
        #output = criterion(input, target)

        #out = c5.squeeze(0)
        #img = out.detach().numpy()
        #img = img.transpose(1, 2, 0)
        #fig2, ax2 = plt.subplots()
        #ax2.imshow(img)
        #plt.show(block=False)


        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch%2000 == 0:
            #img = out.squeeze(0)
            inp = EPI_h_p.float().cpu().detach().numpy()
            inp = inp[:,:,:,:,4]
            inp = inp.transpose(0, 2, 3, 1)

            inp_l = EPI_h_p.float().cpu().detach().numpy()
            inp_l = inp_l[:,:,:,:,0]
            inp_l = inp_l.transpose(0, 2, 3, 1)

            inp_r = EPI_h_p.float().cpu().detach().numpy()
            inp_r = inp_r[:,:,:,:,8]
            inp_r = inp_r.transpose(0, 2, 3, 1)

            inp_u = EPI_v_p.float().cpu().detach().numpy()
            inp_u = inp_u[:,:,:,:,0]
            inp_u = inp_u.transpose(0, 2, 3, 1)


            out = out.float().cpu().detach().numpy()
            #out = masked_out.float().cpu().detach().numpy()
            out = out.transpose(0, 2, 3, 1)
            #gt = traindata_gt.squeeze(0)
            gt_np = traindata_gt_p.cpu().detach().numpy()
            #gt_np = masked_traindata_gt_p.cpu().detach().numpy()
            gt_np = gt_np.transpose(0, 2, 3, 1) 

            savefile = 'Batch'+str(bb)+'Epoch'+str(epoch)+'Loss'+str(loss.cpu().detach().numpy())+'.sav'
            EPI_h_sav = EPI_h_p.float().cpu()
            EPI_v_sav = EPI_v_p.float().cpu()
            db = {'a': EPI_h_sav, 'b': EPI_v_sav}
            torch.save(db, savefile)
#loaded = torch.load(path/'torch_db')
#print( loaded['a'] == tensor_a )
#print( loaded['b'] == tensor_b )
            #axs[0,0].set_title("Input", fontsize=10)1
            #axs[0,1].set_title("Output", fontsize=10)
            #axs[0,2].set_title("GT", fontsize=10)
            sub = 0
            nsub = 0
            for bb in range(batch_size):
                cc = bb - nsub * 2
                axs[0,0].imshow(inp_l[bb,:,:,:])
                axs[0,1].imshow(inp_r[bb,:,:,:])
                axs[0,2].imshow(inp_u[bb,:,:,:])
                axs[1,0].imshow(inp[bb,:,:,:])
                axs[1,1].imshow(out[bb,:,:,:])
                axs[1,2].imshow(gt_np[bb,:,:,:])
                axs[0,0].axis('off')
                axs[0,1].axis('off')
                axs[0,2].axis('off')
                axs[1,0].axis('off')
                axs[1,1].axis('off')
                axs[1,2].axis('off')
                savefile = 'Batch'+str(bb)+'epoch'+str(epoch)+'Loss'+str(loss.cpu().detach().numpy())+'.pdf'
                plt.savefig(savefile, dpi=1200)
                sub = sub + 1
                if sub == 2:
                    sub = 0
                    nsub = nsub + 1


            torch.save(model.state_dict(), 'model.ckpt')        
print('end\n')



