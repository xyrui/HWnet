# -*- coding: utf-8 -*-
"""
@author: Xyrui 
"""
import numpy as np
from skimage.util import random_noise
import random as prand
import torch as torch
from scipy.special import digamma, gammaln
from torch.autograd import Function as autoF
from math import log, pi
import cv2 as cv2
import torch.utils.data as uData
from functools import partial
import scipy.io as sio 

#############################################################################################
# --------------Create Training Data --------------------------------------------------------------
#############################################################################################
def gaussian_kernel2(H, W, B, scale):
    centerSpa1 = np.random.randint(1,H-1, size=B)
    centerSpa2 = np.random.randint(1,W-1, size=B)
    XX, YY = np.meshgrid(np.arange(W), np.arange(H))
    out = np.exp((-(np.expand_dims(XX,-1)-centerSpa1)**2-(np.expand_dims(YY, -1)-centerSpa2)**2)/(2*scale**2))
    return out  

def add_noniid_gaussian(x, *scale):
    pch_size = x.shape
    if scale == ():
        scale = np.random.uniform(32/2,128/2,size = pch_size[2])
    else:
        scale = scale[0]
    sig_mi = 5/255
    sig_ma = 75/255

    p_sigma_ = gaussian_kernel2(pch_size[0], pch_size[1], pch_size[2], scale)  
    p_sigma_ = (p_sigma_ - p_sigma_.min())/(p_sigma_.max()-p_sigma_.min())
    p_sigma_ = sig_mi + p_sigma_*(sig_ma - sig_mi)
    noise = np.random.randn(pch_size[0], pch_size[1], pch_size[2]) * p_sigma_
    x = x+ noise
    return x, p_sigma_

def add_iid_gaussian1(x, *sig):  
    if sig == ():
       sig = prand.uniform(10/255,70/255)
    else:
       sig = sig[0]
    s = x.shape
    x = x + np.random.randn(s[0],s[1],s[2])*sig
    return x, np.ones(s)*sig
 
def add_iid_gaussian2(x):
    s = x.shape
    sig = np.random.rand(s[-1])*(60/255)+10/255
    x = x+ np.random.randn(s[0], s[1], s[2])*sig
    return x, sig*np.ones(s)

def add_impluse(x,bn):
    B = x.shape[-1]
  #  ratio = prand.uniform(0.01,0.15)
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    ratio = np.random.uniform(0.1,0.5,size=bn)
    for i in range(bn):
        x[:,:,band[i]] = random_noise(x[:,:,band[i]], mode = 's&p', clip = False, amount = ratio[i])
    
    return x, band, ratio

def add_stripe(x, bn):
    N = x.shape[-2]
    B = x.shape[-1]
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    stripn = np.random.randint(int(N*0.05),int(N*0.2),size = bn)
    for i in range(bn):
        loc = prand.sample(range(N), stripn[i])
        stripes = np.random.rand(stripn[i])*0.5 - 0.25
        x[:,loc, band[i]] = x[:,loc, band[i]] - stripes
        
    return x, band, stripn

def add_deadline(x, bn):
    N = x.shape[-2]
    B = x.shape[-1]
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    dn = np.random.randint(int(N*0.05),int(N*0.2),size = bn)
    for i in range(bn):
        loc = prand.sample(range(N), dn[i])
        x[:,loc, band[i]] = 0
        
    return x, band, dn

class Train_builder1(uData.Dataset):
    def __init__(self, im_mat_list, num_patch, nlist):
        super(Train_builder1, self).__init__()
        self.num_patch = num_patch
        self.im_mat_list = im_mat_list
        self.ndict = {'iid1':add_iid_gaussian1,
                      'iid2':add_iid_gaussian2,
                      'non':add_noniid_gaussian,
                      'impluse':partial(add_impluse, bn = 3),
                      'stripe':partial(add_stripe, bn = 3),
                      'deadline':partial(add_deadline, bn=3)}
        self.nname = ['iid1','iid2','non','impluse','stripe','deadline']
        self.nlist = nlist
        
    def __len__(self):
        return self.num_patch
    
    def __getitem__(self, index):
        im_label = sio.loadmat(self.im_mat_list[index])['patch']  
        
       # ntype = np.random.randint(0,6)
        ntype = prand.sample(self.nlist, 1)[0]
        tinput = self.ndict[self.nname[ntype]](im_label)
        im_input = tinput[0]
        if ntype in [0,1,2]:
            noi_map = tinput[1]
        else:
            noi_map = None
            
        im_label = torch.from_numpy(np.transpose(im_label.copy(), (2,0,1))).type(torch.float32)  
        im_input = torch.from_numpy(np.transpose(im_input.copy(), (2,0,1))).type(torch.float32)
        noi_map = torch.from_numpy(np.transpose(noi_map.copy(), (2,0,1))).type(torch.float32)

        return im_input, im_label, noi_map
    
class Test_builder(uData.Dataset):
    def __init__(self, im_mat_list):
        super(Test_builder, self).__init__()
        self.num_img = len(im_mat_list)
        self.im_mat_list = im_mat_list

        
    def __len__(self):
        return self.num_img
    
    def __getitem__(self, index):
        data = sio.loadmat(self.im_mat_list[index])
        im_input = data['input']
        im_label = data['label']
        
        im_input = torch.from_numpy(np.transpose(im_input.copy(), (2,0,1))).type(torch.float32)
        im_input = torch.unsqueeze(im_input,0)
        im_label = np.float32(im_label)

        return im_input, im_label
    
def sta(img, mode):
    img = np.float32(img)
    if mode == 'all':
        ma = np.max(img)
        mi = np.min(img)
     #   return (img - mi)/(ma - mi)
        img = (img - mi)/(ma - mi)
        return img
    elif mode == 'pb':
        ma = np.max(img, axis=(0,1))
        mi = np.min(img, axis=(0,1))
        img = (img - mi)/(ma - mi)
        return img
        
    else:
        print('Undefined Mode!')
        return img
    
#########################################################################################################################################
# -------------------------------------- Define Loss ----------------------------------------------------------------------------------------
#########################################################################################################################################  
     
class Log_gamma(autoF):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = gammaln(input_np)
        out = torch.from_numpy(out).to(device=input.device).type(dtype=input.dtype)
        
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output
        
        return grad_input

log_gamma = Log_gamma.apply

def loss_repara(im_input, im_gt, pred_mu, m2, alpha, beta, rsam, ep0, sigmap, p=7):
    pred_mu.clamp_(min = log(1e-10), max=log(1e10))
    sigmap = sigmap**2
    
    log_beta = torch.log(beta)
    alpha_div_beta = torch.exp(torch.log(alpha) - log_beta)
    
    lh = 0.5*log(2*pi) + 0.5*torch.mean(log_beta - torch.digamma(alpha)) + 0.5*torch.mean(m2*alpha_div_beta)+ 0.5*torch.mean(rsam*(im_input - pred_mu)**2)
    
    kl_z_sig_sig = torch.mean( (alpha - p**2/2 -1)*torch.digamma(alpha) + gammaln(p**2/2+1) - log_gamma(alpha) + \
                              (p**2/2+1)*(log_beta - torch.log(p**2*sigmap/2)) + alpha_div_beta*0.5*p**2*sigmap - alpha)
    
    t_m = (pred_mu - im_gt)**2
    kl_z_sig_z = 0.5*torch.mean(t_m/ep0 + m2/ep0 - log(m2/ep0) -1)
    
    loss = lh + kl_z_sig_sig + kl_z_sig_z
    
    mse = torch.mean(t_m)
    
    return loss, lh, kl_z_sig_z, kl_z_sig_sig, mse


    
        
        

