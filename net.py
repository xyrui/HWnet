# -*- coding: utf-8 -*-
"""
@author: Xyrui 
"""
import torch
from torch import nn
from torch.autograd import Function as autoF
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import torch.utils.data as uData
from math import log, pi
from scipy.special import digamma, gammaln

 
###############################################################################################
# --------------------Create DnCNN P3D------------------------------------------------------   
###############################################################################################
def conv3x3x1(in_chn, out_chn, bias=True):
    layer = nn.Conv3d(in_chn, out_chn, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), bias = bias)
    return layer

def conv1x1x3(in_chn, out_chn, bias = True):
    layer = nn.Conv3d(in_chn, out_chn, kernel_size=(3,1,1), stride = 1, padding = (1,0,0), bias = bias)
    return layer

class DnCNN_P3D_B(nn.Module):
    def __init__(self, in_chn=1, out_chn=1, dep=5, num_filters = 64, bias = True):
        super(DnCNN_P3D_B,self).__init__()
        self.conv1 = conv3x3x1(in_chn, num_filters, bias=bias)
        self.conv2 = conv1x1x3(num_filters, num_filters, bias=bias)
        self.relu = nn.ReLU(inplace = True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3x1(num_filters, num_filters, bias=bias))
            mid_layer.append(nn.ReLU(inplace=True))
            mid_layer.append(conv1x1x3(num_filters, num_filters, bias = bias))
            mid_layer.append(nn.BatchNorm3d(num_filters))
            mid_layer.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = nn.Conv3d(num_filters, out_chn, kernel_size = (3,3,3), stride = 1, padding = 1, bias = bias)
        self.BN = nn.BatchNorm3d(1)
      #  self.conv_last2 = nn.Conv3d(num_filters, 20, kernel_size = (3,3,3), stride = 1, padding = 1, bias = bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        return x
    
## Diagnalization supported on Batches #################        
def bdiag(X):
    Y = torch.zeros(X.size(0), X.size(1), X.size(1)).to(device=X.device).type(dtype=X.dtype)
    Y.as_strided(X.size(), [Y.stride(0), Y.size(2) + 1]).copy_(X)
    return Y   #Every element of Y has grad, but it does not matter.
  
    
## Define SVD with back propagation #################  
## Based on "Training deep networks with structured layers by matrix backpropagation"
class difSVD2(torch.autograd.Function): #X\in R^{MxN}, and M<<N
    @staticmethod
    def forward(ctx, input,r):
        X = input.detach().cpu().numpy()
        U,S,_ = np.linalg.svd(np.matmul(X, X.swapaxes(1,2)))        
        U = torch.from_numpy(U[:,:,:r]).to(device=input.device).type(dtype=input.dtype)
        S = torch.from_numpy(np.sqrt(S[:,:r])).to(device=input.device).type(dtype=input.dtype)
        V = torch.matmul(input.transpose(2,1), torch.matmul(U, bdiag(1/S)))

        ctx.save_for_backward(U,S,V)
        
        return U,S,V  #S is of vector form.
    
    @staticmethod
    def backward(ctx, g_U, g_S, g_V):
        U,S,V = ctx.saved_tensors
        K = torch.matmul(torch.unsqueeze(S**2,2),torch.ones(1,S.size(1)).to(device=S.device)) 
        tK = torch.eye(S.size(1)).to(device=S.device)
        K = 1/(K.transpose(1,2) - K + tK) - tK
        D = torch.matmul(g_U, bdiag(1/S))
        S, g_S = bdiag(S), bdiag(g_S)
        T3 = K*(torch.matmul(V.transpose(1,2),g_V - torch.matmul(torch.matmul(torch.matmul(V,D.transpose(1,2)),U),S)))
        T3 = T3 + T3.transpose(1,2)
        T3 = torch.matmul(U, torch.matmul(S, torch.matmul(T3,V.transpose(1,2))))
        T2 = (g_S - torch.matmul(U.transpose(1,2),D))*torch.eye(S.size(1)).to(device=U.device)
        g_X = torch.matmul(D,V.transpose(1,2)) + torch.matmul(U,torch.matmul(T2, V.transpose(1,2))) + T3
        
        return g_X, None 

DifSVD = difSVD2.apply

##################################################################################################
#-----------------------------------HWLRMF-------------------------------------------------------#
##################################################################################################
class HWLRMF(nn.Module):
    def __init__(self, Ite=20, r = 3):
        super(HWLRMF, self).__init__()
        self.Ite = Ite
        self.r = r
        
    def forward(self, x, W, x_g=0):
        Band,Cha,Hei,Wid = x.size()
        W = W.reshape(Band, Cha, Hei*Wid)
 
        
        x = x.reshape(Band, Cha, Hei*Wid)
        r = self.r
        L = x.clone()
        
        rho = 0.5*torch.mean(W)
        
        alpha = 1.05
        loss = []
        loss_F = []
    
        for i in range(self.Ite):
            
            U,_,_ = DifSVD(L, r)
            UV = torch.matmul(U, torch.matmul(U.transpose(1,2), L))
            
            '''
            U,S,V = DifSVD(L,r)
            UV = torch.matmul(U, torch.matmul(bdiag(S), V.transpose(1,2)))
            '''
            loss.append(torch.mean((UV.view(Band, Cha ,Hei, Wid) - x_g)**2).detach().cpu().numpy())          
            loss_F.append(torch.mean(W*(UV - x)**2).detach().cpu().numpy())
            
            mu = rho*torch.exp(-torch.log(W + rho))
            L = (1-mu)*x + mu*UV
            
            rho = alpha*rho
        
        return UV.reshape(Band, Cha, Hei, Wid), np.array(loss_F), np.array(loss)
    
    

    
    
     

    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            