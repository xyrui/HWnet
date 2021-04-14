"""
@author: Xyrui 
"""
from net import HWLRMF, DnCNN_P3D_B
import lib as lib
import numpy as np
import torch
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn

if __name__ == '__main__':
    
    # test on one HSI
    dname = 'ICVL_stripe'
    mname = 'mse'
    data = sio.loadmat('./data/'+dname+'/'+dname+'.mat')
    data_label = np.float32(data['label'])
    data_input = np.float32(data['input'])
    
    h_,w_,k_ = np.shape(data_label)

    netS = DnCNN_P3D_B(in_chn=1, out_chn=2, dep=5)
    netS = netS.cuda()
    netD = HWLRMF(150,3)
    netD = netD.cuda()
    
    checkpoint = torch.load('./checkpoints/' + 'RELRMF_50')
    netS.load_state_dict(checkpoint['model_state_dict'])
    netS.eval()     
    
    with torch.set_grad_enabled(False):
        ti = torch.cuda.FloatTensor(np.expand_dims(np.transpose(data_input,[2,0,1]),0))
        tl = torch.cuda.FloatTensor(np.expand_dims(np.transpose(data_label,[2,0,1]),0))
        pred_map = netS(torch.unsqueeze(ti,1))
        alpha = torch.exp(pred_map[:,0,:,:,:]) +1
        beta = torch.exp(pred_map[:,1,:,:,:]) 
        rsam_si = torch.exp(torch.log(alpha -1 ) - torch.log(beta)) #evaluate weight as (\alpha-1)/\beta
        pred, pred_loss, mse= netD(ti, rsam_si, tl)
          
    im = np.transpose(np.squeeze(pred.cpu().numpy()),[1,2,0])
    im[np.where(im>1.)]=1.
    im[np.where(im<0.)]=0.
    W = rsam_si.cpu().numpy()
    W = np.transpose(np.squeeze(W),[1,2,0])
    to_psnr = []
    to_ssim = []
    for i in range(k_):  
        
        plt.figure(dpi=200)
        plt.subplot(1,3,1)     
        plt.imshow(data_label[:,:,i],cmap = plt.cm.gray)      
        plt.title('Clean')
        plt.xticks([])
        plt.yticks([])
    
        plt.subplot(1,3,2)     
        plt.imshow(data_input[:,:,i],cmap = plt.cm.gray)      
        plt.title('Noisy')
        plt.xlabel('psnr=%.2f, ssim=%.2f' %(PSNR(data_label[:,:,i],data_input[:,:,i]),SSIM(data_label[:,:,i],data_input[:,:,i])))
        plt.xticks([])
        plt.yticks([])
 
        plt.subplot(1,3,3)    
        plt.imshow(im[:,:,i],cmap = plt.cm.gray)
        plt.title('Denoising')
        plt.xlabel('psnr=%.2f,ssim=%.2f' %(PSNR(data_label[:,:,i],im[:,:,i]),SSIM(data_label[:,:,i],im[:,:,i]))) 
        plt.xticks([])
        plt.yticks([])
        
       # plt.savefig('./result/pic/stuffed_' + str(i) + '.png')
        plt.show()
        
        to_psnr.append(PSNR(data_label[:,:,i], im[:,:,i]))
        to_ssim.append(SSIM(data_label[:,:,i], im[:,:,i]))
            
    to_psnr = np.array(to_psnr)
    to_ssim = np.array(to_ssim)
       
        
    print('PSNR: %.6f' % np.mean(to_psnr))
    print('SSIM: %.4f' % np.mean(to_ssim))
    
    '''
    # Plot 1/\sqrt(W)
    s = 1/np.sqrt(W)
    XX, YY = np.meshgrid(np.arange(w_), np.arange(h_))
    
    fig1=plt.figure()
    ax1=Axes3D(fig1)
    ax1.plot_surface(XX,YY,s[:,:,2],cmap=plt.cm.coolwarm)
    plt.show()
    '''
    
