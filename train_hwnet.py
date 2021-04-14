import torch
import numpy as np
import os
import argparse
import torch.nn as nn
import torch.optim as optim
from net import HWLRMF, DnCNN_P3D_B
from torch.utils.data import DataLoader
from skimage.measure import compare_psnr as PSNR
import time as ti
import glob
import lib as lib
from math import log as log
from os.path import join
import socket
from datetime import datetime
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description = 'HWLRMF')
parser.add_argument('--patch_size', dest = 'patch_size', default = [96,20])
parser.add_argument('--batch_size', dest = 'batch_size', default = 20)
parser.add_argument('--save_path', dest = 'save_path', default = './checkpoints/', help = 'pretrained models are saved here')
parser.add_argument('--saved_model',dest = 'saved_model', default ='', help='pre trained model')
parser.add_argument('--dataroot', dest = 'dataroot', type=str, default = '/media/jd/Model/Ruixy/DATA/RELRMF_patches', help = 'data path')
parser.add_argument('--learning_rate', dest = 'learning_rate', type = float, default = 1e-3, help = 'learning rate')    
parser.add_argument('--epoch', dest = 'epoch', type=int, default = 50)
parser.add_argument('--gpu_en', default="1", help = 'GPU ids')
args = parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_en
    print(args)
    
    patch_size = args.patch_size
    batch_size = args.batch_size
    
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    data_list_path =glob.glob(join(args.dataroot,'*.mat'))
    lent = len(data_list_path)
    Train_data = lib.Train_builder1(data_list_path, lent, [2])
    Train_dataset = DataLoader(Train_data, batch_size, shuffle=True)
       
    Vali_data = lib.Vali_builder('/media/jd/Model/Ruixy/DATA/HWLRMF_vali')
    Vali_dataset = DataLoader(Vali_data, 1, shuffle=False)

    netS = DnCNN_P3D_B(in_chn=1, out_chn=2, dep=5)
    netS = nn.DataParallel(netS.cuda())
    netD = HWLRMF(20,3)
    netD = netD.cuda()
    
    optimizer = optim.Adam(netS.parameters(), lr=args.learning_rate)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.8)

    if args.saved_model:
        print('Load pre trained model ' + args.saved_model)
        checkpoint = torch.load(args.saved_model)
        epoch_start = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        step = checkpoint['step']
        netS.load_state_dict(checkpoint['model_state_dict'])
        print('Load Successfully!')
    else:
        print('Initializing...')
        for m in netS.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if not m.bias is None:
                 nn.init.constant_(m.bias, 0)
                 
        epoch_start = 0
        step = 0
        
    # build summary writer
    log_dir = join(args.save_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 
    log_dir = join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    print('start training')
    
    param = [x for name, x in netS.named_parameters()]
    
    lend = len(Train_data)/batch_size
    for ep in range(epoch_start, args.epoch):
        
        tic = ti.time()
        netS.train()
        lr = optimizer.param_groups[0]['lr']
        if lr< 1e-6:
            print('Lowest learning rate warning! Break!')
            break
        else:
            print('lr:',lr)
        
        wloss = 0
        wlh = 0
        wklz = 0
        wklsig = 0
        wmse = 0
            
        for ii, t_p in enumerate(Train_dataset):           
            Binput, Blabel, Bmap = [x.type(torch.cuda.FloatTensor) for x in t_p]
            
            optimizer.zero_grad()
            pred_map = netS(torch.unsqueeze(Binput, 1))
            pred_map.clamp_(min = log(1e-10), max = log(1e10))
            alpha = torch.exp(pred_map[:,0,:,:,:]) + 1
            beta = torch.exp(pred_map[:,1,:,:,:])
            
            u = torch.distributions.gamma.Gamma(alpha, 1)  
            sample_size = 2
            rsam = u.rsample(torch.Size([sample_size]))*torch.exp(-torch.log(beta)) 
            rsam = rsam.reshape(-1, patch_size[1], patch_size[0], patch_size[0])
            
            pred, loss_F, mse = netD(Binput.repeat(sample_size,1,1,1), rsam, Blabel.repeat(sample_size,1,1,1))
            
            loss, lh, kl_z, kl_sig, mse = lib.loss_repara(Binput.repeat(sample_size,1,1,1), Blabel.repeat(sample_size,1,1,1), pred, 1e-2, alpha, beta, rsam, 1e-5, Bmap)
            
            loss.backward()
            total_norm = nn.utils.clip_grad_norm_(param, 1e3)
            optimizer.step()
            
            writer.add_scalar(join('train_loss_step'), loss.item(), step)
            writer.add_scalar(join('train_lh_step'), lh.item(), step)
            writer.add_scalar(join('train_kl_z_step'), kl_z.item(), step)
            writer.add_scalar(join('train_kl_sig_step'), kl_sig.item(), step)
            writer.add_scalar(join('train_mse_step'), mse.item(), step)
            
            wloss += loss.item()
            wlh += lh.item()
            wklz += kl_z.item()
            wklsig += kl_sig.item()
            wmse += mse.item()
        
            if (ii+1)%20 == 0:
                print('----- Batch [%4d]/[%4d][%4d] -----' %(ii+1, lend, ep+1))
                print('--- loss: %.3e -- lh:%.3e -- kl_z: %.3e -- kl_sigma: %.3e -- mse:%.3e -- norm:%.3e' % 
                      (loss.item(), lh.item(), kl_z.item(), kl_sig.item(), mse.item(), total_norm))
            step +=1            

        toc = ti.time()
        lr_scheduler.step()
        
        writer.add_scalar(join('train_loss_epoch'), wloss/lend, ep)
        writer.add_scalar(join('train_lh_epoch'), wlh/lend, ep)
        writer.add_scalar(join('train_kl_z_epoch'), wklz/lend, ep)
        writer.add_scalar(join('train_kl_sig_epoch'), wklsig/lend, ep)
        writer.add_scalar(join('train_mse_epoch'), wmse/lend, ep)
        
        print('----- Epoch [%4d]/[%4d] -- time: %.4f -----' %(ep+1, args.epoch, toc-tic))
        print('--- loss: %.3e -- likelihood: %.3e -- KL_Z: %.3e -- KL_Sigma: %.3e -- mse: %.3e ---' % (wloss/lend, 
                                     wlh/lend, wklz/lend, wklsig/lend, wmse/lend))
        print('#######################')
            
        #save model
        ep +=1
        torch.save({
            'epoch': ep,
            'step': step ,
            'model_state_dict': netS.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, args.save_path+'/HWLRMF_' + str(ep))

        
        #Validation       
        netS.eval()
        vp = []
        with torch.no_grad():
            for i, (Input, Label) in enumerate(Vali_dataset):
                Input = Input.type(torch.cuda.FloatTensor)
                Label = np.squeeze(Label.numpy())
                
                pred_map = netS(Input)
                alpha = torch.exp(pred_map[:,0,:,:,:]) +1
                beta = torch.exp(pred_map[:,1,:,:,:]) 
                rsam_si = torch.exp(torch.log(alpha -1 ) - torch.log(beta))
                pred, vali_pred_loss,_ = netD(Input, rsam_si)
        
                vali_pred = np.transpose(np.squeeze(pred.cpu().numpy()),[1,2,0])
                vali_pred[np.where(vali_pred<0.)]=0.
                vali_pred[np.where(vali_pred>1.)]=1.
                vali_psnr = 0
                for j in range(Label.shape[-1]):
                    vali_psnr += PSNR(Label[:,:,j], vali_pred[:,:,j])
                vali_psnr /= Label.shape[-1]
                vp.append(vali_psnr)
        vp = np.mean(np.array(vp))
                
        writer.add_scalar(join('vali_psnr'), vp, ep-1)
        
        print('-- Eval -- PSNR: %.6f --' % (vp))
        print('#######################')

        
    print('Finish training!')
    
if __name__ == '__main__':
    main()
    
    
        
            
                
            
            
        


















