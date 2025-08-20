##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
from func import *
import argparse
import scipy.io as sio
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import MSD_model, ADMM_Iter
import metric as final_metric
import warnings 

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main_Raman_De(data_name, args):
    #----------------------- Data Configuration -----------------------#
    dataset_dir = '../Raman_Denoising_Dataset/PSBall_data/PS_PMMA/'
    result_dir = './Results/RealScene/' + data_name + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    data = sio.loadmat(dataset_dir + data_name + '.mat')
    try:
        meas_data = torch.from_numpy(data['Noisy_RHSI']).float().to(device).unsqueeze(0)
        meas_data_tensor = meas_data.permute(0, 3, 1, 2)/args.scale
        GT_data_tensor = torch.from_numpy(data['HR_RHSI']).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)
        GT_data_tensor = (GT_data_tensor - torch.min(GT_data_tensor)) / (torch.max(GT_data_tensor) - torch.min(GT_data_tensor))
    except:
        # Without High-SNR Data
        print(20*'-', 'Test without HR_RHSI', 20*'-')
        GT_data_tensor = torch.from_numpy(data[args.noise_type]).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)/args.scale
        meas_data = GT_data_tensor
        meas_data_tensor = GT_data_tensor
    
    if GT_data_tensor.shape != meas_data_tensor.shape:
        GT_data_tensor = F.interpolate(GT_data_tensor, size=(meas_data_tensor.shape[2], meas_data_tensor.shape[3]), mode='bilinear', align_corners=False)
        print('GT data size:', GT_data_tensor.shape, 'Noisy data size:', meas_data_tensor.shape) 
        
    print('\n')
    print('------------------------------------ Test Scene:', data_name, ' / Band-height-width : ', meas_data_tensor.shape[1:-1])
    print(f"GT mean: {torch.mean(GT_data_tensor).item():.3f}", f"GT min: {torch.min(GT_data_tensor).item():.3f}", f"GT max: {torch.max(GT_data_tensor).item():.3f}")
    print(f"Meas mean: {torch.mean(meas_data).item():.3f}", f"Meas min: {torch.min(meas_data).item():.4f}", f"Meas max: {torch.max(meas_data).item():.3f}")
    print(f"Adjust Meas mean: {torch.mean(meas_data_tensor).item():.3f}", f"Meas min: {torch.min(meas_data_tensor).item():.3f}", f"Meas max: {torch.max(meas_data_tensor).item():.3f}")
    print('----------------------------------- Adjust the mean of noisy Raman images ----------------------------------')


    #------------------------- Training Model -------------------------#
    if args.PnP_i:
        recon = ADMM_Iter(GT_data_tensor.to(device), meas_data_tensor.to(device), result_dir, args)
    else:
        args.Epoc_num = 2400
        recon = MSD_model(GT_data_tensor.to(device), meas_data_tensor.to(device), meas_data_tensor.to(device), args)
        recon = recon.squeeze().permute(1, 2, 0).cpu().numpy()

        print('Our results:')
        GT_data = GT_data_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        final_metric.evaluate_all_results(GT_data, recon)
        sio.savemat('{}/LSD_PnP{}.mat'.format(result_dir, args.PnP_i), {'R_hsi':recon})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset',      default = 'Real',         help="Choose test dataset: Real")
    parser.add_argument('--noise_type',   default = 'Noisy_RHSI',   help="Choose test nois data")
    parser.add_argument('--PnP_i',        default = 1,              help="State of solo or PnP model")
    parser.add_argument('--vis_plot',     default = 0,              help="State of plot spectrum")
    parser.add_argument('--iter_num',     default = 20,             help="Number of ADMM iterations")
    parser.add_argument('--Epoc_num',     default = 1200,           help="Number of MSD iterations")
    parser.add_argument('--scale',        default = 2.0,            help="Factor of scaling mesurements")
    parser.add_argument('--rho',          default = 20.,            help="Factor of dual variable")
    parser.add_argument('--lambda_S',     default = 1,              help="Factor of sparse noise")
    parser.add_argument('--lambda_R',     default = 0.4,            help="Factor of TV/STV regularization")
    parser.add_argument('--lambda_STV',   default = 0.2,            help="Ratio of STV regularization")
    parser.add_argument('--lambda_D',     default = 0.8,            help="Decay factor of lambda_R")
    args = parser.parse_args()

    if args.Dataset   == 'Real':
        # 'Cell_2DScene', 'Cell_Paramecium', 'Cell_ParameciumS2',
        # 'PS_ball', 'PS_ball_data_532', 'PS_ball_data_633', 'PS_PMMA', 'PMMA'
        # 'Tablet1_64x64_50x', 'Tablet1_128x128_10x', 'Tablet1_linescan'
        data_list     =  ['PS_ball'] 
    else:
        print("---------- Ensure the name of dataset ----------") 

    for file_name in data_list:
        if file_name == 'PS_ball':
            # --------          PS_ball include: Noisy_RHSI (10% Laser) / HR_RHSI (100% Laser)       -------- #
            args.scale = 345.95
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.7, 0.2, 0.8
        elif file_name == 'PS_ball_data_532':
            # --------                 PS_ball_data_532 include: Noisy_RHSI_100 (0.01s)              -------- #
            args.noise_type = 'Noisy_RHSI_100'
            args.scale = 162
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.1, 0.2, 0.8
        elif file_name == 'PS_ball_data_633':
            # --------                 PS_ball_data_633 include: Noisy_RHSI_100 (0.01s)              -------- #
            args.noise_type = 'Noisy_RHSI_100_50Laser'
            args.scale = 162
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.1, 0.3, 0.8
        elif file_name == 'PS_PMMA':
            # --------                     PS_PMMA include: Noisy_RHSI_10 (0.1s)                     -------- #
            args.noise_type = 'Noisy_RHSI_10'
            args.scale = 702
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.1, 0.4, 0.8
        elif file_name == 'PMMA':
            # --------                       PMMA include: Noisy_RHSI_10 (0.1s)                      -------- #
            args.noise_type = 'Noisy_RHSI_10'
            args.scale = 677
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.1, 0.2, 0.8
        elif file_name == 'Tablet1_64x64_50x':
            # --------          Tablet1_64x64_50x include: Noisy_RHSI (0.001s) / HR_RHSI (1s)        -------- #
            args.scale = 168
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.4, 0.3, 0.8
        elif file_name == 'Tablet1_128x128_10x':
            # --------         Tablet1_128x128_10x include: Noisy_RHSI (0.01s) / HR_RHSI (0.1s)       ------- #
            args.scale = 926.4
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.01, 0.01, 0.8
        elif file_name == 'Tablet1_linescan':
            # --------           Tablet1_linescan include: Noisy_RHSI (0.3s) / HR_RHSI (1s)          -------- #
            args.scale = 1
            args.Epoc_num, args.lambda_R, args.lambda_STV, args.lambda_D = 400, 0.01, 0.2, 1.1  # For 1s
            #args.Epoc_num, args.lambda_R, args.lambda_STV, args.lambda_D = 100, 0.1, 0.3, 1.1 # For 0.3s
        elif file_name == 'Cell_2DScene':
            # --------             Cell_2DScene include: Noisy_RHSI (0.1s) / HR_RHSI (1s)             -------- #
            args.scale = 227
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.1, 2.0, 0.8
        elif file_name == 'Cell_Paramecium' or 'Cell_ParameciumS2':
            # ------------------ Cell_Paramecium include: HR_RHSI (0.1s) / Noisy_RHSI (0.01s) ---------------- #
            args.scale = 100
            args.lambda_R, args.lambda_STV, args.lambda_D = 0.1, 0.3, 0.8

        if os.path.exists('Results/model_weights.pth'):
            os.remove('Results/model_weights.pth')
        main_Raman_De(file_name, args)