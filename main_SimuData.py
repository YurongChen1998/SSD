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
import matplotlib.pyplot as plt
from model import MSD_model, ADMM_Iter
import metric as final_metric
import warnings 

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main_Raman_De(data_name, args):
    #----------------------- Data Configuration -----------------------#
    dataset_dir = '../Raman_Denoising_Dataset/Simu_Data/'
    result_dir = './Results/SimuScene/' + data_name + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    data = sio.loadmat(dataset_dir + data_name + '.mat')
    GT_data_tensor = torch.from_numpy(data['HR_RHSI']).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)
    GT_data_tensor = (GT_data_tensor - torch.min(GT_data_tensor)) / (torch.max(GT_data_tensor) - torch.min(GT_data_tensor))

    meas_data = torch.from_numpy(data['Noisy_RHSI_20']).float().to(device).unsqueeze(0)
    meas_data_tensor = meas_data.permute(0, 3, 1, 2)/args.scale
    meas_data_tensor = (meas_data_tensor - torch.min(meas_data_tensor)) / (torch.max(meas_data_tensor) - torch.min(meas_data_tensor))

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
        args.Epoc_num, args.lambda_R, args.lambda_STV = 1400, 0.1, 0.4
        recon = MSD_model(GT_data_tensor.to(device), meas_data_tensor.to(device), meas_data_tensor.to(device), args)
        recon = recon.squeeze().permute(1, 2, 0).cpu().numpy()

        print('Our results:')
        GT_data = GT_data_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        final_metric.evaluate_all_results(GT_data, recon)
        sio.savemat('{}/SSD_Solo.mat'.format(result_dir), {'R_hsi':recon})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset',      default = 'Simu',         help="Choose test dataset: Simu/CMa")
    parser.add_argument('--PnP_i',        default = 1,              help="State of solo or PnP model")
    parser.add_argument('--vis_plot',     default = 0,              help="State of plot spectrum")
    parser.add_argument('--iter_num',     default = 30,             help="Number of ADMM iterations")
    parser.add_argument('--Epoc_num',     default = 1200,           help="Number of MSD iterations")
    parser.add_argument('--scale',        default = 1.0,            help="Factor of scaling mesurements")
    parser.add_argument('--rho',          default = 20.,            help="Factor of dual variable")
    parser.add_argument('--lambda_S',     default = 1,              help="Factor of sparse noise")
    parser.add_argument('--lambda_R',     default = 4.0,            help="Factor of TV/STV regularization")
    parser.add_argument('--lambda_STV',   default = 0.4,            help="Ratio of STV regularization")
    parser.add_argument('--lambda_D',     default = 0.8,            help="Decay factor of lambda_R")
    args = parser.parse_args()

    if args.Dataset   == 'Simu':
        data_list     =  ['Pattern_data'] #'Pattern_data', 'Gaussian_data', 'Chessboard_data', 'PS_ball_data'
    else:
        print("---------- Ensure the name of dataset ----------") 

    for file_name in data_list:
        if file_name == 'PS_ball_data':
            args.lambda_S, args.iter_num, args.lambda_R, args.lambda_STV, args.lambda_D = 1000, 30, 0.05, 0.2, 0.8
        elif file_name == 'Pattern_data':
            args.iter_num, args.lambda_R, args.lambda_STV, args.lambda_D = 40, 6.0, 0.4, 0.8
        elif file_name == 'Gaussian_data':
            args.iter_num, args.lambda_R, args.lambda_STV, args.lambda_D = 30, 8.0, 0.2, 0.8
        elif file_name == 'Chessboard_data':
            args.iter_num, args.lambda_R, args.lambda_STV, args.lambda_D = 30, 5.0, 0.4, 0.8
        else:
            args.lambda_R, args.Epoc_num = parser.get_default('lambda_R'), parser.get_default('Epoc_num')
        
        if os.path.exists('Results/model_weights.pth'):
            os.remove('Results/model_weights.pth')
        main_Raman_De(file_name, args)