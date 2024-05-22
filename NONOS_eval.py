import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from scipy.optimize import curve_fit
import sklearn.model_selection as ms
import numpy as np
from NONOS import *
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default='mymodel.checkpoint', type=str,
                        help='data directory for saved model')
    parser.add_argument('--fname_osc', default='osc', type=str,
                        help='input data directory for oscillation')
    parser.add_argument('--fname_nosc', default='nosc', type=str,
                        help='input data directory for non-oscillation')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus              #
    os.environ['MASTER_ADDR'] = 'localhost'                 #
    os.environ['MASTER_PORT'] = '12355'                     #
    mp.spawn(train, nprocs=args.gpus, args=(args,))    
  
    #########################################################

def train(gpu, args):
    ############################################################
    rank = gpu	                                               # global rank of the process within all of the processes   
    dist.init_process_group(                                   # Initialize the process and join up with the other processes
    	backend='nccl',                                        # This is 'blocking', meaning that no process will continue
   		init_method='env://',                                  # untill all processes have joined.  
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    ############################################################
    
    ############################################################
    # Data loading
    fs = 500
    t_len = 4
    data_len = fs*t_len

    data_nosc = np.load(args.fname_nosc)
    data_osc = np.load(args.fname_osc)
    data = data_nosc + data_osc

    data = torch.tensor(data).transpose(1,2)
    data_nosc = torch.tensor(data_nosc).transpose(1,2)
    data_osc = torch.tensor(data_osc).transpose(1,2)

    amp_nosc = torch.abs(torch.fft.rfft(data_nosc, dim=-1))
    amp_osc = torch.abs(torch.fft.rfft(data_osc, dim=-1))
    amp = torch.concatenate((amp_osc, amp_nosc), dim=1) # B x 2 x L
    amp = amp.transpose(1, 2)  # B x L x 2

    y = torch.concatenate((data_osc.unsqueeze(1), data_nosc.unsqueeze(1)), dim=1)
    X_train, X_test, Y_train, Y_test, = ms.train_test_split(amp, y,
                                                            test_size=0.2, random_state=100)
    ############################################################

    ############################################################
    ## Hyperparameters for model structure
    input_dim = 1 #data_len//2+1
    inner_dim = 256
    kernel_size = 11
    depth = 5
    num_layers = 4
    target_length = 2048

    ## Hyperparameters for training
    batch_size = 32
    lr = 0.5*1e-5 # 0.5*1e-5
    beta = 20
    rho = 1
    ############################################################

    ############################################################
    # Data sampling
    dataset = list(zip(X_train, Y_train))

    sampler = torch.utils.data.distributed.DistributedSampler( # this makes sure that each process gets a different slice of the training data
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=False, ## should change to false; because we use distributedsampler instead of shuffling
                                        num_workers=0,
                                        pin_memory=True,
                                        sampler=sampler ## should specify sampler
                                        )
    ############################################################

    ############################################################
    # Define a model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    model = NONOS_UNET(input_dim, inner_dim, kernel_size, depth, num_layers) #UNET_1D(input_dims, hidden_layer, kernel_size, depth)
    model.cuda(gpu)

    Optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.5, 0.999),
        eps=1e-08,
    )
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer = Optimizer,
                                            lr_lambda = lambda epoch: 0.95 ** epoch)
    
    ############################################################

    ###############################################################
    # Wrap the model; this reproduces the model onto the GPU for the proesses
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])                                              
    ###############################################################

    model.load_state_dict(torch.load(args.model_file))

    start = datetime.now()
    ###############################################################
    cycle_loss_l1 = torch.nn.L1Loss()

    # Training part
    model.eval()

    if args.mode == 'simple':
        with torch.no_grad():
            result = torch.tensor([]).cuda(gpu)
            dataset = torch.tensor([]).cuda(gpu)
            dataset_x = torch.tensor([]).cuda(gpu)

            if gpu==0:
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")

                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        R = R[:, :, 0] + R[:, :, 1]
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]

                        R = R.cuda(non_blocking=True)

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        result = torch.cat((result, fake_x_ap), dim=0)
                        x = batch[1].type(torch.float32)
                        temp_x = torch.cat((x[:, 0, :], x[:, 1, :]), dim=1).cuda(non_blocking=True)
                        dataset_x = torch.cat((dataset_x, temp_x), dim=0)     
                
            else:
                for batch in loader:
                    R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                    R = R[:, :, 0] + R[:, :, 1]
                    R = torch.log10(R+1e-6)
                    R = R.type(torch.float32)
                    R = R.unsqueeze(1) # B x 1 x L//2+1
                    R = R[:, :, 1::]

                    R = R.cuda(non_blocking=True)

                    x = batch[1].type(torch.float32).cuda(non_blocking=True)
                    x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                    x = symmetric_zero_pad(x, target_length)

                    ######### Forward #########
                    fake_x_ap = unpad(model(x), data_len)
                    R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                    R_ap = R_ap[:, :, 1::]
                    R_ap = torch.log10(R_ap+1e-6)

                    result = torch.cat((result, fake_x_ap), dim=0)
                    x = batch[1].type(torch.float32)
                    temp_x = torch.cat((x[:, 0, :], x[:, 1, :]), dim=1).cuda(non_blocking=True)
                    dataset_x = torch.cat((dataset_x, temp_x), dim=0)     

         
    if args.mode == 'SpecParam':
        for epoch in range(args.epochs):
            if gpu==0:
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")

                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        batch_params = R[:, -1, :] # B x 2
                        R = R[:, :-1, 0] + R[:, :-1, 1]
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]

                        R = R.cuda(non_blocking=True) 

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)  

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        result = torch.cat((result, fake_x_ap), dim=0)
                        x = batch[1].type(torch.float32)
                        temp_x = torch.cat((x[:, 0, :], x[:, 1, :]), dim=1).cuda(non_blocking=True)
                        dataset_x = torch.cat((dataset_x, temp_x), dim=0)     
                
            else:
                for batch in loader:
                    R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                    batch_params = R[:, -1, :] # B x 2
                    R = R[:, :-1, 0] + R[:, :-1, 1]
                    R = torch.log10(R+1e-6)
                    R = R.type(torch.float32)
                    R = R.unsqueeze(1) # B x 1 x L//2+1
                    R = R[:, :, 1::]
                            
                    R = R.cuda(non_blocking=True) 

                    x = batch[1].type(torch.float32).cuda(non_blocking=True)
                    x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                    x = symmetric_zero_pad(x, target_length)  

                    ######### Forward #########
                    fake_x_ap = unpad(model(x), data_len)
                    R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                    R_ap = R_ap[:, :, 1::]
                    R_ap = torch.log10(R_ap+1e-6)

                    result = torch.cat((result, fake_x_ap), dim=0)
                    x = batch[1].type(torch.float32)
                    temp_x = torch.cat((x[:, 0, :], x[:, 1, :]), dim=1).cuda(non_blocking=True)
                    dataset_x = torch.cat((dataset_x, temp_x), dim=0)   
            
        print('Elapsed time: ', datetime.now()-start)

    if gpu==0:
        result = result.cpu().detach().numpy()
        dataset_x = dataset_x.cpu().detach().numpy()
        np.save('result.npy', result)
        np.save('dataset_x.npy', dataset_x)

        err_t = np.zeros((result.shape[0], 1))
        R2_t = np.zeros((result.shape[0], 1))

        err_f = np.zeros((result.shape[0], 1))
        R2_f = np.zeros((result.shape[0], 1))

        for data_idx in range(result.shape[0]):
            fake_x_ap = result[data_idx, 0, :]
            x_ap = dataset_x[data_idx, 1, :]
            x = (dataset_x[data_idx, 0, :] + dataset_x[data_idx, 1, :])

            err_ap = np.abs(fake_x_ap - x_ap)

            err_t[data_idx] = np.mean(err_ap)
            R2_t[data_idx] = calculate_R_squared(x_ap, fake_x_ap)

            fake_R_ap = np.log10(np.abs(np.fft.rfft(fake_x_ap))+1e-6)
            R_ap = np.log10(np.abs(np.fft.rfft(x_ap))+1e-6)
            err_f[data_idx] = np.mean(np.abs(fake_R_ap - R_ap))
            R2_f[data_idx] = calculate_R_squared(R_ap, fake_R_ap)
            
        print('Time error: ', np.mean(err_t))
        print('Time R2: ', np.mean(R2_t))
        print('Freq error: ', np.mean(err_f))
        print('Freq R2: ', np.mean(R2_f))

    ###############################################################
 
if __name__ == '__main__':
    main()