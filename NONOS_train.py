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
    parser.add_argument('--fname_osc', default='osc', type=str,
                        help='input data directory for oscillation')
    parser.add_argument('--fname_nosc', default='nosc', type=str,
                        help='input data directory for non-oscillation')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use')
    parser.add_argument('--mode', default='simple', type=str,
                        help='mode of the curve fitting method')
    parser.add_argument('--specpara-results', default='results', type=str,
                        help='input data directory for precalculated SpecParam results')
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

    fpath = '/home/bcc/Documents/Data/NONOS_v4/'
    exp_num = '1_1'

    data_nosc = np.load(args.fname_nosc)
    data_osc = np.load(args.fname_osc)
    data = data_nosc + data_osc

    data = torch.tensor(data).transpose(1,2)
    data_nosc = torch.tensor(data_nosc).transpose(1,2)
    data_osc = torch.tensor(data_osc).transpose(1,2)

    amp_nosc = torch.abs(torch.fft.rfft(amp_nosc, dim=-1))
    amp_osc = torch.abs(torch.fft.rfft(data_osc, dim=-1))
    amp = torch.concatenate((amp_osc, amp_nosc), dim=1) # B x 2 x L
    amp = amp.transpose(1, 2)  # B x L x 2

    if args.mode == 'SpecParam':
        ap_params = np.load(args.specpara_results)
        ap_params = torch.tensor(ap_params)
        amp = torch.concatenate((amp, ap_params.unsqueeze(1)), dim=1)
    
    y = torch.concatenate((data_osc.unsqueeze(1), data_nosc.unsqueeze(1)), dim=1)
    X_train, X_test, Y_train, Y_test, = ms.train_test_split(amp, y,
                                                            test_size=0.2, random_state=100)

    X_train, X_valid, Y_train, Y_valid = ms.train_test_split(X_train, Y_train,
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
    # Model saved location
    default_path = 'training/'
    current_datetime = datetime.now()
    save_name = current_datetime.strftime("%Y%m%d%H%M%S")
    model_checkpoint_name = default_path + save_name + "_" + args.mode + ".checkpoint"

    ############################################################

    ############################################################
    # Data sampling
    dataset = list(zip(X_train, Y_train))
    dataset_valid = list(zip(X_valid, Y_valid))
    
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
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True)
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

    start = datetime.now()
    ###############################################################
    cycle_loss_l1 = torch.nn.L1Loss()

    # Training part
    model.train()

    if args.mode == 'simple':
        for epoch in range(args.epochs):
            model.train()

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

                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            params, _ = curve_fit(ap_fit, freqs[i,0,:].numpy(), R[i,0,:].numpy())
                            fits = torch.cat((fits, ap_fit(freqs[i,0,:], params[0], params[1]).unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True)

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)

                        Optimizer.zero_grad()

                        loss_t = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f = cycle_loss_l1(fits, R_ap)
                        loss_pen = torch.mean(torch.max(R_res, dim=-1)[0])

                        loss_G = beta*loss_f + loss_t + rho*loss_pen
                        loss_G.backward()
                        Optimizer.step()
                
            else:
                for batch in loader:
                    R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                    R = R[:, :, 0] + R[:, :, 1]
                    R = torch.log10(R+1e-6)
                    R = R.type(torch.float32)
                    R = R.unsqueeze(1) # B x 1 x L//2+1
                    R = R[:, :, 1::]

                    # Get initial curve fitting
                    freqs = get_freqs(data_len, fs)
                    freqs = torch.tensor(freqs, dtype=torch.float32)
                    freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                    fits = torch.tensor([])
                    for i in range(R.shape[0]):
                        params, _ = curve_fit(ap_fit, freqs[i,0,:].numpy(), R[i,0,:].numpy())
                        fits = torch.cat((fits, ap_fit(freqs[i,0,:], params[0], params[1]).unsqueeze(0)), dim=0)
                    fits = fits.unsqueeze(1)
                    fits = fits.cuda(non_blocking=True)

                    R = R.cuda(non_blocking=True)

                    x = batch[1].type(torch.float32).cuda(non_blocking=True)
                    x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                    x = symmetric_zero_pad(x, target_length)

                    ######### Forward #########
                    fake_x_ap = unpad(model(x), data_len)
                    R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                    R_ap = R_ap[:, :, 1::]
                    R_ap = torch.log10(R_ap+1e-6)

                    R_res = R_ap - fits
                    R_res.cuda(non_blocking=True)
                    
                    Optimizer.zero_grad()

                    loss_t = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                    loss_f = cycle_loss_l1(fits, R_ap)
                    loss_pen = torch.mean(torch.max(R_res, dim=-1)[0])

                    loss_G = beta*loss_f + loss_t + rho*loss_pen
                    loss_G.backward()
                    Optimizer.step()

            if gpu==0:
                val_loss = 0.0
                with torch.no_grad():
                    model.eval()
                    for batch in loader_valid:
                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        R = R[:, :, 0] + R[:, :, 1]
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]

                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            params, _ = curve_fit(ap_fit, freqs[i,0,:].numpy(), R[i,0,:].numpy())
                            fits = torch.cat((fits, ap_fit(freqs[i,0,:], params[0], params[1]).unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True)

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        loss_t_val = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f_val = cycle_loss_l1(fits, R_ap)
                        loss_pen_val = torch.mean(torch.max(R_res, dim=-1)[0])

                        loss_G_val = beta*loss_f_val + loss_t_val + rho*loss_pen_val

                        val_loss += loss_G_val.item()
                val_loss = val_loss / len(loader_valid)
                            
                print('[Epoch {}/{}] [loss_G]: {:.6f} [loss_t]: {:.6f} [loss_f]: {:.6f} | [valid loss]: {:.6f}'.format(
                    epoch+1, args.epochs,
                    loss_G.item(),
                    loss_t.item(),
                    loss_f.item(),
                    val_loss
                    )
                )
            else:
                with torch.no_grad():
                    model.eval()
                    for batch in loader_valid:
                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        R = R[:, :, 0] + R[:, :, 1]
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]

                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            params, _ = curve_fit(ap_fit, freqs[i,0,:].numpy(), R[i,0,:].numpy())
                            fits = torch.cat((fits, ap_fit(freqs[i,0,:], params[0], params[1]).unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True)

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        model.zero_grad()

                        loss_t_val = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f_val = cycle_loss_l1(fits, R_ap)
                        loss_pen_val = torch.mean(torch.max(R_res, dim=-1)[0])

                        loss_G_val = beta*loss_f_val + loss_t_val + rho*loss_pen_val

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
                                
                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            temp_fit = ap_fit(freqs[i,0,:], batch_params[i, 0], batch_params[i, 1])
                            fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True) 

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)  

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        ## Calculate gradients and update parameters
                        Optimizer.zero_grad()

                        loss_t = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f = cycle_loss_l1(fits, R_ap)
                        loss_pen = torch.mean(torch.max(R_res, dim=-1)[0])

                        loss_G = beta*loss_f + loss_t + rho*loss_pen

                        loss_G.backward()
                        Optimizer.step()
                
            else:
                for batch in loader:
                    R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                    batch_params = R[:, -1, :] # B x 2
                    R = R[:, :-1, 0] + R[:, :-1, 1]
                    R = torch.log10(R+1e-6)
                    R = R.type(torch.float32)
                    R = R.unsqueeze(1) # B x 1 x L//2+1
                    R = R[:, :, 1::]
                            
                    # Get initial curve fitting
                    freqs = get_freqs(data_len, fs)
                    freqs = torch.tensor(freqs, dtype=torch.float32)
                    freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                    fits = torch.tensor([])
                    for i in range(R.shape[0]):
                        temp_fit = ap_fit(freqs[i,0,:], batch_params[i, 0], batch_params[i, 1])
                        fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                    fits = fits.unsqueeze(1)
                    fits = fits.cuda(non_blocking=True)

                    R = R.cuda(non_blocking=True) 

                    x = batch[1].type(torch.float32).cuda(non_blocking=True)
                    x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                    x = symmetric_zero_pad(x, target_length)  

                    ######### Forward #########
                    fake_x_ap = unpad(model(x), data_len)
                    R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                    R_ap = R_ap[:, :, 1::]
                    R_ap = torch.log10(R_ap+1e-6)

                    R_res = R_ap - fits
                    R_res.cuda(non_blocking=True)
                    
                    ## Calculate gradients and update parameters
                    model.zero_grad()

                    loss_t = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                    loss_f = cycle_loss_l1(fits, R_ap)
                    loss_pen = torch.mean(torch.max(R_res, dim=-1)[0])

                    loss_G = beta*loss_f + loss_t + rho*loss_pen

                    loss_G.backward()
                    model.step()
            
            if gpu==0:
                val_loss = 0.0
                with torch.no_grad():
                    model.eval()
                    for batch in loader_valid:
                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        batch_params = R[:, -1, :] # B x 2
                        R = R[:, :-1, 0] + R[:, :-1, 1]
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]
                                
                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            temp_fit = ap_fit(freqs[i,0,:], batch_params[i, 0], batch_params[i, 1])
                            fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True) 

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)  

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        loss_t_val = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f_val = cycle_loss_l1(fits, R_ap)
                        loss_pen_val = torch.mean(torch.max(R_res, dim=-1)[0])
                        loss_G_val = beta*loss_f_val + loss_t_val + rho*loss_pen_val
                        val_loss += loss_G_val.item()
                val_loss = val_loss / len(loader_valid)
                print('[Epoch {}/{}] [loss_G]: {:.6f} [loss_t]: {:.6f} [loss_f]: {:.6f} | [valid loss]: {:.6f}'.format(
                    epoch+1, args.epochs,
                    loss_G.item(),
                    loss_t.item(),
                    loss_f.item(),
                    val_loss
                    )
                )
            else:
                with torch.no_grad():
                    model.eval()
                    for batch in loader_valid:
                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        batch_params = R[:, -1, :] # B x 2
                        R = R[:, :-1, 0] + R[:, :-1, 1]
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]
                                
                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            temp_fit = ap_fit(freqs[i,0,:], batch_params[i, 0], batch_params[i, 1])
                            fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True) 

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        x = x[:, 0, :] + x[:, 1, :] # B x 1 x L
                        x = symmetric_zero_pad(x, target_length)  

                        ######### Forward #########
                        fake_x_ap = unpad(model(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        loss_t_val = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f_val = cycle_loss_l1(fits, R_ap)
                        loss_pen_val = torch.mean(torch.max(R_res, dim=-1)[0])
                        loss_G_val = beta*loss_f_val + loss_t_val + rho*loss_pen_val
                        val_loss += loss_G_val.item()
                val_loss = val_loss / len(loader_valid)
                
            scheduler.step()
        print('Elapsed time: ', datetime.now()-start)

    if gpu == 0:
        torch.save(model.state_dict(), model_checkpoint_name)
    ###############################################################
 
if __name__ == '__main__':
    main()
