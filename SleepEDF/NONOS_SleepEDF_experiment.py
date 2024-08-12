import os
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn.utils.weight_norm as weight_norm

# from distribution.dilated_conv import DilatedConvEncoder
# from distribution.losses import hierarchical_contrastive_loss
# from distribution.encoder import AP_Encoder_v3
from tqdm import tqdm
from scipy.optimize import curve_fit

import sklearn.model_selection as ms
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, stride, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=kernel_size, 
                               stride=stride, dilation=dilation, padding=kernel_size//2, bias=True)
        self.bn = nn.BatchNorm1d(out_dims)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class SEBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_dims//8, out_dims, kernel_size=1, padding=0)
        self.activation = nn.GELU()

    def forward(self, x):
        x_se = nn.functional.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = self.activation(x_se)
        x_se = self.conv2(x_se)
        x_se = self.activation(x_se)
        x_out = torch.add(x, x_se)
        return x_out
    
class REBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, dilation):
        super().__init__()
        self.ConvBlock1 = ConvBlock(in_dims, out_dims, kernel_size, 1, dilation)
        self.ConvBlock2 = ConvBlock(out_dims, out_dims, kernel_size, 1, dilation)
        self.SEBlock = SEBlock(out_dims, out_dims)


    def forward(self, x):
        x_re = self.ConvBlock1(x)
        x_re = self.ConvBlock2(x_re)
        x_re = self.SEBlock(x_re)
        x_out = torch.add(x, x_re)
        return x_out
    
class UNET_1D(nn.Module):
    def __init__(self, input_dim, inner_dim, kernel_size, depth, num_layers):
        super().__init__()

        self.avg_pool1d = nn.ModuleList([nn.AvgPool1d(input_dim, stride=4 ** i) for i in range(1, num_layers-1)])
        
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        self.down_layers.append(self.down_layer(input_dim, inner_dim, kernel_size, 1, depth))
        for i in range(1, num_layers):
            in_channels = inner_dim if i == 1 else (inner_dim * i + input_dim)
            out_channels = inner_dim * (i + 1)
            self.down_layers.append(self.down_layer(in_channels, out_channels, kernel_size, 4, depth))

        for i in range(num_layers, 1, -1):
            self.up_layers.append(ConvBlock(inner_dim * (i + (i-1)), inner_dim * (i - 1), kernel_size, 1, 1))

        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

        self.outconv = nn.Conv1d(inner_dim, 1, kernel_size=1, stride=1, padding=0)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(ConvBlock(input_layer, out_layer, kernel, stride, 1))
        for _ in range(depth):
            block.append(REBlock(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x):
        pool_xs = [avg_pool(x) for avg_pool in self.avg_pool1d]
        
        outs = []
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x) if i <= 1 else down_layer(torch.cat([x, pool_xs[i-2]], 1))
            outs.append(x) if i < len(self.down_layers)-1 else None

        for i, up_layer in enumerate(self.up_layers):
            up = self.upsample(x) if i == 0 else self.upsample(up)
            up = torch.cat([up, outs[-(i + 1)]], 1)
            up = up_layer(up)

        out = self.outconv(up)
        return out

def symmetric_zero_pad(input_tensor, target_length):
    """
    Apply symmetric zero-padding to a 3D tensor along the last dimension.
    
    Parameters:
    - input_tensor: A tensor of shape [batch_size, 1, L].
    - target_length: The target length after padding.
    
    Returns:
    - A tensor of shape [batch_size, 1, target_length].
    """
    # Current length of the sequences
    current_length = input_tensor.shape[2]
    
    # Calculate the total amount of padding needed
    total_padding = max(target_length - current_length, 0)
    
    # Calculate padding to be added on each side
    padding_left = total_padding // 2
    padding_right = total_padding - padding_left
    
    # Apply symmetric padding
    padded_tensor = torch.nn.functional.pad(input_tensor, (padding_left, padding_right), "constant", 0)
    
    return padded_tensor

def unpad(padded_tensor, original_length):
    """
    Remove symmetric padding from a 3D tensor along the last dimension.
    
    Parameters:
    - padded_tensor: A tensor that has been padded to a target length.
    - original_length: The original length of the sequences before padding.
    
    Returns:
    - A tensor with padding removed, of shape [batch_size, 1, original_length].
    """
    # Current length of the sequences (after padding)
    current_length = padded_tensor.shape[2]
    
    # Calculate the total amount of padding added
    total_padding = current_length - original_length
    
    # Calculate the padding that was added on each side
    padding_left = total_padding // 2
    
    # If there's no padding, return the tensor as is
    if total_padding <= 0:
        return padded_tensor
    
    # Remove symmetric padding
    unpadded_tensor = padded_tensor[:, :, padding_left:padding_left+original_length]
    
    return unpadded_tensor

def get_freqs(L, fs):
    freqs = np.fft.rfftfreq(L, d=1./fs)
    return freqs[1::] # exclude DC component

def get_curve_fit(freqs, offset=None, knee=None, exponent=None):
    fits = offset - torch.log10(freqs**(exponent))
    return fits

def ap_fit_torch(freqs, offset, exponent):
    fits = offset - torch.log10(freqs**(exponent))
    return fits

def ap_fit(freqs, offset, exponent):
    fits = offset - np.log10(freqs**(exponent))
    return fits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--mode', default=0, type=int, 
                        metavar='N',
                        help='0: training | 1: encoding | 2: decoding')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus              #
    os.environ['MASTER_ADDR'] = 'localhost'                 #
    os.environ['MASTER_PORT'] = '12355'                     #
    mp.spawn(train, nprocs=args.gpus, args=(args,))    
  
    #########################################################

def cleanup():
    dist.destroy_process_group()

def train(gpu, args):
    ############################################################
    rank = gpu                                                  # global rank of the process within all of the processes   
    dist.init_process_group(                                   # Initialize the process and join up with the other processes
       backend='nccl',                                        # This is 'blocking', meaning that no process will continue
         init_method='env://',                                  # untill all processes have joined.  
       world_size=args.world_size,                              
       rank=rank                                               
    )                                                          
    ############################################################
    
    ############################################################
    # Data loading
    fs = 100
    t_len = 20
    data_len = int(fs*t_len)

    fpath = '/home/Documents/Data/'
    exp_num = '9'
    
    train_data = np.load(fpath + 'Experiment' + exp_num + '/bigBW/' + 'X_train20sec_1_50Hz_bigBW.npy')
    train_labels = np.load(fpath + 'Experiment' + exp_num + '/bigBW/' + 'Y_train20sec_1_50Hz_bigBW.npy')
    train_ap_params = np.load(fpath + 'Experiment' + exp_num + '/bigBW/' + 'X_train_ap_params20sec_1_50Hz_bigBW.npy')
    train_data = torch.tensor(train_data, dtype=torch.float32) # B x L
    train_data = train_data.unsqueeze(1) # B x 1 x L

    train_amp = torch.abs(torch.fft.rfft(train_data, dim=-1)) # B x 1 x L//2+1

    train_labels = torch.tensor(train_labels, dtype=torch.float32) # B x 1
    train_labels = train_labels.unsqueeze(1) # B x 1 x 1
    train_data = torch.cat((train_data, train_labels), dim=-1) # B x 1 x L+1

    train_ap_params = torch.tensor(train_ap_params) # B x 2
    train_ap_params = train_ap_params.unsqueeze(1) # B x 1 x 2
    train_amp = torch.concatenate((train_amp, train_ap_params), dim=-1) # B x 1 x L//2+1+2

    test_data = np.load(fpath + 'Experiment' + exp_num + '/bigBW/' + 'X_test20sec_1_50Hz_bigBW.npy')
    test_labels = np.load(fpath + 'Experiment' + exp_num + '/bigBW/' + 'Y_test20sec_1_50Hz_bigBW.npy')
    test_ap_params = np.load(fpath + 'Experiment' + exp_num + '/bigBW/' + 'X_test_ap_params20sec_1_50Hz_bigBW.npy')
    test_data = torch.tensor(test_data, dtype=torch.float32) # B x L
    test_data = test_data.unsqueeze(1) # B x 1 x L
    
    test_amp = torch.abs(torch.fft.rfft(test_data, dim=-1)) # B x 1 x L//2+1

    test_labels = torch.tensor(test_labels, dtype=torch.float32) # B x 1
    test_labels = test_labels.unsqueeze(1) # B x 1 x 1
    test_data = torch.cat((test_data, test_labels), dim=-1) # B x 1 x L+1

    test_ap_params = torch.tensor(test_ap_params) # B x 2
    test_ap_params = test_ap_params.unsqueeze(1) # B x 1 x 2
    test_amp = torch.concatenate((test_amp, test_ap_params), dim=-1) # B x 1 x L//2+1+2

    # split
    # y = labels #torch.cat((labels, params), dim=1)
    # X_train, X_test, Y_train, Y_test, = ms.train_test_split(amp, data,
    #                                                     test_size=0.2, random_state=100)

    X_train, X_valid, Y_train, Y_valid = ms.train_test_split(train_amp, train_data,
                                                            test_size=0.2, random_state=100)

    Y_test_data_idx = torch.tensor(np.load(fpath + 'Experiment' + exp_num + '/bigBW/' + 'X_test_data_idx.npy'))
    Y_test_data_idx = Y_test_data_idx.unsqueeze(1).unsqueeze(1) # B x 1 x 1

    X_test = test_amp
    Y_test = test_data

    Y_test = torch.cat((Y_test, Y_test_data_idx), dim=-1)
    ############################################################

    ############################################################
    ## Hyperparameters for model structure
    input_dim = 1 #data_len//2+1
    inner_dim = 256
    kernel_size = 13
    depth = 5
    num_layers = 4 # >=4
    target_length = 2048
    beta = 0.05
    rho = 0.05
    freq_range = [1, 50]

    ## Hyperparameters for training
    batch_size = 32
    lr = 0.5*1e-5 # 0.5*1e-5

    # ## Initialize the early stopping variables
    best_val_loss = float('inf')
    # patience = 10  # Number of epochs to wait for improvement before stopping
    # epochs_no_improve = 0
    ############################################################

    ############################################################
    # Model saved location
    model_checkpoint_path = fpath
    if args.mode == 0:
        current_datetime = datetime.now()
        fname = current_datetime.strftime("%Y%m%d%H%M%S")
    elif (args.mode == 1) or (args.mode == 2):
        fname = '20240802000122' # '20240801144820'

    model_checkpoint_path = fpath + "/Experiment" + exp_num + "/"
    model_checkpoint_G_theta = model_checkpoint_path + fname + '_G_theta.checkpoint'

    ############################################################

    ############################################################
    # Data sampling
    if args.mode == 0:
        dataset = list(zip(X_train, Y_train))
        dataset_valid = list(zip(X_valid, Y_valid))
    elif args.mode == 1:
        dataset = list(zip(X_test, Y_test))
    
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
    if args.mode == 0:
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

    if args.mode == 0:
        G_theta = UNET_1D(input_dim, inner_dim, kernel_size, depth, num_layers) #UNET_1D(input_dims, hidden_layer, kernel_size, depth)
        G_theta.cuda(gpu)

        G_Optimizer = optim.AdamW(
            G_theta.parameters(),
            lr=lr,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = G_Optimizer, T_0=10)

        # scheduler = optim.lr_scheduler.LambdaLR(optimizer = G_Optimizer,
        #                                        lr_lambda = lambda epoch: 0.95 ** epoch)
    
    elif (args.mode == 1) or (args.mode==2):
        G_theta = UNET_1D(input_dim, inner_dim, kernel_size, depth, num_layers) #UNET_1D(input_dims, hidden_layer, kernel_size, depth)
        G_theta.cuda(gpu)

    ############################################################

    ###############################################################
    # Wrap the model; this reproduces the model onto the GPU for the proesses
    G_theta = nn.parallel.DistributedDataParallel(G_theta,
                                                device_ids=[gpu])                                              
    ###############################################################
    
    ###############################################################
    # Load the trained model
    if (args.mode == 1):
        G_theta.load_state_dict(torch.load(model_checkpoint_G_theta))
        
    # total_step = len(loader)
    ###############################################################

    start = datetime.now()
    ###############################################################
    cycle_loss_l1 = torch.nn.L1Loss()

    # Training part
    if args.mode == 0:
        for epoch in range(args.epochs):
            G_theta.train()

            if gpu==0:
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")
                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        batch_params = R[:, :, -2:] # B x 1 x 2
                        R = R[:, :, :-2] # B x 1 x L//2+1
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        # R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]
                                    
                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freq_indices = torch.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                        freqs = freqs[freq_indices]
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x f

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            temp_fit = ap_fit(freqs[i,0,:], batch_params[i, :, 0], batch_params[i, :, 1])
                            fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True) 

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)                        
                        labels = x[:, :, -1]
                        x = x[:, 0, :-1]  # B x L
                        x = x.unsqueeze(1)
                        x = symmetric_zero_pad(x, target_length)
                        x = x.cuda(non_blocking=True)
                        # freqs_ = get_freqs(data_len, fs)
                        # freqs_ = torch.tensor(freqs_, dtype=torch.float32)

                        ######### Forward #########
                        fake_x_ap = unpad(G_theta(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)
                        # R_ap_high_freq = R_ap[:, :, torch.where(freqs_ > 80)[0]]
                        R_ap = R_ap[:, :, freq_indices]

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        # R_high_freq = R[:, :, torch.where(freqs_ > 80)[0]]

                        ###### G_theta ######
                        ## Calculate gradients and update parameters
                        G_Optimizer.zero_grad()

                        loss_t = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f = cycle_loss_l1(fits, R_ap)
                        # loss_f_remained = cycle_loss_l1(R_high_freq, R_ap_high_freq)
                        loss_pen = torch.mean(torch.max(R_res, dim=-1)[0])

                        loss_G = loss_f + beta*loss_t + rho*loss_pen #+ beta*loss_f_remained
                        loss_G.backward()
                        G_Optimizer.step()
                
            else:
                for batch in loader:
                    R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                    batch_params = R[:, :, -2:] # B x 1 x 2
                    R = R[:, :, :-2] # B x 1 x L//2+1
                    R = torch.log10(R+1e-6)
                    R = R.type(torch.float32)
                    # R = R.unsqueeze(1) # B x 1 x L//2+1
                    R = R[:, :, 1::]
                                
                    # Get initial curve fitting
                    freqs = get_freqs(data_len, fs)
                    freqs = torch.tensor(freqs, dtype=torch.float32)
                    freq_indices = torch.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                    freqs = freqs[freq_indices]
                    freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x f

                    fits = torch.tensor([])
                    for i in range(R.shape[0]):
                        temp_fit = ap_fit(freqs[i,0,:], batch_params[i, :, 0], batch_params[i, :, 1])
                        fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                    fits = fits.unsqueeze(1)
                    fits = fits.cuda(non_blocking=True)

                    R = R.cuda(non_blocking=True) 

                    x = batch[1].type(torch.float32).cuda(non_blocking=True)
                    labels = x[:, :, -1]
                    x = x[:, 0, :-1]  # B x L
                    x = x.unsqueeze(1)
                    x = symmetric_zero_pad(x, target_length)
                    x = x.cuda(non_blocking=True)

                    # freqs_ = get_freqs(data_len, fs)
                    # freqs_ = torch.tensor(freqs_, dtype=torch.float32)

                    ######### Forward #########
                    fake_x_ap = unpad(G_theta(x), data_len)
                    R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                    R_ap = R_ap[:, :, 1::]
                    R_ap = torch.log10(R_ap+1e-6)
                    # R_ap_high_freq = R_ap[:, :, torch.where(freqs_ > 80)[0]]
                    R_ap = R_ap[:, :, freq_indices]

                    R_res = R_ap - fits
                    R_res.cuda(non_blocking=True)
                    
                    # R_high_freq = R[:, :, torch.where(freqs_ > 80)[0]]

                    ###### G_theta ######
                    ## Calculate gradients and update parameters
                    G_Optimizer.zero_grad()

                    loss_t = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                    loss_f = cycle_loss_l1(fits, R_ap)
                    # loss_f_remained = cycle_loss_l1(R_high_freq, R_ap_high_freq)
                    loss_pen = torch.mean(torch.max(R_res, dim=-1)[0])

                    loss_G = loss_f + beta*loss_t + rho*loss_pen #+ beta*loss_f_remained

                    loss_G.backward()
                    G_Optimizer.step()
                

            if gpu==0:
                val_loss = 0.0
                with torch.no_grad():
                    G_theta.eval()
                    for batch in loader_valid:
                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        batch_params = R[:, :, -2:] # B x 1 x 2
                        R = R[:, :, :-2] # B x 1 x L//2+1
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        # R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]
                                    
                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freq_indices = torch.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                        freqs = freqs[freq_indices]
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x f

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            temp_fit = ap_fit(freqs[i,0,:], batch_params[i, :, 0], batch_params[i, :, 1])
                            fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True) 

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        labels = x[:, :, -1]
                        x = x[:, 0, :-1]  # B x L
                        x = x.unsqueeze(1)
                        x = symmetric_zero_pad(x, target_length)
                        x = x.cuda(non_blocking=True)

                        # freqs_ = get_freqs(data_len, fs)
                        # freqs_ = torch.tensor(freqs_, dtype=torch.float32)

                        ######### Forward #########
                        fake_x_ap = unpad(G_theta(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)
                        # R_ap_high_freq = R_ap[:, :, torch.where(freqs_ > 80)[0]]
                        R_ap = R_ap[:, :, freq_indices]

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        # R_high_freq = R[:, :, torch.where(freqs_ > 80)[0]]

                        ###### G_theta ######
                        ## Calculate gradients and update parameters
                        G_Optimizer.zero_grad()

                        loss_t_val = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f_val = cycle_loss_l1(fits, R_ap)
                        # loss_f_remained = cycle_loss_l1(R_high_freq, R_ap_high_freq)
                        loss_pen_val = torch.mean(torch.max(R_res, dim=-1)[0])

                        loss_G_val = loss_f_val + beta*loss_t_val + rho*loss_pen_val #+ beta*loss_f_remained
                        val_loss += loss_G_val.item()
                val_loss = val_loss / len(loader_valid)
            
                # # # Check if the validation loss has improved
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     # Save the model
                #     torch.save(G_theta.state_dict(), model_checkpoint_G_theta)
                
                
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
                    G_theta.eval()
                    for batch in loader_valid:
                        R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                        batch_params = R[:, :, -2:] # B x 1 x 2
                        R = R[:, :, :-2] # B x 1 x L//2+1
                        R = torch.log10(R+1e-6)
                        R = R.type(torch.float32)
                        # R = R.unsqueeze(1) # B x 1 x L//2+1
                        R = R[:, :, 1::]
                                    
                        # Get initial curve fitting
                        freqs = get_freqs(data_len, fs)
                        freqs = torch.tensor(freqs, dtype=torch.float32)
                        freq_indices = torch.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                        freqs = freqs[freq_indices]
                        freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x f

                        fits = torch.tensor([])
                        for i in range(R.shape[0]):
                            temp_fit = ap_fit(freqs[i,0,:], batch_params[i, :, 0], batch_params[i, :, 1])
                            fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                        fits = fits.unsqueeze(1)
                        fits = fits.cuda(non_blocking=True)

                        R = R.cuda(non_blocking=True) 

                        x = batch[1].type(torch.float32).cuda(non_blocking=True)
                        labels = x[:, :, -1]
                        x = x[:, 0, :-1]  # B x L
                        x = x.unsqueeze(1)
                        x = symmetric_zero_pad(x, target_length)
                        x = x.cuda(non_blocking=True)

                        # freqs_ = get_freqs(data_len, fs)
                        # freqs_ = torch.tensor(freqs_, dtype=torch.float32)

                        ######### Forward #########
                        fake_x_ap = unpad(G_theta(x), data_len)
                        R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                        R_ap = R_ap[:, :, 1::]
                        R_ap = torch.log10(R_ap+1e-6)
                        # R_ap_high_freq = R_ap[:, :, torch.where(freqs_ > 80)[0]]
                        R_ap = R_ap[:, :, freq_indices]

                        R_res = R_ap - fits
                        R_res.cuda(non_blocking=True)
                        
                        # R_high_freq = R[:, :, torch.where(freqs_ > 80)[0]]

                        ###### G_theta ######
                        ## Calculate gradients and update parameters
                        G_Optimizer.zero_grad()

                        loss_t_val = cycle_loss_l1(unpad(x, data_len), fake_x_ap)
                        loss_f_val = cycle_loss_l1(fits, R_ap)
                        # loss_f_remained = cycle_loss_l1(R_high_freq, R_ap_high_freq)
                        loss_pen_val = torch.mean(torch.max(R_res, dim=-1)[0])

                        loss_G = loss_f_val + beta*loss_t_val + rho*loss_pen_val #+ beta*loss_f_remained
                    
            # # If the validation loss has not improved for 'patience' epochs, stop training
            # if epochs_no_improve == patience:
            #     print('Early stopping!')
            scheduler.step()
        torch.save(G_theta.state_dict(), model_checkpoint_G_theta)
  
    ###############################################################

    ###############################################################
    # Test: encoding + decoding
    elif args.mode == 1:
        start = datetime.now()
        total_step = len(loader)
        with torch.no_grad():
            result = torch.tensor([]).cuda(gpu)
            dataset_x = torch.tensor([]).cuda(gpu)
            dataset_y = torch.tensor([]).cuda(gpu)
            result_cf = torch.tensor([]).cuda(gpu)
            dataset_idx = torch.tensor([]).cuda(gpu)

            for batch in loader:
                R = batch[0].type(torch.float32) # amplitude (radius) in frequency domain
                batch_params = R[:, :, -2:] # B x 1 x 2
                R = R[:, :, :-2] # B x 1 x L//2+1
                R = torch.log10(R+1e-6)
                R = R.type(torch.float32)
                # R = R.unsqueeze(1) # B x 1 x L//2+1
                R = R[:, :, 1::]
                            
                # Get initial curve fitting
                freqs = get_freqs(data_len, fs)
                freqs = torch.tensor(freqs, dtype=torch.float32)
                freq_indices = torch.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                freqs = freqs[freq_indices]
                freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x f

                fits = torch.tensor([])
                for i in range(R.shape[0]):
                    temp_fit = ap_fit(freqs[i,0,:], batch_params[i, :, 0], batch_params[i, :, 1])
                    fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                fits = fits.unsqueeze(1)
                fits = fits.cuda(non_blocking=True)

                R = R.cuda(non_blocking=True) 

                x = batch[1].type(torch.float32).cuda(non_blocking=True)
                data_idx = x[:, :, -1]
                x = x[:, :, :-1]
                labels = x[:, :, -1]
                x = x[:, 0, :-1]  # B x L
                x = x.unsqueeze(1)
                x = symmetric_zero_pad(x, target_length)
                x = x.cuda(non_blocking=True)

                ######### Forward #########
                fake_x_ap = unpad(G_theta(x), data_len)

                result = torch.cat((result, fake_x_ap), dim=0)
                result_cf = torch.cat((result_cf, fits), dim=0)
                dataset_x = torch.cat((dataset_x, x), dim=0)
                dataset_y = torch.cat((dataset_y, labels), dim=0)
                dataset_idx = torch.cat((dataset_idx, data_idx), dim=0)
        if gpu == 0:
            print("Test complete in: " + str(datetime.now() - start))
            np.save(model_checkpoint_path + '/Experiment' + exp_num + '_result.npy', result.cpu().numpy())
            np.save(model_checkpoint_path + '/Experiment' + exp_num + '_dataset_x.npy', dataset_x.cpu().numpy())
            np.save(model_checkpoint_path + '/Experiment' + exp_num + '_dataset_y.npy', dataset_y.cpu().numpy())

        np.save(model_checkpoint_path + '/Experiment' + exp_num + '_result_gpu' + str(gpu) + '.npy', result.cpu().numpy())
        np.save(model_checkpoint_path + '/Experiment' + exp_num + '_dataset_x_gpu' + str(gpu) + '.npy', dataset_x.cpu().numpy())
        np.save(model_checkpoint_path + '/Experiment' + exp_num + '_dataset_y_gpu' + str(gpu) + '.npy', dataset_y.cpu().numpy())
        np.save(model_checkpoint_path + '/Experiment' + exp_num + '_dataset_idx_gpu' + str(gpu) + '.npy', dataset_idx.cpu().numpy())


    ###############################################################
 
if __name__ == '__main__':
    main()