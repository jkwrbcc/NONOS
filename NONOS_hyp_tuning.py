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

# modules for hyperparameter tuning
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='/home/', type=str)
    parser.add_argument('--exp_num', default='1_1', type=str)
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use')
    parser.add_argument('-n', '--num_samples', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    args = parser.parse_args()
    
    config = {
        "inner_dim": tune.choice([2**i for i in range(5, 9)]),
        "kernel_size": tune.choice([3, 5, 7, 9, 11, 21, 31]),
        "depth": tune.choice([3, 4, 5]),
        "num_layers": tune.choice([3, 4, 5]),
        "beta": tune.choice([0.5, 0.1, 0.05, 0.01, 0.005]), #tune.choice([1, 5, 10, 20]),
        "rho": tune.choice([0.5, 0.1, 0.05, 0.01, 0.005]), #tune.choice([1, 5, 10, 20]),
    }

    optim_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10, #args.epochs,
        grace_period=1,
        reduction_factor=2,
    )

    data_dir = args.file_path
    exp_num = args.exp_num
    load_data(data_dir, exp_num)

    result = tune.run(
        partial(train_nonos, data_dir=data_dir, args=args),
        resources_per_trial={"cpu": 8, "gpu": args.gpus},
        config=config,
        num_samples=args.num_samples,
        scheduler=optim_scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

def load_data(data_dir='/home/', exp_num='1_1'):
    fpath = data_dir
    fs = 500
    t_len = 4
    exp_num = exp_num
    fname_ap = 'aperiodic_'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'
    fname_p = 'periodic_'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'
    fname_ap_param = 'aperiodic_params'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'

    data_ap = np.load(fpath + 'Experiment' + exp_num + '/' + fname_ap)
    data_p = np.load(fpath + 'Experiment' + exp_num + '/' + fname_p)
    data = data_ap + data_p

    data = torch.tensor(data).transpose(1,2)
    data_ap = torch.tensor(data_ap).transpose(1,2)
    data_p = torch.tensor(data_p).transpose(1,2)

    amp_ap = torch.abs(torch.fft.rfft(data_ap, dim=-1))
    amp_p = torch.abs(torch.fft.rfft(data_p, dim=-1))
    amp = torch.concatenate((amp_p, amp_ap), dim=1) # B x 2 x L
    amp = amp.transpose(1, 2)  # B x L x 2

    ap_params = np.load(fpath + 'Experiment' + exp_num + '/' + fname_ap_param)
    ap_params = torch.tensor(ap_params)
    amp = torch.concatenate((amp, ap_params.unsqueeze(1)), dim=1)

    # split
    y = torch.concatenate((data_p.unsqueeze(1), data_ap.unsqueeze(1)), dim=1)
    X_train, X_test, Y_train, Y_test, = ms.train_test_split(amp,y,
                                                        test_size=0.2, random_state=100)

    X_train, X_valid, Y_train, Y_valid = ms.train_test_split(X_train,Y_train,
                                                            test_size=0.2, random_state=100)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def train_nonos(gpu, args, data_dir='/home/):
    ############################################################
    # Data loading
    fs = 500
    t_len = 4
    data_len = fs*t_len

    fpath = data_dir
    
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_data(fpath)

    ############################################################

    ############################################################
    ## Hyperparameters for model structure
    input_dim = 1 #data_len//2+1
    inner_dim = config['inner_dim'] #256
    kernel_size = config['kernel_size'] #11
    depth = config['depth'] #5
    num_layers = config['num_layers'] #4 # >=4
    target_length = 2048
    beta = config['beta'] #20
    rho = config['rho'] #1

    ## Hyperparameters for training
    batch_size = 32
    lr = 0.5*1e-5 # 0.5*1e-5
    num_epochs = 100
    ############################################################

    ############################################################
    # Data sampling
    dataset = list(zip(X_train, Y_train))
    dataset_valid = list(zip(X_valid, Y_valid))
    
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=True, ## should change to false; because we use distributedsampler instead of shuffling
                                        num_workers=0,
                                        pin_memory=True,
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

    model = NONOS_UNET(input_dim, inner_dim, kernel_size, depth, num_layers) #UNET_1D(input_dims, hidden_layer, kernel_size, depth)
    model.cuda()

    Optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.5, 0.999),
        eps=1e-08,
    )
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer = Optimizer,
                                            lr_lambda = lambda epoch: 0.95 ** epoch)
    ############################################################

    ############################################################
    # Model saved location
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            print(data_path)
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            Optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    ############################################################

    # ###############################################################
    # # Wrap the model; this reproduces the model onto the GPU for the proesses
    # model = nn.parallel.DistributedDataParallel(model,
    #                                             device_ids=[gpu])                                              
    # ###############################################################

    start = datetime.now()
    ###############################################################
    cycle_loss_l1 = torch.nn.L1Loss()

    # Training part
    model.train()

    if args.mode == 'simple':
        for epoch in range(args.epochs):
            model.train()

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
              
            val_loss = 0.0
            acc = 0.0
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
                    x_ap = x[:, 1, :]
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

                    acc += torch.mean(torch.abs(x_ap.cpu() - fake_x_ap.cpu())).item()

            val_loss = val_loss / len(loader_valid)
            acc      = acc / len(loader_valid)
              
            checkpoint_data = {
                        "epoch": epoch,
                        "net_state_dict": G_theta.state_dict(),
                        "optimizer_state_dict": G_Optimizer.state_dict(),
                    }

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": val_loss, "accuracy": 1/acc},
                    checkpoint=checkpoint,
                )

            print('[Epoch {}/{}] [loss_G]: {:.6f} [loss_t]: {:.6f} [loss_f]: {:.6f} | [valid loss]: {:.6f}'.format(
                epoch+1, args.epochs,
                loss_G.item(),
                loss_t.item(),
                loss_f.item(),
                val_loss
                )
            )

    if args.mode == 'SpecParam':
        for epoch in range(args.epochs):
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
                x_ap = x[:, 1, :]
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
                
            val_loss = 0.0
            acc = 0.0
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
                    acc += torch.mean(torch.abs(x_ap.cpu() - fake_x_ap.cpu())).item()

            val_loss = val_loss / len(loader_valid)
            acc      = acc / len(loader_valid)

            checkpoint_data = {
                        "epoch": epoch,
                        "net_state_dict": G_theta.state_dict(),
                        "optimizer_state_dict": G_Optimizer.state_dict(),
                    }

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": val_loss, "accuracy": 1/acc},
                    checkpoint=checkpoint,
                )

            print('[Epoch {}/{}] [loss_G]: {:.6f} [loss_t]: {:.6f} [loss_f]: {:.6f} | [valid loss]: {:.6f}'.format(
                epoch+1, args.epochs,
                loss_G.item(),
                loss_t.item(),
                loss_f.item(),
                val_loss
                )
            )
           
            scheduler.step()
    ###############################################################
 
if __name__ == '__main__':
    main()