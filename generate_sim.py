# Import sim functions
from neurodsp.sim.combined import sim_combined, sim_peak_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim import sim_synaptic_current
from neurodsp.sim import sim_oscillation
from neurodsp.utils import set_random_seed

# Import function to compute power spectra
from neurodsp.spectral import compute_spectrum

# Import utilities for plotting data
from neurodsp.utils import create_times
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series

import os
import numpy as np
import torch
import specparam
from specparam import SpectralGroupModel
from specparam.utils import trim_spectrum
from neurodsp.spectral import compute_spectrum
import time
from random import randrange

def main():
    ###### Hyperparameters ########
    # Set some general settings, to be used across all simulations
    fs = 500
    t_len = 4 # time legnth [sec]
    num_pts = t_len * fs
    
    times = create_times(t_len, fs)
    num_data = 10000 # number of data to generate

    fc_range = np.arange(5, 151) #np.linspace(5, 150, 30) #[5, 40] # range of center frequencies

    # random start index, random integer duration <= len(signal)
    dur_range = [0.01, t_len]
    amp_range = [0.5, 2]

    num_ch = 1

    # data name
    fpath = '/data/'
    exp_num = '4_1_1'
    save_path = fpath + '/'

    fname_nosc = 'nosc_'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'
    fname_osc = 'osc_'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'
    
    data_nosc = np.zeros((num_data, len(times), num_ch), dtype=np.float32)
    data_osc = np.zeros((num_data, len(times), num_ch), dtype=np.float32)

    opt = 0 # 0: sinusoidal ; 1: non-sinusoidal
    rdsym = 0.2

    for i in range(num_data):   
        if i % 100 == 0:
            print('i: ', i)

        set_random_seed(i)
        np.random.seed(i)
        
        num_neurons = np.random.randint(1000, 5000, (1,))[0]
        firing_rate = np.random.randint(2, 10, (1,))[0]
        tau_r = np.random.uniform(0, 0.01, (1,))[0]
        tau_d = np.random.uniform(0.01, 0.05, (1,))[0]
        ap = sim_synaptic_current(t_len, fs, num_neurons, firing_rate, tau_r, tau_d)

        rand_duration = dur_range[1]
        idx_range = [0, num_pts-int(rand_duration*fs)]
        rand_start_idx = idx_range[0]

        amp = np.random.uniform(amp_range[0], amp_range[1], (1,))
        fc_idx = np.random.randint(0, fc_range.shape[0], (1,))  
        fc = fc_range[fc_idx] 

        if opt == 0:
            p = sim_oscillation(rand_duration, fs, fc, cycle='sine')
        elif opt == 1:
            p = sim_oscillation(rand_duration, fs, fc, cycle='asine', rdsym=rdsym)

        p = amp * p
        
        data_nosc[i, :, :] = np.reshape(ap, (len(ap), 1))
        data_osc[i, :, :] = np.reshape(p, (len(p), 1))

    data = data_nosc + data_osc

    t = time.time()

    fg = SpectralGroupModel(peak_width_limits=[1, 8], min_peak_height=0.1, max_n_peaks=6)
    data = data.squeeze(2) # (num_data, num_time)
    R = torch.abs(torch.fft.rfft(torch.tensor(data), dim=-1))
    R = R + 1e-6 # (num_data, num_freqs)
    freqs = torch.fft.rfftfreq(data.shape[-1], 1/fs)

    fg.fit(freqs.numpy(), R.numpy(), [1, fs//2]) 
    params = fg.get_params('aperiodic_params')
    print('Elappsed time to obtain specparam results: {}'.format(time.time() - t))

    fname_nosc_params = 'nosc_params'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'

    np.save(save_path + fname_nosc, data_nosc)
    np.save(save_path + fname_osc, data_osc)
    np.save(save_path + fname_nosc_params, params)


if __name__ == '__main__':
    main()
