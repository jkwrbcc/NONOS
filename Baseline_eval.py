import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import argparse
import statsmodels
from statsmodels.tsa.seasonal import STL
from scipy.signal import butter, filtfilt

def eval_NONOS(dataset, result, data_idx):
    x_ap_hat = result[data_idx, 0, :]

    return x_ap_hat

def eval_STL(dataset, result, data_idx):
    x_p = dataset_x[data_idx, 0, :]
    x_ap = dataset_x[data_idx, 1, :]
    x = (dataset_x[data_idx, 0, :] + dataset_x[data_idx, 1, :])
    stl = STL(x, period=2, robust=True)
    res_robust = stl.fit()
    x_ap_hat = res_robust.trend

    return x_ap_hat

def eval_MA(dataset, result, data_idx):
    x_p = dataset_x[data_idx, 0, :]
    x_ap = dataset_x[data_idx, 1, :]
    x = (dataset_x[data_idx, 0, :] + dataset_x[data_idx, 1, :])
    fs = 500
    filter_width = 1/10*fs
    conv_filter = np.ones((int(filter_width),))/filter_width
    x_ap_hat = np.convolve(x, conv_filter, mode='same')

    return x_ap_hat

def eval_LPF(dataset, result, data_idx):
    x_p = dataset_x[data_idx, 0, :]
    x_ap = dataset_x[data_idx, 1, :]
    x = (dataset_x[data_idx, 0, :] + dataset_x[data_idx, 1, :])
    fs = 500
    cutoff_freq = [0.5, 10]
    x_ap_hat = filtfilt(b, a, x)

    return x_ap_hat

def calculated_R_squared(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    SS_tot = np.sum((y_true - y_true_mean) ** 2)
    SS_res = np.sum((y_true - y_pred) ** 2)
    return 1 - SS_res / SS_tot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file_path', default='/home/', type=str,
                         help='Path to the result files')
    parser.add_argument('-e', '--exp_num', default='1_1', type=str,
                        help='Experiment number')
    args = parser.parse_args()

    fpath = args.path

    # Load the data
    result = np.load(fpath + '/' + exp_num + '/' + 'Experiment' + exp_num + '_result.npy')
    dataset_x = np.load(fpath + '/' + exp_num + '/' + 'Experiment' + exp_num + '_dataset_x.npy')

    total_err = {}
    total_R2 = {}

    compare_list = ['NONOS', 'STL', 'MA', 'LPF']

    for compare in compare_list:
        total_err[compare] = []
        total_R2[compare] = []

        err_t = np.zeros((result.shape[0], 1))
        R2_t = np.zeros((result.shape[0], 1))
        err_f = np.zeros((result.shape[0], 1))
        R2_f = np.zeros((result.shape[0], 1))

        for data_idx in range(result.shape[0]):
            if compare == 'NONOS':
                x_ap_hat = eval_NONOS(dataset, result, data_idx)
            elif compare == 'STL':
                x_ap_hat = eval_STL(dataset, result, data_idx)
            elif compare == 'MA':
                x_ap_hat = eval_MA(dataset, result, data_idx)
            elif compare == 'LPF':
                x_ap_hat = eval_LPF(dataset, result, data_idx)

            err_ap = np.abs(x_ap_hat - x_ap)
            err_t[data_idx] = np.mean(err_ap)
            R2_t[data_idx] = calculated_R_squared(x_ap, x_ap_hat)

            R_ap_hat = np.log10(np.abs(np.fft.rfft(x_ap_hat))+1e-6)
            R_ap = np.log10(np.abs(np.fft.rfft(x_ap))+1e-6)
            err_f[data_idx] = np.mean(np.abs(R_ap_hat - R_ap))
            R2_f[data_idx] = calculated_R_squared(R_ap, R_ap_hat)

        total_err[compare]['time'] = np.mean(err_t)
        total_R2[compare]['time'] = np.mean(R2_t)
        total_err[compare]['freq'] = np.mean(err_f)
        total_R2[compare]['freq'] = np.mean(R2_f)

    print('Time domain error')
    print(total_err)
    print('Time domain R2')
    print(total_R2)
    print('Frequency domain error')
    print(total_err)
    print('Frequency domain R2')
    print(total_R2)

if __name__ == '__main__':
    main()
