import argparse
from pyedflib import highlevel
import numpy as np
import glob
import torch
import specparam
from specparam import SpectralGroupModel
from specparam.utils import trim_spectrum
from neurodsp.spectral import compute_spectrum
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file_path', default='/home/', type=str,
                        help='Path to the sleep-edf dataset')
    parser.add_argument('-n', '--num_sampling', default=5, type=int,
                        help='Number of random sampling data')
    args = parser.parse_args()

    path = args.file_path
    data_list = glob.glob(path+'/**')

    trains = [x for x in data_list if x.endswith('PSG.edf')]
    labels = [x for x in data_list if x.endswith('Hypnogram.edf')]

    num_data = len(trains)
    selected_num_data = args.num_sampling

    np.random.seed(0)
    selected_data_indices = np.random.choice(num_data, selected_num_data, replace=False)

    fs = 100 # Hz
    duration = 20 # sec
    num_pts = duration * fs

    total_segments = torch.tensor([])
    total_data_indices1 = torch.tensor([])
    total_data_indices2 = torch.tensor([])

    for data_idx in selected_data_indices:
        psg_data, psg_headers, psg_header = highlevel.read_edf(trains[data_idx])
        hypnogram_data, hypnogram_headers, hypnogram_header = highlevel.read_edf(labels[data_idx])

        # Find the index of the EEG Fpz-Cz channel
        index = next((i for i, header in enumerate(psg_headers) if header['label'] == 'EEG Fpz-Cz'), None)

        data = psg_data[index]
        annot = hypnogram_header['annotations']
        annot = np.array(annot)
        annot_float = annot[:, 0:2]
        annot_float = annot_float.astype(float)
        start_time = annot_float[:, 0]
        end_time = start_time + annot_float[:, 1]

        splited = np.char.split(annot[:, 2], 'Sleep stage ')
        labels_ = np.array([])
        for i in range(len(splited)):
            if len(splited[i]) > 1: # == 1: movement time
                labels_ = np.append(labels_, splited[i][1])

        # W -> others
        change_indices1 = np.where(np.logical_and(labels_[:-1] == 'W', labels_[1:] != 'W'))[0]

        # others -> W
        change_indices2 = np.where(np.logical_and(labels_[:-1] != 'W', labels_[1:] == 'W'))[0]

        transition_start_indices = []
        transition_end_indices = []
        # Data split where W -> others which has total 20 sec duration
        end_indices_list = end_time[change_indices1]
        reject_idx = np.where(end_indices_list + num_pts//2 > len(data))[0]
        end_indices_list[reject_idx] = []
        reject_idx = np.where(end_indices_list - num_pts//2 < 0)[0]
        end_indices_list[reject_idx] = []
        data_indices1 = np.zeros((len(end_indices_list), num_pts))
        for i in range(len(end_indices_list)):
            start_idx = int(end_indices_list[i]) - num_pts//2
            end_idx = int(end_indices_list[i]) + num_pts//2
            data_indices1[i, :] = data[start_idx:end_idx]

            transition_start_indices.append(start_idx)
            transition_end_indices.append(end_idx)

        # Data split where others -> W which has total 20 sec duration
        end_indices_list = end_time[change_indices2]
        reject_idx = np.where(end_indices_list + num_pts//2 > len(data))[0]
        end_indices_list[reject_idx] = []
        reject_idx = np.where(end_indices_list - num_pts//2 < 0)[0]
        end_indices_list[reject_idx] = []
        data_indices2 = np.zeros((len(end_indices_list), num_pts))
        for i in range(len(end_indices_list)):
            start_idx = int(end_indices_list[i]) - num_pts//2
            end_idx = int(end_indices_list[i]) + num_pts//2
            data_indices2[i, :] = data[start_idx:end_idx]

            transition_start_indices.append(start_idx)
            transition_end_indices.append(end_idx)

        # Split data into 20 sec
        segment_length = num_pts
        num_segments = len(data) // segment_length
        segments = np.zeros((num_segments, segment_length))
        reject_idx = []

        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            
            # Check if segment overlaps with any transition segment
            if not any(t_start < end and t_end > start for t_start, t_end in zip(transition_start_indices, transition_end_indices)):
                segments[i, :] = data[start:end]
            
            else:
                reject_idx.append(i)

        # Remove rejected segments
        segments = np.delete(segments, reject_idx, axis=0)

        # Append to total data
        total_segments = torch.cat((total_segments, torch.tensor(segments)))
        total_data_indices1 = torch.cat((total_data_indices1, torch.tensor(data_indices1)))
        total_data_indices2 = torch.cat((total_data_indices2, torch.tensor(data_indices2)))

    X_train = total_segments.numpy()
    Y_train = np.zeros((X_train.shape[0], 1))
    X_test = np.concatenate((total_data_indices1.numpy(), total_data_indices2.numpy()), axis=0)
    temp_labels1 = np.ones((data_indices1.shape[0], 1))
    temp_labels2 = np.ones((data_indices2.shape[0], 1)) * 2
    Y_test = np.concatenate((temp_labels1, temp_labels2), axis=0)
    
    # z-score normalization
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / np.std(X_train, axis=1, keepdims=True)
    X_test = (X_test - np.mean(X_test, axis=1, keepdims=True)) / np.std(X_test, axis=1, keepdims=True)

    # save data
    np.save(path + '/X_train.npy', X_train)
    np.save(path + '/Y_train.npy', Y_train)
    np.save(path + '/X_test.npy', X_test)
    np.save(path + '/Y_test.npy', Y_test)

    # Perform SpecParam algorithm for X_train
    t = time.time()
    fg = SpectralGroupModel(peak_width_limits=[2, 12], min_peak_height=0.2, max_n_peaks=6)
    R = torch.abs(torch.fft.rfft(torch.tensor(X_train), dim=-1))
    R = R + 1e-6 # (num_data, num_freqs)
    freqs = torch.fft.rfftfreq(X_train.shape[-1], 1/fs)
    freqs = torch.round(freqs*100)/100
    fg.fit(freqs.numpy(), R.numpy(), [1, fs//2]) 
    params = fg.get_params('aperiodic_params')
    print('Elappsed time to obtain specparam results: {}'.format(time.time() - t))
    np.save(path + '/X_train_ap_params.npy', params)

    # Perform SpecParam algorithm for X_test
    t = time.time()
    fg = SpectralGroupModel(peak_width_limits=[2, 12], min_peak_height=0.2, max_n_peaks=6)
    R = torch.abs(torch.fft.rfft(torch.tensor(X_test), dim=-1))
    R = R + 1e-6 # (num_data, num_freqs)
    freqs = torch.fft.rfftfreq(X_test.shape[-1], 1/fs)
    freqs = torch.round(freqs*100)/100
    fg.fit(freqs.numpy(), R.numpy(), [1, fs//2]) 
    params = fg.get_params('aperiodic_params')
    print('Elappsed time to obtain specparam results: {}'.format(time.time() - t))
    np.save(path + '/X_test_ap_params.npy', params)

if __name__ == '__main__':
    main()
