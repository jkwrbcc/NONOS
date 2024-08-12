clc
clear
close all

addpath('/home/Documents/GitHub/npy-matlab/npy-matlab') % for converting to Phy

%%
fpath = "/home/Documents/Data/";

fs = 100;

x_org = readNPY(strcat(fpath, 'x_org.npy'));

x_bp = single(zeros(52, 1, 2000));

cutoff_freq = [1 4];
order = 3300;
bpFilt = fir1(order, cutoff_freq./(fs/2)); % bandpass filteringw using hamming window
D = mean(grpdelay(bpFilt));

for i = 1:52
    x_filt = filter(bpFilt, 1, [squeeze(x_org(i, :, :)); zeros(D, 1)]); % Append D zeros to the input data
    x_filt = x_filt(D+1:end);                  % Shift data to compensate for delay
    x_bp(i, :, :) = x_filt;
end

writeNPY(x_bp, strcat(fpath, 'x_bp_from1Hz.npy'));