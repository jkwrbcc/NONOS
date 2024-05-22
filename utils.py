
import torch
import numpy as np

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

def calculate_R_squared(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    SS_tot = np.sum((y_true - y_true_mean) ** 2)
    SS_res = np.sum((y_true - y_pred) ** 2)
    return 1 - SS_res / SS_tot

