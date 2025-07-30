import torch
import torch.fft as fft
import math
import pdb

'''This code is from freeinit => https://github.com/TianxingWu/FreeInit'''

def gaussian_low_pass_filter(shape, d_s=0.3, d_t=0.3):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask
    
def gaussian_lowmid_pass_filter(shape, d_s=0.7, d_t=0.3):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask
    
def gaussian_midhigh_pass_filter(shape, d_s=0.3, d_t=0.3):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1- math.exp(-1/(2*d_s**2) * d_square)
    return mask
    
def gaussian_high_pass_filter(shape, d_s=0.7, d_t=0.3):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =1- math.exp(-1/(2*d_s**2) * d_square)
    return mask


def process_content_frequency(cnt_latent, d_s_high=0.7, d_t_high=0.3, d_s_midhigh=0.3, d_t_midhigh=0.3, freq_weight=0.3):
    """
    Process content latent with dual frequency filtering using frequency_util functions
    
    The approach:
    1. Use high-pass filter to get only high frequencies
    2. Use mid-high pass filter to get mid+high frequencies  
    3. Combine: high_freq * weight + midhigh_freq
    
    This enhances high frequencies while preserving mid frequencies and removing low frequencies.
    """
    # Get shape and device
    shape = cnt_latent.shape
    device = cnt_latent.device
    dtype = cnt_latent.dtype
    
    # Convert to float64 for FFT operations (as done in freq_2d)
    cnt_latent_f64 = cnt_latent.to(torch.float64)
    
    # Apply 2D FFT
    cnt_freq = fft.fftn(cnt_latent_f64, dim=(-3, -2, -1))
    cnt_freq_shifted = fft.fftshift(cnt_freq, dim=(-3, -2, -1))
    
    # Create filters directly (as done in the original code)
    # High-pass filter (only high frequencies)
    high_pass_filter = gaussian_high_pass_filter(shape, d_s=d_s_high, d_t=d_t_high).to(device)
    
    # Mid-high pass filter (mid + high frequencies) 
    midhigh_pass_filter = gaussian_midhigh_pass_filter(shape, d_s=d_s_midhigh, d_t=d_t_midhigh).to(device)

    cnt_freq_high = cnt_freq_shifted * high_pass_filter
    
    cnt_freq_midhigh = cnt_freq_shifted * midhigh_pass_filter
    
    #nofilt test
    #cnt_freq_midhigh = cnt_freq_shifted 

    cnt_enhance = cnt_freq_high * freq_weight  + cnt_freq_midhigh

    cnt_enhance_shifter_back = fft.ifftshift(cnt_enhance, dim=(-3, -2, -1))
    cnt_latent_enhance = fft.ifftn(cnt_enhance_shifter_back, dim=(-3, -2, -1)).real

    cnt_latent_enhance = cnt_latent_enhance.to(dtype)
    
    return cnt_latent_enhance

def process_style_frequency(sty_latent, d_s_lowmid=0.7, d_t_lowmid=0.3, freq_weight=0.3):
    """
    Process style latent with low-mid pass filtering
    
    The approach:
    1. Use low-mid pass filter to get low+mid frequencies
    2. Combine: lowmid_freq * weight + original
    
    This enhances low and mid frequencies while preserving high frequencies.
    """
    # Get shape and device
    shape = sty_latent.shape
    device = sty_latent.device
    dtype = sty_latent.dtype
    
    # Convert to float64 for FFT operations
    sty_latent_f64 = sty_latent.to(torch.float64)
    
    # Apply 2D FFT
    sty_freq = fft.fftn(sty_latent_f64, dim=(-3, -2, -1))
    sty_freq_shifted = fft.fftshift(sty_freq, dim=(-3, -2, -1))
    
    # Create low-mid pass filter (keeps low and mid frequencies, removes high)
    lowmid_pass_filter = gaussian_lowmid_pass_filter(shape, d_s=d_s_lowmid, d_t=d_t_lowmid).to(device)
    
    # Apply filter in frequency domain
    sty_freq_lowmid = sty_freq_shifted * lowmid_pass_filter

    sty_freq_enhance = sty_freq_lowmid * freq_weight  + sty_freq_shifted
    
    sty_freq_enhance_shifted_back = fft.ifftshift(sty_freq_enhance, dim=(-3, -2, -1))
    sty_latent_enhance = fft.ifftn(sty_freq_enhance_shifted_back, dim=(-3, -2, -1)).real

    sty_latent_enhance = sty_latent_enhance.to(dtype)
    
    return sty_latent_enhance
