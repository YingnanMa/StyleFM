import torch
import torch.fft as fft
import math
import pdb

'''This code is from freeinit => https://github.com/TianxingWu/FreeInit'''
def freq_2d(x, LPF, alpha):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    #x_freq_low = x_freq * (a * LPF + b * HPF)
    #x_freq_high = x_freq * ( (1-a) * LPF + (1-b) * HPF )
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    
    #x_freq_sum = x_freq_low + x_freq_high
    x_freq_sum = x_freq

    # IFFT
    _x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
    x_low = fft.ifftn(_x_freq_low, dim=(-3, -2, -1)).real
    x_low_alpha = fft.ifftn(_x_freq_low*alpha, dim=(-3, -2, -1)).real
    
    _x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
    x_high = fft.ifftn(_x_freq_high, dim=(-3, -2, -1)).real
    x_high_alpha = fft.ifftn(_x_freq_high*alpha, dim=(-3, -2, -1)).real
    
    _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
    
    _x_freq_low_alpha_high = fft.ifftshift(x_freq_low + x_freq_high*alpha, dim=(-3, -2, -1))
    x_low_alpha_high = fft.ifftn(_x_freq_low_alpha_high, dim=(-3, -2, -1)).real
    
    _x_freq_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high, dim=(-3, -2, -1))
    x_high_alpha_low = fft.ifftn(_x_freq_high_alpha_low, dim=(-3, -2, -1)).real

    _x_freq_alpha_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high*alpha, dim=(-3, -2, -1))
    x_alpha_high_alpha_low = fft.ifftn(_x_freq_alpha_high_alpha_low, dim=(-3, -2, -1)).real

    return x_low, x_high, x_sum, x_low_alpha, x_high_alpha, x_low_alpha_high, x_high_alpha_low, x_alpha_high_alpha_low

def freq_1d(x, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3))
    x_freq = fft.fftshift(x_freq, dim=(-3))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    #x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain
    #x_freq_mixed = x_freq_low + x_freq_high

    # IFFT
    x_freq_low = fft.ifftshift(x_freq_low, dim=(-3))
    x_low = fft.ifftn(x_freq_low, dim=(-3)).real
    
    x_freq_high = fft.ifftshift(x_freq_high, dim=(-3))
    x_high = fft.ifftn(x_freq_high, dim=(-3)).real

    return x_low, x_high

def freq_3d(x, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    a = 1.0
    b = 0.1
    #x_freq_low = x_freq * (a * LPF + b * HPF)
    #x_freq_high = x_freq * ( (1-a) * LPF + (1-b) * HPF )
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    
    #x_freq_sum = x_freq_low + x_freq_high
    x_freq_sum = x_freq

    # IFFT
    x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
    x_low = fft.ifftn(x_freq_low, dim=(-3, -2, -1)).real
    
    x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
    x_high = fft.ifftn(x_freq_high, dim=(-3, -2, -1)).real
    
    x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(x_freq_sum, dim=(-3, -2, -1)).real

    return x_low, x_high, x_sum

def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = (noise_freq) * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain
    
    #x_freq_high = x_freq * HPF
    #x_freq_mixed = 1.5*x_freq_low + x_freq_high

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed

def org_freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed


def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian_l":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "gaussian_h":
        return gaussian_high_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "gaussain_lm":
        return gaussian_lowmid_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "gaussain_mh":
        return gaussian_midhigh_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

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

def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
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
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

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
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    #mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0
    mask[..., crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

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