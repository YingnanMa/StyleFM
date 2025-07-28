import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torch.fft as fft
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

# Import frequency utilities
from util.frequency_util import gaussian_high_pass_filter, gaussian_midhigh_pass_filter, gaussian_lowmid_pass_filter,process_content_frequency,process_style_frequency

feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)


def extract_q_features_from_latent(model, z_latent, timestep, self_attn_indices, uc):
    """Extract Q features from a given latent at specific timestep
    
    Returns Q features for each attention layer
    """
    device = z_latent.device
    t_tensor = torch.full((z_latent.shape[0],), timestep, device=device, dtype=torch.long)
    
    # Dictionary to save Q features
    q_features = {}
    
    # Define hook function to capture Q features
    def make_hook(layer_name):
        def hook(module, input, output):
            # Access the q attribute from the first transformer block's self-attention
            if hasattr(module, 'transformer_blocks') and len(module.transformer_blocks) > 0:
                attn = module.transformer_blocks[0].attn1
                if hasattr(attn, 'q') and attn.q is not None:
                    q_features[layer_name] = attn.q.detach()
        return hook
    
    # Register hooks on attention layers
    hooks = []
    for idx, block in enumerate(model.model.diffusion_model.output_blocks):
        if idx in self_attn_indices and len(block) > 1:
            if "SpatialTransformer" in str(type(block[1])):
                hook = block[1].register_forward_hook(
                    make_hook(f'output_block_{idx}_self_attn_q')
                )
                hooks.append(hook)
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model.apply_model(z_latent, t_tensor, uc)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return q_features


def extract_k_features_from_latent(model, z_latent, timestep, self_attn_indices, uc):
    """Extract K features from a given latent at specific timestep
    
    Returns K features for each attention layer
    """
    device = z_latent.device
    t_tensor = torch.full((z_latent.shape[0],), timestep, device=device, dtype=torch.long)
    
    # Dictionary to save K features
    k_features = {}
    
    # Define hook function to capture K features
    def make_hook(layer_name):
        def hook(module, input, output):
            # Access the k attribute from the first transformer block's self-attention
            if hasattr(module, 'transformer_blocks') and len(module.transformer_blocks) > 0:
                attn = module.transformer_blocks[0].attn1
                if hasattr(attn, 'k') and attn.k is not None:
                    k_features[layer_name] = attn.k.detach()
        return hook
    
    # Register hooks on attention layers
    hooks = []
    for idx, block in enumerate(model.model.diffusion_model.output_blocks):
        if idx in self_attn_indices and len(block) > 1:
            if "SpatialTransformer" in str(type(block[1])):
                hook = block[1].register_forward_hook(
                    make_hook(f'output_block_{idx}_self_attn_k')
                )
                hooks.append(hook)
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model.apply_model(z_latent, t_tensor, uc)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return k_features
    
def extract_v_features_from_latent(model, z_latent, timestep, self_attn_indices, uc):
    """Extract v features from a given latent at specific timestep
    
    Returns v features for each attention layer
    """
    device = z_latent.device
    t_tensor = torch.full((z_latent.shape[0],), timestep, device=device, dtype=torch.long)
    
    # Dictionary to save v features
    v_features = {}
    
    # Define hook function to capture v features
    def make_hook(layer_name):
        def hook(module, input, output):
            # Access the v attribute from the first transformer block's self-attention
            if hasattr(module, 'transformer_blocks') and len(module.transformer_blocks) > 0:
                attn = module.transformer_blocks[0].attn1
                if hasattr(attn, 'v') and attn.v is not None:
                    v_features[layer_name] = attn.v.detach()
        return hook
    
    # Register hooks on attention layers
    hooks = []
    for idx, block in enumerate(model.model.diffusion_model.output_blocks):
        if idx in self_attn_indices and len(block) > 1:
            if "SpatialTransformer" in str(type(block[1])):
                hook = block[1].register_forward_hook(
                    make_hook(f'output_block_{idx}_self_attn_v')
                )
                hooks.append(hook)
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model.apply_model(z_latent, t_tensor, uc)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return v_features

def feat_merge(opt, cnt_feats, sty_feats, start_step=0, q_prime_weight=0.25, k_prime_weight=0.25,
               cnt_z_enc=None, sty_z_enc=None, model=None, uc=None, time_range=None, self_attn_indices=None):
    
    feat_maps = [{'config': {
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]
    
    first_processed_idx = 0
    
    if first_processed_idx is None:
        return feat_maps
    
    q_dict = extract_q_features_from_latent(
        model,
        cnt_z_enc,
        time_range[first_processed_idx],
        self_attn_indices,
        uc
    )
    k_dict = extract_k_features_from_latent(
        model,
        sty_z_enc,
        time_range[first_processed_idx],
        self_attn_indices,
        uc
    )
    v_dict = extract_v_features_from_latent(
        model,
        sty_z_enc,
        time_range[first_processed_idx],
        self_attn_indices,
        uc
    )
    
    processed_count = 0
    for i in range(50):
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()
        processed_count += 1
        
        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                if i == first_processed_idx:  
                    if ori_key in q_dict:
                        mixed_q = cnt_feat[ori_key] + q_prime_weight * q_dict[ori_key]
                        feat_maps[i][ori_key] = mixed_q
                        del q_dict[ori_key]
                    else:
                        feat_maps[i][ori_key] = cnt_feat[ori_key]
                else:  
                    if feat_maps[i-1].get(ori_key) is not None:  
                        prev_mixed_q = feat_maps[i-1][ori_key]
                        curr_original_q = cnt_feats[i][ori_key]
                        mixed_q = curr_original_q + q_prime_weight * prev_mixed_q
                        feat_maps[i][ori_key] = mixed_q
                    else:
                        feat_maps[i][ori_key] = cnt_feat[ori_key]
                        
            elif ori_key[-1] == 'k':
                if i == first_processed_idx: 
                    if ori_key in k_dict:
                        mixed_k = sty_feat[ori_key] + k_prime_weight * k_dict[ori_key]
                        feat_maps[i][ori_key] = mixed_k
                        del k_dict[ori_key]
                    else:
                        feat_maps[i][ori_key] = sty_feat[ori_key]
                else:  
                    if feat_maps[i-1].get(ori_key) is not None:
                        prev_mixed_k = feat_maps[i-1][ori_key]
                        curr_original_k = sty_feats[i][ori_key]
                        mixed_k = curr_original_k + k_prime_weight * prev_mixed_k
                        feat_maps[i][ori_key] = mixed_k
                    else:
                        feat_maps[i][ori_key] = sty_feat[ori_key]
                        
            elif ori_key[-1] == 'v':
                if i == first_processed_idx:  
                    if ori_key in v_dict:
                        mixed_v = sty_feat[ori_key] + k_prime_weight * v_dict[ori_key]
                        feat_maps[i][ori_key] = mixed_v
                        del v_dict[ori_key]
                    else:
                        feat_maps[i][ori_key] = sty_feat[ori_key]
                else:  
                    if feat_maps[i-1].get(ori_key) is not None:
                        prev_mixed_v = feat_maps[i-1][ori_key]
                        curr_original_v = sty_feats[i][ori_key]
                        mixed_v = curr_original_v + k_prime_weight * prev_mixed_v
                        feat_maps[i][ori_key] = mixed_v
                    else:
                        feat_maps[i][ori_key] = sty_feat[ori_key]
    
    return feat_maps
    
def load_img(path):
    if os.path.isdir(path):
        print(f"Skipping directory: {path}")
        return None
        
    image = Image.open(path).convert("RGB")
    h = w = 512
    image = image.resize((w, h))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty', default = './data/sty')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.0, help='attention temperature scaling hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output', 
                       help='output path for results')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    
    # Q' and K' recursive mixing weights
    parser.add_argument('--q_prime_weight', type=float, default=0.1, 
                       help='Weight for recursive Q\' mixing (default: 0.25)')
    parser.add_argument('--k_prime_weight', type=float, default=0.1, 
                       help='Weight for recursive K\' mixing (default: 0.25)')
    
    # Content frequency filtering parameters
    parser.add_argument('--cnt_d_s_high', type=float, default=0.7, 
                       help='Spatial parameter for content high-pass filter ')
    parser.add_argument('--cnt_d_t_high', type=float, default=0.3, 
                       help='Temporal parameter for content high-pass filter ')
    parser.add_argument('--cnt_d_s_midhigh', type=float, default=0.5, 
                       help='Spatial parameter for content mid-high pass filter ')
    parser.add_argument('--cnt_d_t_midhigh', type=float, default=0.3, 
                       help='Temporal parameter for content mid-high pass filter ')
    parser.add_argument('--cnt_freq_weight', type=float, default=0.5, 
                       help='Weight for content high frequency component')
    
    # Style frequency filtering parameters
    parser.add_argument('--sty_d_s_lowmid', type=float, default=0.7, 
                       help='Spatial parameter for style low-mid pass filter ')
    parser.add_argument('--sty_d_t_lowmid', type=float, default=0.3, 
                       help='Temporal parameter for style low-mid pass filter ')
    parser.add_argument('--sty_freq_weight', type=float, default=0.3, 
                       help='Weight for style low-mid frequency component')
    
    opt = parser.parse_args()

    seed_everything(22)
    
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    
    # Print configuration
    print("\nStyleID - Frequency Processing + Recursive Q'/K' Mixing")
    print(f"Output path: {output_path}\n")
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    
    # Get sorted lists of images
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))
    
    # Filter only image files
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    sty_img_list = [f for f in sty_img_list if os.path.splitext(f.lower())[1] in img_extensions]
    cnt_img_list = [f for f in cnt_img_list if os.path.splitext(f.lower())[1] in img_extensions]
    
    print(f"Found {len(cnt_img_list)} content images and {len(sty_img_list)} style images")
    print(f"Will process {len(cnt_img_list) * len(sty_img_list)} combinations\n")
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()

    begin = time.time()
    
    total_combinations = len(cnt_img_list) * len(sty_img_list)
    combination_idx = 0
    
    # Process each content image with all style images
    for cnt_idx, cnt_name in enumerate(cnt_img_list):
        print(f"Processing content image {cnt_idx+1}/{len(cnt_img_list)}: {cnt_name}")
        
        # Load and process content image (only once per content)
        cnt_path = os.path.join(opt.cnt, cnt_name)
        init_cnt = load_img(cnt_path).to(device)
        
        # Reset feature maps for content
        feat_maps = [{'config': {'T':opt.T}} for _ in range(50)]
        
        # Content DDIM inversion
        init_cnt_encoded = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
        cnt_z_enc_raw, _ = sampler.encode_ddim(
            init_cnt_encoded.clone(), 
            num_steps=ddim_inversion_steps, 
            unconditional_conditioning=uc,
            end_step=time_idx_dict[ddim_inversion_steps-1-start_step],
            callback_ddim_timesteps=save_feature_timesteps,
            img_callback=ddim_sampler_callback
        )
        cnt_feat = copy.deepcopy(feat_maps)
        cnt_z_enc_raw = feat_maps[0]['z_enc']
        
        # Apply frequency processing to content latent
        cnt_z_enc = process_content_frequency(
            cnt_z_enc_raw,
            d_s_high=opt.cnt_d_s_high,
            d_t_high=opt.cnt_d_t_high,
            d_s_midhigh=opt.cnt_d_s_midhigh,
            d_t_midhigh=opt.cnt_d_t_midhigh,
            freq_weight=opt.cnt_freq_weight
        )
        
        # Update the z_enc in content features
        cnt_feat[0]['z_enc'] = cnt_z_enc
        
        # Process with each style image
        for sty_idx, sty_name in enumerate(sty_img_list):
            # Clean GPU memory before processing new style
            if sty_idx > 0:
                torch.cuda.empty_cache()
            combination_idx += 1
            
            # Load and process style image
            sty_path = os.path.join(opt.sty, sty_name)
            init_sty = load_img(sty_path).to(device)
            
            # Reset feature maps for style
            feat_maps = [{'config': {'T':opt.T}} for _ in range(50)]
            
            # Style DDIM inversion
            init_sty_encoded = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc_raw, _ = sampler.encode_ddim(
                init_sty_encoded.clone(), 
                num_steps=ddim_inversion_steps, 
                unconditional_conditioning=uc,
                end_step=time_idx_dict[ddim_inversion_steps-1-start_step],
                callback_ddim_timesteps=save_feature_timesteps,
                img_callback=ddim_sampler_callback
            )
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc_raw = feat_maps[0]['z_enc']
            
            # Apply frequency processing to style latent
            sty_z_enc = process_style_frequency(
                sty_z_enc_raw,
                d_s_lowmid=opt.sty_d_s_lowmid,
                d_t_lowmid=opt.sty_d_t_lowmid,
                freq_weight=opt.sty_freq_weight
            )
            
            # Update the z_enc in style features
            sty_feat[0]['z_enc'] = sty_z_enc
            
            # Synthesis
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        # Generate output filename with content and style names
                        output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"
                        
                        # Apply AdaIN if enabled
                        if opt.without_init_adain:
                            adain_z_enc = cnt_z_enc
                        else:
                            adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                        
                        # Merge features with recursive Q' and K' mixing
                        feat_maps = feat_merge(
                            opt, 
                            cnt_feat, 
                            sty_feat, 
                            start_step=start_step,
                            q_prime_weight=opt.q_prime_weight,
                            k_prime_weight=opt.k_prime_weight,
                            cnt_z_enc=cnt_z_enc,
                            sty_z_enc=sty_z_enc,
                            model=model,
                            uc=uc,
                            time_range=time_range,
                            self_attn_indices=self_attn_output_block_indices
                        )
                        if opt.without_attn_injection:
                            feat_maps = None
                        
                        # DDIM sampling
                        samples_ddim, intermediates = sampler.sample(
                            S=ddim_steps,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=adain_z_enc,
                            injected_features=feat_maps,
                            start_step=start_step,
                        )
                        
                        # Decode and save
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        
                        output_path_full = os.path.join(output_path, output_name)
                        img.save(output_path_full)
                        
                        # Progress update
                        if combination_idx % 10 == 0 or combination_idx == total_combinations:
                            elapsed_time = time.time() - begin
                            avg_time_per_combo = elapsed_time / combination_idx
                            remaining_combos = total_combinations - combination_idx
                            eta = remaining_combos * avg_time_per_combo
                            
                            print(f"Progress: {combination_idx}/{total_combinations} ({100*combination_idx/total_combinations:.1f}%) | ETA: {eta:.0f}s")
                        
                        # Clean up GPU memory after each combination
                        del samples_ddim, x_samples_ddim, img
                        torch.cuda.empty_cache()
        
        # Clean up content-related memory after processing all styles
        del init_cnt, cnt_z_enc, cnt_z_enc_raw, init_cnt_encoded
        torch.cuda.empty_cache()
    
    total_time = time.time() - begin
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total combinations processed: {total_combinations}")
    print(f"Average time per combination: {total_time/total_combinations:.2f}s")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()