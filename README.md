# StyleFM

![imgs](asset/imgs.png)

## Overview




## Usage

### Prerequisites
- Single GPU with 20GB+ VRAM
- Python 3.8+
- PyTorch 1.8.1+

### Setup

1. **Create Conda Environment**
```bash
conda env create -f environment.yaml
conda activate StyleFM
```

2. **Download Stable Diffusion Weights**
Download the StableDiffusion weights from the CompVis organization at Hugging Face (download the sd-v1-4.ckpt file), and link them:
```bash
# Download sd-v1-4.ckpt from HuggingFace
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt
```

3. **Install All dependency**
```bash
pip install -r requirements.txt
```

## Run Frequency-Enhanced StyleID

### Basic Usage
```bash
python run_styleid_freq.py --cnt <content_dir> --sty <style_dir>
```

### Example Commands
```bash
# Default settings
python run_styleid_freq.py --cnt ./samplecnt --sty ./samplesty

# Strong frequency enhancement
python run_styleid_freq.py --cnt ./samplecnt --sty ./samplesty \
    --cnt_freq_weight 0.7 --sty_freq_weight 0.5 \
    --q_prime_weight 0.3 --k_prime_weight 0.3

# Subtle style transfer
python run_styleid_freq.py --cnt ./samplecnt --sty ./samplesty \
    --cnt_freq_weight 0.3 --sty_freq_weight 0.2 \
    --q_prime_weight 0.15 --k_prime_weight 0.15
```

## Key Parameters

### Frequency Processing Parameters

#### Content Frequency Enhancement
- `--cnt_d_s_high`: Spatial parameter for high-pass filter (default: 0.7)
- `--cnt_d_s_midhigh`: Spatial parameter for mid-high pass filter (default: 0.5)
- `--cnt_freq_weight`: Weight for high frequency enhancement (default: 0.5)
- `--cnt_alpha`: Noise control factor (default: 0.7)

#### Style Frequency Enhancement
- `--sty_d_s_lowmid`: Spatial parameter for low-mid pass filter (default: 0.7)
- `--sty_freq_weight`: Weight for low-mid frequency enhancement (default: 0.3)

### Recursive Mixing Parameters
- `--q_prime_weight`: Recursive Q mixing weight (default: 0.25)
- `--k_prime_weight`: Recursive K/V mixing weight (default: 0.25)

### Original StyleID Parameters
- `--T`: Attention temperature scaling (default: 1.0)
- `--start_step`: Starting step for feature injection (default: 49)
- `--ddim_steps`: Number of DDIM steps (default: 50)
- `--ddim_eta`: DDIM stochasticity (default: 0.0 for deterministic)

### Additional Options
- `--without_init_adain`: Disable initial AdaIN
- `--without_attn_injection`: Disable attention injection (not recommended)
- `--enable_verification`: Enable detailed debugging output
- `--save_debug`: Save debugging information to file

## Technical Details

### Frequency Processing Pipeline
```
Content Image → DDIM Inversion → Frequency Enhancement (High-pass) → Q Feature Extraction
Style Image → DDIM Inversion → Frequency Enhancement (Low-mid) → K/V Feature Extraction
```

### Recursive Mixing Process
```
t=0: Q[0]' = Q[0] + α×Q[FM], K[0]' = K[0] + β×K[FM]
t>0: Q[t]' = Q[t] + α×Q[t-1]', K[t]' = K[t] + β×K[t-1]'
```

## Output Structure
```
output_path/
├── content1_stylized_style1.png
├── content1_stylized_style2.png
├── ...
└── debug_info_pair_0.pkl (if --save_debug)
```

## Performance

- **Speed**: ~15-16 it/s on NVIDIA RTX GPU
- **Memory**: ~18.5GB VRAM usage
- **Quality**: Enhanced detail preservation with better style transfer

## Troubleshooting

1. **GPU Memory Issues**
   - Reduce batch size to 1
   - Use smaller images (512x512)
   - Clear GPU cache between pairs

2. **Weak Style Transfer**
   - Increase `--sty_freq_weight` (up to 0.5)
   - Increase `--k_prime_weight` (up to 0.4)
   - Decrease `--start_step` to 40-45

3. **Loss of Content Structure**
   - Decrease `--cnt_freq_weight` (down to 0.3)
   - Decrease `--q_prime_weight` (down to 0.15)
   - Ensure `--cnt_alpha` ≥ 0.7

## Evaluation

Use the same evaluation metrics as StyleID:

```bash
# Art-LPIPS
cd evaluation
python eval_artlpips.py --tar ../output

```



## Acknowledgments

This work is based on:
- [StyleID](https://github.com/jiwoogit/StyleID) - The original style injection method
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) - The base diffusion model
- [MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play) - Attention manipulation techniques

## License

This project inherits the license from StyleID and Stable Diffusion. Please refer to their respective licenses for usage terms.
