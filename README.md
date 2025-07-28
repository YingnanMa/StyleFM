# StyleFM

![imgs](asset/imgs.png)

## Overview




## Usage

### Prerequisites
- Single GPU with 20GB+ VRAM
- Python 3.8+
- PyTorch 1.8.1+

### Setup
Our codebase is built on ([CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) and [Jiwoogit/StyleID](https://github.com/jiwoogit/StyleID/blob/main/README.md))
and has similar dependencies and model architecture.

1. **Create Conda Environment**
```bash
conda env create -f environment.yaml
conda activate StyleFM
```

2. **Download Stable Diffusion Weights**
Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```

3. **Install All dependency**
```bash
pip install -r requirements.txt
```

## Run StyleFM

### Basic Usage
```bash
python run_inference.py --cnt <content_dir> --sty <style_dir>
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
