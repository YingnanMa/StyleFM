# StyleID: Identity-Preserving Text-to-Image Generation

StyleID is a novel approach for text-to-image generation that preserves identity information while applying artistic style transformations. This implementation leverages Stable Diffusion with advanced attention mechanisms and frequency-domain processing to achieve high-quality style transfer with identity preservation.

## Features

- **Identity-Preserving Style Transfer**: Maintains the structural identity of content images while applying artistic styles
- **Frequency-Domain Processing**: Separate handling of high-frequency content details and low-mid frequency style patterns
- **Recursive Attention Mixing**: Progressive blending of Q' and K' features across denoising timesteps
- **AdaIN Integration**: Adaptive Instance Normalization for better style-content alignment
- **Batch Processing**: Efficient processing of multiple content-style combinations

## Requirements

- Python 3.8+
- PyTorch 1.11.0 with CUDA support
- Stable Diffusion v1 checkpoint

Install dependencies:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StyleID.git
cd StyleID
```

2. Download the Stable Diffusion v1 model checkpoint:
   - Place the model checkpoint at `models/ldm/stable-diffusion-v1/model.ckpt`

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run style transfer on default content and style images:

```bash
python run_inference.py
```

### Custom Inputs

Specify your own content and style directories:

```bash
python run_inference.py --cnt path/to/content/images --sty path/to/style/images --output_path results
```

### Advanced Parameters

```bash
python run_inference.py \
    --cnt ./data/cnt \
    --sty ./data/sty \
    --output_path ./output \
    --ddim_inv_steps 50 \
    --ddim_eta 0.0 \
    --T 1.0 \
    --q_prime_weight 0.1 \
    --k_prime_weight 0.1 \
    --cnt_freq_weight 0.5 \
    --sty_freq_weight 0.3
```

### Key Parameters

#### Attention Control
- `--T`: Attention temperature scaling (default: 1.0)
- `--attn_layer`: Injection attention feature layers (default: '6,7,8,9,10,11')
- `--q_prime_weight`: Weight for recursive Q' mixing (default: 0.1)
- `--k_prime_weight`: Weight for recursive K' mixing (default: 0.1)

#### Frequency Processing
- `--cnt_d_s_high`: Spatial parameter for content high-pass filter (default: 0.7)
- `--cnt_d_t_high`: Temporal parameter for content high-pass filter (default: 0.3)
- `--cnt_freq_weight`: Weight for content high frequency component (default: 0.5)
- `--sty_d_s_lowmid`: Spatial parameter for style low-mid pass filter (default: 0.7)
- `--sty_freq_weight`: Weight for style low-mid frequency component (default: 0.3)

#### Other Options
- `--without_init_adain`: Disable initial AdaIN
- `--without_attn_injection`: Disable attention injection
- `--start_step`: Starting step for feature injection (default: 49)

## Input Format

- **Content Images**: Place content images in `./data/cnt/` directory
- **Style Images**: Place style images in `./data/sty/` directory
- Supported formats: PNG, JPG, JPEG, BMP, TIFF
- Images will be automatically resized to 512x512

## Output

Results are saved to the specified output directory with filenames in the format:
`{content_name}_stylized_{style_name}.png`

## Method Overview

StyleID combines several techniques for identity-preserving style transfer:

1. **DDIM Inversion**: Encodes both content and style images into latent representations
2. **Frequency Separation**: 
   - High and mid-high frequencies from content (preserving details)
   - Low-mid frequencies from style (capturing artistic patterns)
3. **Recursive Feature Mixing**: Progressive blending of attention features (Q', K', V) across denoising steps
4. **AdaIN**: Optional adaptive normalization for initial style-content alignment

## Project Structure

```
StyleID/
├── data/
│   ├── cnt/          # Content images
│   └── sty/          # Style images
├── ldm/              # Latent Diffusion Model components
│   ├── models/       # Model architectures
│   └── modules/      # Model modules (attention, etc.)
├── models/           # Model configurations
├── util/             # Utility functions
│   └── frequency_util.py  # Frequency processing utilities
├── evaluation/       # Evaluation metrics
├── run_inference.py  # Main inference script
└── requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{styleid2024,
  title={StyleID: Identity-Preserving Text-to-Image Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built upon Stable Diffusion v1
- Inspired by attention-based style transfer methods
- Uses frequency domain analysis for content-style separation
