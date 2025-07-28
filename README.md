# StyleID: Identity-Preserving Text-to-Image Generation

StyleID is a novel approach for text-to-image generation that preserves identity information while applying artistic style transformations. This implementation leverages Stable Diffusion with advanced attention mechanisms and frequency-domain processing to achieve high-quality style transfer with identity preservation.


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


## Output

Results are saved to the specified output directory with filenames in the format:
`{content_name}_stylized_{style_name}.png`


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
