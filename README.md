# StyleFM

![imgs](asset/imgs.png)

## Usage

### Prerequisites
- Single GPU with 24GB+ VRAM
- Python 3.8+
- PyTorch 1.8.1+

## Setup
Our codebase is built on ([CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), [Jiwoogit/StyleID](https://github.com/jiwoogit/StyleID/blob/main/README.md) and [MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play))
and has similar dependencies and model architecture.

### Create a Conda Environment

```
conda env create -f environment.yaml
conda activate StyleFM
```

### Download StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```


## Run StyleFM

For running StyleFM, run:

```
python run_inference.py --cnt <content_img_dir> --sty <style_img_dir>
```

For running default configuration in sample image files, run:
```
python run_inference.py --cnt data/cnt --sty data/sty  # default
```

To fine-tune the parameters, you have control over the following aspects in the style transfer:

- **Attention-based style injection** is removed by the `--without_attn_injection` parameter.
- **Attention Recursive weight** is controlled by the `--q_prime_weight` and `--k_prime_weight` parameter.
- **Content ** is controlled through the `--T` parameter.
- **Initial latent AdaIN** is removed by the `--without_init_adain` parameter.

Key Parameters
Frequency Processing Parameters
Content Frequency Enhancement

--cnt_d_s_high: Spatial parameter for high-pass filter (default: 0.7)
--cnt_d_s_midhigh: Spatial parameter for mid-high pass filter (default: 0.5)
--cnt_freq_weight: Weight for high frequency enhancement (default: 0.5)
--cnt_alpha: Noise control factor (default: 0.7)

Style Frequency Enhancement

--sty_d_s_lowmid: Spatial parameter for low-mid pass filter (default: 0.7)
--sty_freq_weight: Weight for low-mid frequency enhancement (default: 0.3)

Recursive Mixing Parameters

--q_prime_weight: Recursive Q mixing weight (default: 0.25)
--k_prime_weight: Recursive K/V mixing weight (default: 0.25)

Original StyleID Parameters

--T: Attention temperature scaling (default: 1.0)
--start_step: Starting step for feature injection (default: 49)
--ddim_steps: Number of DDIM steps (default: 50)
--ddim_eta: DDIM stochasticity (default: 0.0 for deterministic)

Additional Options

--without_init_adain: Disable initial AdaIN
--without_attn_injection: Disable attention injection (not recommended)
--enable_verification: Enable detailed debugging output
--save_debug: Save debugging information to file

## Evaluation


For a quantitative evaluation, we incorporate a set of randomly selected inputs from [MS-COCO](https://cocodataset.org) and [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) in "./data" directory.


Before executing evalution code, please duplicate the content and style images to match the number of stylized images first. (40 styles, 20 contents -> 800 style images, 800 content images)

run:
```
python util/copy_inputs.py --cnt data/cnt --sty data/sty
```
For running the evaluation:

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

