# StyleFM

![imgs](asset/imgs.png)

## Usage

### Prerequisites
- Single GPU with 24GB+ VRAM
- Python 3.8+
- PyTorch 1.8.1+

### Setup
Our codebase is built on ([CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), [Jiwoogit/StyleID](https://github.com/jiwoogit/StyleID/blob/main/README.md) and [MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play))
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

## Evaluation


For a quantitative evaluation, we incorporate a set of randomly selected inputs from [MS-COCO](https://cocodataset.org) and [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) in "./data" directory.


Before executing evalution code, please duplicate the content and style images to match the number of stylized images first. (40 styles, 20 contents -> 800 style images, 800 content images)

run:
```
python util/copy_inputs.py --cnt data/cnt --sty data/sty
```

We largely employ [matthias-wright/art-fid](https://github.com/matthias-wright/art-fid) and [mahmoudnafifi/HistoGAN](https://github.com/mahmoudnafifi/HistoGAN) for our evaluation.


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

