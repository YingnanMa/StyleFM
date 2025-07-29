# StyleFM


### Prerequisites
- NVIDIA V100 GPU
- Python 3.8+
- PyTorch 1.8.1+

## Setup
Our codebase is built on ([CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), [Jiwoogit/StyleID](https://github.com/jiwoogit/StyleID/blob/main/README.md) and [MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play)).

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

- **Attention Recursive weight** is controlled by the `--q_prime_weight` and `--k_prime_weight` parameter(k,v share the same weight).
- **Content frequency enhance weight** is controlled through the `--cnt_freq_weight` parameter.
- **Style frequency enhance weight** is controlled through the `--sty_freq_weight` parameter.


## Test StyleFM

Testset (from StyleID) is provided in "./data" directory. Before executing evalution code, please duplicate the content and style images to match the number of stylized images first (40 styles, 20 contents -> 800 style images, 800 content images).
The visualiaztion images we used in the paper are located under "/data_vis " directory.

run:
```
python util/copy_inputs.py --cnt data/cnt --sty data/sty
```

For running the evaluation:

```bash
cd evaluation
python eval.py --tar ../output

```

## Acknowledgments

This work is based on:
- [StyleID](https://github.com/jiwoogit/StyleID) - The original style injection method
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) - The base diffusion model
- [MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play) - Attention manipulation techniques

