import argparse
import glob
import numpy as np
import os
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Grayscale
from tqdm import tqdm
import requests
import tempfile

import image_metrics
import lpips

ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_image_paths(path, sort=False):
    """Returns the paths of the images in the specified directory, filtered by allowed file extensions.

    Args:
        path (str): Path to image directory.
        sort (bool): Sort paths alphanumerically.

    Returns:
        (list): List of image paths with allowed file extensions.

    """
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    
    paths = []
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f'*.{extension}')))
    
    if not paths:
        raise ValueError(f"No images found in {path} with extensions {ALLOWED_IMAGE_EXTENSIONS}")
    
    if sort:
        paths.sort()
    return paths


def compute_lpips_distance(path_to_stylized, path_to_target, batch_size, eval_metric='lpips', device='cuda', num_workers=1, gray=False):
    """Computes the distance for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_target (str): Path to the content/style images.
        batch_size (int): Batch size for computing activations.
        eval_metric (str): Metric to use for content distance. Currently only 'lpips' is supported.
        device (str): Device for computing activations.
        num_workers (int): Number of threads for data loading.
        gray (bool): Whether to convert images to grayscale.

    Returns:
        (float) Distance value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    target_image_paths = get_image_paths(path_to_target, sort=True)

    assert len(stylized_image_paths) == len(target_image_paths), \
           'Number of stylized images and number of target images must be equal.'

    if gray:
        target_transforms = Compose([Resize(512), Grayscale(), ToTensor()])
    else:
        target_transforms = Compose([Resize(512), ToTensor()])
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=target_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_target = ImagePathDataset(target_image_paths, transforms=target_transforms)
    dataloader_target = torch.utils.data.DataLoader(dataset_target,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
    
    metric = image_metrics.LPIPS().to(device)

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_target in zip(dataloader_stylized, dataloader_target):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_target.to(device))
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N


def compute_patch_simi(path_to_stylized, path_to_content, batch_size, device, num_workers=1):
    """Computes the patch similarity for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_content (str): Path to the content images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for computing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) Patch similarity value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    style_transforms = Compose([Resize(512), ToTensor()])
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=style_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=style_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
    metric = image_metrics.PatchSimi(device=device).to(device)

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N


def compute_cfsd(path_to_stylized, path_to_content, device, num_workers=1):
    """Computes CFSD for the given paths.
    
    Note: This function always uses batch_size=1 for patch similarity computation.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_content (str): Path to the content images.
        device (str): Device for computing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (str) CFSD value formatted as string.
    """
    print('Compute CFSD value...')
    
    # Always use batch_size=1 for patch similarity
    simi_val = compute_patch_simi(path_to_stylized, path_to_content, 1, device, num_workers)
    simi_dist = f'{simi_val.item():.4f}'
    return simi_dist


def compute_artLpips(lpips_content_value, lpips_style_value):
    """Computes ArtLPIPS metric.

    Args:
        lpips_content_value (float): LPIPS value between content and stylized images.
        lpips_style_value (float): LPIPS value between style and stylized images.

    Returns:
        (float) ArtLPIPS value.
    """
    artlpips = (1 + lpips_content_value) * (1 + lpips_style_value)
    return artlpips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for computing activations.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--sty', type=str, default='../data/sty_eval', help='Path to style images.')
    parser.add_argument('--cnt', type=str, default='../data/cnt_eval', help='Path to content images.')
    parser.add_argument('--tar', type=str, required=True, help='Path to stylized images.')    
    args = parser.parse_args()
    
    # Handle device
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and args.device == 'cuda':
        print("Warning: CUDA requested but not available, using CPU")
    

    print('Compute LPIPS (content, stylized)...')
    lpips_content = compute_lpips_distance(args.tar, args.cnt, args.batch_size, 'lpips', device.type, args.num_workers)
       
    print('Compute LPIPS (style, stylized)...')
    lpips_style = compute_lpips_distance(args.tar, args.sty, args.batch_size, 'lpips', device.type, args.num_workers)

    cfsd = compute_cfsd(args.tar, args.cnt, device.type, args.num_workers)

    artlpips = compute_artLpips(lpips_content.item(), lpips_style.item())

    # Print summary
    print('\n=== Evaluation Summary ===')
    print(f'\nDetailed Metrics:')
    print(f'  LPIPS(content, stylized): {lpips_content.item():.4f}')
    print(f'  LPIPS(style, stylized): {lpips_style.item():.4f}')
    print(f'  CFSD: {cfsd}')
    print(f'  ArtLPIPS: {artlpips:.4f}')


if __name__ == '__main__':
    main()
