import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models


def normalize(x):
    """Normalize input tensor using ImageNet statistics."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, -1, 1, 1)
    x = (x - mean) / std
    return x


class Metric(nn.Module):
    """Generic metric class for VGG, AlexNet, SSIM, and MS-SSIM metrics."""
    
    def __init__(self, metric_type='vgg'):
        super(Metric, self).__init__()
        self.metric_type = metric_type
        
        
        if metric_type == 'alexnet':
            self.model = lpips.pn.alexnet()
        else:
            raise ValueError(f'Invalid metric type: {metric_type}')

    def forward(self, x, y):
        """Compute distance between two images."""
        if self.metric_type in ['ssim', 'ms-ssim']:
            dist = self.model(x, y)
            return dist
        else:
            # For VGG and AlexNet
            features_x = self.model(normalize(x))._asdict()
            features_y = self.model(normalize(y))._asdict()
            
            dist = 0.0
            for layer in features_x.keys():
                dist += torch.mean(torch.square(features_x[layer] - features_y[layer]), dim=(1, 2, 3))
            return dist / len(features_x)


class LPIPS(nn.Module):
    """LPIPS metric using AlexNet backbone."""
    
    def __init__(self):
        super(LPIPS, self).__init__()
        self.dist = lpips.LPIPS(net='alex')

    def forward(self, x, y):
        """Compute LPIPS distance. Images must be in range [0, 1]."""
        # Convert from [0, 1] to [-1, 1] as required by LPIPS
        dist = self.dist(2 * x - 1, 2 * y - 1)
        return dist


class LPIPS_vgg(nn.Module):
    """LPIPS metric using VGG backbone."""
    
    def __init__(self):
        super(LPIPS_vgg, self).__init__()
        self.dist = lpips.LPIPS(net='vgg')

    def forward(self, x, y):
        """Compute LPIPS distance. Images must be in range [0, 1]."""
        # Convert from [0, 1] to [-1, 1] as required by LPIPS
        dist = self.dist(2 * x - 1, 2 * y - 1)
        return dist


class PatchSimi(nn.Module):
    """Patch similarity metric for content feature similarity distance (CFSD)."""
    
    def __init__(self, device=None):
        super(PatchSimi, self).__init__()
        self.device = device
        
        # Load pre-trained VGG19 and extract features
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Use conv3 layer for patch similarity
        self.layers = {"11": "conv3"}
        
        # ImageNet normalization parameters
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)
        
        # KL divergence loss
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def get_feats(self, img):
        """Extract features from specified layers."""
        features = []
        for name, layer in self.model._modules.items():
            img = layer(img)
            if name in self.layers:
                features.append(img)
        return features
    
    def normalize(self, input):
        """Normalize input using ImageNet statistics."""
        return transforms.functional.normalize(input, self.norm_mean, self.norm_std)

    def patch_simi_cnt(self, input):
        """Compute log-softmax patch similarity for content features."""
        b, c, h, w = input.size()
        input = torch.transpose(input, 1, 3)
        features = input.reshape(b, h*w, c).div(c)  # Resize F_XL into \hat F_XL
        feature_t = torch.transpose(features, 1, 2)
        patch_simi = F.log_softmax(torch.bmm(features, feature_t), dim=-1)
        return patch_simi.reshape(b, -1)

    def patch_simi_out(self, input):
        """Compute softmax patch similarity for output features."""
        b, c, h, w = input.size()
        input = torch.transpose(input, 1, 3)
        features = input.reshape(b, h*w, c).div(c)
        feature_t = torch.transpose(features, 1, 2)
        patch_simi = F.softmax(torch.bmm(features, feature_t), dim=-1)
        return patch_simi.reshape(b, -1)

    def forward(self, input, target):
        """Compute patch similarity distance between input and target."""
        src_feats = self.get_feats(self.normalize(input))
        target_feats = self.get_feats(self.normalize(target))
        
        init_loss = 0.0
        for idx in range(len(src_feats)):
            init_loss += F.kl_div(
                self.patch_simi_cnt(src_feats[idx]), 
                self.patch_simi_out(target_feats[idx]), 
                reduction='batchmean'
            )
        return init_loss
