import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models


class LPIPS(nn.Module):
    """LPIPS metric using AlexNet backbone."""
    
    def __init__(self):
        super(LPIPS, self).__init__()
        self.dist = lpips.LPIPS(net='alex')

    def forward(self, x, y):
        """
        Compute LPIPS distance between two images.
        
        Args:
            x: First image tensor, should be in range [0, 1]
            y: Second image tensor, should be in range [0, 1]
            
        Returns:
            LPIPS distance value
        """
        # Convert from [0, 1] to [-1, 1] as required by LPIPS
        dist = self.dist(2 * x - 1, 2 * y - 1)
        return dist

#cfsd function obtained from StyleID
class PatchSimi(nn.Module):
    """Patch similarity metric for content feature similarity distance (CFSD)."""
    
    def __init__(self, device=None):
        super(PatchSimi, self).__init__()
        self.device = device
        
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Use conv3 layer for patch similarity
        self.layers = {"11": "conv3"}
        
        # ImageNet normalization parameters
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)

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
        """
        Compute patch similarity distance between input and target.
        
        Args:
            input: Input image tensor
            target: Target image tensor
            
        Returns:
            Patch similarity distance (KL divergence)
        """
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


class Metric(nn.Module):
    """Legacy metric class for AlexNet - redirects to LPIPS."""
    
    def __init__(self, metric_type='alexnet'):
        super(Metric, self).__init__()
        if metric_type != 'alexnet':
            raise ValueError(f'Only alexnet is supported, got: {metric_type}')
        self.metric = LPIPS()
    
    def forward(self, x, y):
        return self.metric(x, y)
