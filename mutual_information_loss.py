import torch
import torch.nn.functional as F
import numpy as np

def crop_feature(img_feature, gt, patch_size):
    """
    Crop a patch from opt_heatmap based on the top-left coordinate gt so that 
    the output patch has the same size as patch_size (e.g. sar_heatmap spatial dimensions).
    
    Args:
        opt_heatmap (torch.Tensor): Tensor of shape (B, C, H_opt, W_opt)
        gt (torch.Tensor): Tensor of shape (B, 2) with (x, y) coordinates for the top-left corner.
        patch_size (tuple): Tuple (H_patch, W_patch) specifying the desired spatial size.
    
    Returns:
        torch.Tensor: Cropped patch of shape (B, C, H_patch, W_patch)
    """
    B, C, H_opt, W_opt = img_feature.shape
    H_patch, W_patch = patch_size

    # Create a base grid for the patch (in pixel coordinates)
    # grid_y and grid_x will have shape (H_patch, W_patch)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_patch, device=img_feature.device, dtype=torch.float32),
        torch.arange(W_patch, device=img_feature.device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Expand the grid to batch size
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # (B, H_patch, W_patch)
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H_patch, W_patch)
    
    # Offset the grid by the ground truth coordinate.
    # Assuming gt has (x, y) coordinates for each batch element.
    grid_x = grid_x + gt[:, 0].view(B, 1, 1)
    grid_y = grid_y + gt[:, 1].view(B, 1, 1)
    
    # Normalize grid coordinates to [-1, 1] range required by grid_sample.
    # For x: scale by (W_opt - 1) and for y: scale by (H_opt - 1)
    grid_x = 2.0 * grid_x / (W_opt - 1) - 1.0
    grid_y = 2.0 * grid_y / (H_opt - 1) - 1.0
    
    # Combine the grids into a single grid tensor of shape (B, H_patch, W_patch, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    
    # Use grid_sample to crop the patch from opt_heatmap
    cropped_feature = F.grid_sample(img_feature, grid, mode='bilinear', align_corners=True)
    return cropped_feature

def mutual_independence_loss(img_feature_1, img_feature_2, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
    """
    Mutual Information loss function.
    Args:
        feature_1 (Tensor): tensor of shape (B, C, H, W)
        feature_2 (Tensor): tensor of shape (B, C, H, W)
        sigma_ratio (float): Ratio to control Gaussian kernel bandwidth
        minval (float): Minimum intensity value to consider
        maxval (float): Maximum intensity value to consider
        num_bin (int): Number of histogram bins
    Returns:
        torch.Tensor: Scalar loss value
    """

    feature_1 = F.avg_pool2d(img_feature_1, (8, 8)).view(img_feature_1.size(0), -1)
    feature_2 = F.avg_pool2d(img_feature_2, (8, 8)).view(img_feature_2.size(0), -1)

    # Clamp input values to [minval, maxval]
    feature_1 = torch.clamp(feature_1, min=minval, max=maxval)
    feature_2 = torch.clamp(feature_2, min=minval, max=maxval)

    # Flatten inputs and add channel dimension for bins
    feature_1 = feature_1.view(feature_1.shape[0], -1).unsqueeze(2)
    feature_2 = feature_2.view(feature_2.shape[0], -1).unsqueeze(2)
    nb_voxels = feature_2.shape[1]

    # Create bin centers and Gaussian sigma
    bin_centers = torch.linspace(minval, maxval, num_bin).to(feature_1.device)
    sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
    preterm = 1 / (2 * sigma ** 2)

    # Reshape bin centers for broadcasting
    vbc = bin_centers.view(1, 1, -1)

    # Compute soft histograms (Gaussian kernel smoothing)
    I_a = torch.exp(-preterm * (feature_1 - vbc).pow(2))
    I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

    I_b = torch.exp(-preterm * (feature_2 - vbc).pow(2))
    I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

    # Estimate joint and marginal probabilities
    pab = torch.bmm(I_a.permute(0, 2, 1), I_b) / nb_voxels
    pa = torch.mean(I_a, dim=1, keepdim=True)
    pb = torch.mean(I_b, dim=1, keepdim=True)
    papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6

    # Compute mutual information
    mi = torch.sum(pab * torch.log(pab / papb + 1e-6), dim=(1, 2))

    # Return negative MI as loss
    return mi.mean()

def cross_modal_mi_loss(opt_feature, sar_feature, gt):
    cropped_opt_feature = crop_feature(opt_feature, gt, sar_feature.shape[2:])
    return -mutual_independence_loss(cropped_opt_feature, sar_feature)