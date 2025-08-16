import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def crop_heatmap(opt_heatmap, gt, patch_size):
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
    B, C, H_opt, W_opt = opt_heatmap.shape
    H_patch, W_patch = patch_size

    # Create a base grid for the patch (in pixel coordinates)
    # grid_y and grid_x will have shape (H_patch, W_patch)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_patch, device=opt_heatmap.device, dtype=torch.float32),
        torch.arange(W_patch, device=opt_heatmap.device, dtype=torch.float32),
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
    cropped_patch = F.grid_sample(opt_heatmap, grid, mode='bilinear', align_corners=True)
    return cropped_patch

def feature_similarity_loss(opt_heatmap, sar_heatmap, gt):
    # Determine the spatial size of sar_heatmap (assume sar_heatmap has shape (B, C, H_sar, W_sar))
    patch_size = sar_heatmap.shape[2:]  # (H_sar, W_sar)
    
    # Crop the corresponding patch from opt_heatmap based on gt
    cropped_opt_heatmap = crop_heatmap(opt_heatmap, gt, patch_size)

    '''
    opt_sample = sar_heatmap[0].detach().cpu().numpy()  # shape (C, H, W)
    cropped_sample = cropped_opt_heatmap[0].detach().cpu().numpy()

    opt_img = opt_sample.mean(axis=0)
    cropped_img = cropped_sample.mean(axis=0)
    
    # Plot side by side.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(opt_img, cmap='viridis')
    plt.title("Original opt_heatmap (mean across channels)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_img, cmap='viridis')
    plt.title("Cropped opt_heatmap (mean across channels)")
    plt.axis('off')

    plt.savefig("./test")
    
    # Now you can compute a similarity loss between cropped_opt_heatmap and sar_heatmap.
    # For example, using mean squared error:'
    '''
    loss = F.l1_loss(cropped_opt_heatmap, sar_heatmap)
    return loss
