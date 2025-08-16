import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

debug = False

def get_hard_label(gt, H, W):
    """
    Generate a hard label heatmap of shape (B, size, size) where only the ground truth coordinate is 1,
    and all other positions are 0.
    
    Args:
        gt (torch.Tensor): Ground truth coordinates of shape (B, 2)
        size (int): The spatial size of the output heatmap (assumed square, e.g., 65).
        
    Returns:
        torch.Tensor: Hard label heatmaps of shape (B, size, size).
    """
    B = gt.shape[0]
    # Create a flat tensor of zeros: shape (B, size*size)
    hard_label_flat = torch.zeros(B, H * W, device=gt.device)
    
    # Convert (row, col) into a flat index.
    indices = (gt[:, 1] * H + gt[:, 0]).long().unsqueeze(1)  # shape: (B, 1)
    
    # Set the corresponding index to 1 using scatter.
    hard_label_flat.scatter_(1, indices, 1.0)
    
    # Reshape back to (B, size, size)
    hard_label = hard_label_flat.view(B, H, W)
    return hard_label


def get_soft_label(gt, H, W, sigma=2.0):

    """
    Generate a soft label heatmap of shape (B, size, size) using a Gaussian centered at gt.

    Args:
        gt (torch.Tensor): Ground truth coordinates of shape (B, 2)
        size (int): The spatial size of the output heatmap (assumed square, e.g., 65).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Soft label heatmaps of shape (B, size, size).
    """
    B = gt.shape[0]
    # Create a coordinate grid for the heatmap
    xs = torch.arange(W).float().to(gt.device)
    ys = torch.arange(H).float().to(gt.device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')  # shape (size, size)

    # Expand grid to batch size: (B, size, size)
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

    # Unpack ground truth coordinate
    gt_x = gt[:, 0].unsqueeze(1).unsqueeze(2)
    gt_y = gt[:, 1].unsqueeze(1).unsqueeze(2)

    # Compute squared distances from the ground truth coordinate for each pixel
    dist_squared = (x_grid - gt_x) ** 2 + (y_grid - gt_y) ** 2

    # Compute the Gaussian: maximum at the gt coordinate
    heatmap = torch.exp(-dist_squared / (2 * sigma ** 2))
    
    return heatmap

def cross_entropy_with_nmining(y_pred, gt, temperature = 1/30, n_neg_samples=16):
    """
    Computes a combined loss with negative mining.
    
    Args:
        y_true (torch.Tensor): Soft ground truth labels of shape (B, 65, 65). 
                               This is a probability distribution over locations.
        y_pred (torch.Tensor): Predicted logits of shape (B, 65, 65
        n_neg_samples (int): Number of negative samples to mine from the non-matching region.
        
    Returns:
        torch.Tensor: Scalar loss (mean over the batch).
    """
    B, H, W = y_pred.shape

    y_true_soft = get_soft_label(gt, H, W)
    # 1. Matching Region: Compute the average predicted score over the matching region.
    # Count the nonzero elements (i.e. the region where y_true > 0).
    count_nonzero = (y_true_soft != 0).sum(dim=(1, 2)).float() + 1e-8

    matching_region_samples = (y_pred * y_true_soft).sum(dim=(1,2)) / count_nonzero  # shape (B,)

    # 2. Negative Region: Predictions outside the matching region.
    negative_region = y_pred * (1 - y_true_soft)
    # Replace zeros with ones to avoid selecting them as negatives.
    negative_region_filtered = torch.where(negative_region == 0, torch.ones_like(negative_region), negative_region)
    # Flatten to shape (B, 65*65)
    negative_region_filtered = negative_region_filtered.view(B, -1)
    
    # Select the n_neg_samples smallest values (i.e. hardest negatives).
    # We use topk on the negative of the tensor to get the smallest values.
    neg_topk, _ = torch.topk(-negative_region_filtered, k=n_neg_samples, dim=1)
    neg_samples = -neg_topk.mean(dim=1) + 1  # shape (B,); add a margin of 1

    # 3. Negative Mining Term: If the matching region's average is less than the negatives plus 1, penalize.
    nm = torch.clamp(-(matching_region_samples - neg_samples), min=0)  # shape (B,)

    # 4. Cross Entropy Term:
    # Flatten predictions.
    y_pred_flatten = y_pred.view(B, -1)  # shape (B, 4225)
    y_pred_flatten = y_pred_flatten / temperature

    y_true = get_hard_label(gt, H, W)
    y_true_flatten = y_true.view(B, -1)

    # Compute standard softmax cross entropy.
    xent = F.cross_entropy(y_pred_flatten, y_true_flatten)

    # 5. Total Loss: Sum the cross entropy term and the negative mining term, then average over batch.
    total_loss = xent

    return total_loss


def cross_entropy_loss(y_pred, gt, temperature = 1/100):
    """
    Compute a combined loss that first creates a soft ground truth label via gaussian_blur,
    then computes a cross entropy term (using the hard target derived from the soft label)
    and a negative mining term (using the soft label).
    
    Args:
        output (torch.Tensor): Predicted similarity logits with shape (B, 65, 65).
        gt (torch.Tensor): Ground truth offsets with shape (B, 2), each row is (row, col).
        threshold (float): Threshold used in the gaussian_blur function.
        n_neg_samples (int): Number of negative samples to mine.
        
    Returns:
        torch.Tensor: Scalar loss.
    """
    B, H, W = y_pred.shape

    y_true = get_hard_label(gt, H, W)

    y_true_flatten = y_true.view(B, -1)  # shape (B,)
    y_pred_flatten = y_pred.view(B, -1)  # shape (B, 65*65))
    
    y_pred_flatten = y_pred_flatten / temperature

    loss = F.cross_entropy(y_pred_flatten, y_true_flatten)

    if debug:
        batch_idx = 0
        y_true_np = y_true[batch_idx].detach().cpu().numpy()
        y_pred_np = y_pred[batch_idx].detach().cpu().numpy()
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(y_true_np, cmap='hot')
        plt.title("Ground Truth Soft Label")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(y_pred_np, cmap='hot')
        plt.title("Predicted Heatmap")
        plt.colorbar()
        
        plt.savefig(f"./loss_map_{batch_idx}")
        plt.close()

    return loss