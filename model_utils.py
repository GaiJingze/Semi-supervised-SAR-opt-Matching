import torch
def get_predicted_matching_pos(corr_heatmap):
    """
    Given a heatmap of shape (B, 1, H, W), returns predicted coordinates of the maximum value
    for each sample as a tensor of shape (B, 2) where each row is (row, col).
    """
    B, H, W = corr_heatmap.shape
    # Flatten the spatial dimensions and get the index of the maximum value for each sample.
    max_idx = corr_heatmap.view(B, -1).argmax(dim=1)
    # Compute the row and column coordinates.
    pred_y = max_idx // W
    pred_x = max_idx % W

    pred_matching_pos = torch.stack([pred_x, pred_y], dim=1)  # Shape: (B, 2)
    return pred_matching_pos
