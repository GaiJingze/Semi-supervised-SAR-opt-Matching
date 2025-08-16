import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SAROptMatcher
from QXSLAB_dataset import QXSLAB_dataset
from Sen12_dataset import Sen12_dataset
from supervised_loss import cross_entropy_loss, cross_entropy_with_nmining
from model_utils import get_predicted_matching_pos

def evaluate_prediction_accuracy(test_dataset, model_path, save_heatmap=False):

    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model with the same parameters used during saving.
    model = SAROptMatcher()
    model.to(device)

    #checkpoint = torch.load("./train_checkpoints/sar_opt_matcher_epoch_1.pth", map_location=device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    # Load the saved state dictionary. If loading on CPU, you can use map_location.
    checkpoint = torch.load(model_path, map_location=device)
    if  'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    idx = 0
    total_distance = 0
    within_1px_count = 0
    within_2px_count = 0
    within_5px_count = 0

    # Set the model to evaluation mode (if needed)
    model.eval()
    # set model to evaluation mode
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating", leave=False)
        for batch in progress_bar:

            # Move images and ground truth to device
            img_1 = batch['img_1'].to(device)
            img_2 = batch['img_2'].to(device)
            gt = batch['gt'].to(device)  # assumed shape (B, 2)
            # Forward pass through the model to obtain the output heatmap (B, 65, 65)
            pred_coords = model(img_1, img_2)
            
            # Get predicted coordinates using the function defined earlier
            #pred_coords = get_predicted_matching_pos(output)  # shape (B, 2)
            #B = output.shape[0]

            # Compute per-sample Euclidean distances
            distances = torch.norm(pred_coords.float() - gt.float(), dim=1)
            #loss_ce = cross_entropy_loss(output, gt)
            #loss_nce = cross_entropy_negative_mining(output, gt)

            #total_ce += loss_ce
            #total_nce += loss_nce

            for distance in distances:
                idx += 1
                total_distance += distance

                if distance <= 1:
                    within_1px_count += 1

                if distance <= 2:
                    within_2px_count += 1

                if distance <= 5:
                    within_5px_count += 1

    avg_distance = total_distance / idx
    within_1px_percent = within_1px_count / idx
    within_2px_percent = within_2px_count / idx
    within_5px_percent = within_5px_count / idx

    print(f"avg RMSE: {avg_distance:.4f}")
    print(f"CMR (T=1): {within_1px_percent:.4f}")
    print(f"CMR (T=2): {within_2px_percent:.4f}")
    print(f"CMR (T=5): {within_5px_percent:.4f}")

if __name__ == "__main__":
    test_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(280000, 282384))
    evaluate_prediction_accuracy(test_dataset, "./model_checkpoints/sar_opt_matcher.pth", False)