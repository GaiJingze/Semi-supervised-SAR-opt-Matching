import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Sen12_dataset import Sen12_dataset

from ResNetFPN import ResNetFPN
from fft_ncc import FFTCrossCorrelation, NormalizationLayer
from feature_enhance import CrossModalFeatureEnhancer
from model_utils import get_predicted_matching_pos
    
class SAROptMatcher(nn.Module):
    def __init__(self):
        """
        reference_im_shape: tuple (H_opt, W_opt) for optical image (assumed single channel)
        floating_im_shape: tuple (H_sar, W_sar) for SAR image (assumed single channel)
        """
        super(SAROptMatcher, self).__init__()
        self.backbone = ResNetFPN()
        self.deep_feature_enhancer = CrossModalFeatureEnhancer(channels=256, num_layers=1)
        self.shallow_feature_enhancer = CrossModalFeatureEnhancer(channels=32, num_layers=1)
        
        # FFT cross-correlation module
        self.fft = FFTCrossCorrelation()
        self.norm_layer= NormalizationLayer()

    def forward(self, opt_img, sar_img, visualize_heatmap = False):
        """
        opt_img: optical input tensor of shape (B, 3, H_opt, W_opt) (e.g., 256x256)
        sar_img: SAR input tensor of shape (B, 3, H_sar, W_sar) (e.g., 192x192)
        """

        # Backbone feature extraction:
        deep_opt_feature, shallow_opt_feature = self.backbone(opt_img)  # (B, 32, H_opt, W_opt)
        deep_sar_feature, shallow_sar_feature = self.backbone(sar_img)    # (B, 32, H_sar, W_sar)
        

        deep_opt_feature, deep_sar_feature, _ = self.deep_feature_enhancer(deep_opt_feature, deep_sar_feature, visualize_heatmap)
        shallow_opt_feature, shallow_sar_feature, _ = self.shallow_feature_enhancer(shallow_opt_feature, shallow_sar_feature, visualize_heatmap)
        
        _, _, H_sar_d, W_sar_d = deep_sar_feature.shape
        _, _, H_sar_s, W_sar_s = shallow_sar_feature.shape

        # FFT-based Cross-Correlation
        deep_fft_corr = self.fft(deep_opt_feature, deep_sar_feature)
        deep_fft_corr = self.norm_layer(deep_opt_feature, deep_sar_feature, deep_fft_corr)

        crop_H = H_sar_d - 1
        crop_W = W_sar_d - 1

        deep_fft_corr = deep_fft_corr[:, :, crop_H:-crop_H, crop_W:-crop_W]

        shallow_fft_corr = self.fft(shallow_opt_feature, shallow_sar_feature)
        shallow_fft_corr = self.norm_layer(shallow_opt_feature, shallow_sar_feature, shallow_fft_corr)

        crop_H = H_sar_s - 1
        crop_W = W_sar_s - 1
        
        shallow_fft_corr = shallow_fft_corr[:, :, crop_H:-crop_H, crop_W:-crop_W]

        deep_fft_corr = deep_fft_corr.mean(dim=1, keepdim=True)
        shallow_fft_corr = shallow_fft_corr.mean(dim=1, keepdim=True)

        deep_fft_corr_upscaled = F.interpolate(deep_fft_corr, size=shallow_fft_corr.shape[-2:], mode='nearest')
        deep_fft_corr_upscaled = deep_fft_corr_upscaled.squeeze(1)
        shallow_fft_corr = shallow_fft_corr.squeeze(1)

        fft_for_prediction = deep_fft_corr_upscaled * shallow_fft_corr

        prediction = get_predicted_matching_pos(fft_for_prediction)
        return prediction

if __name__ == "__main__": #For debug
    model = SAROptMatcher()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(277384, 282384))
    dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    with torch.no_grad():
        for batch in dataloader:
            # Move images and ground truth to device
            opt_img = batch['img_1'].to(device)
            sar_img = batch['img_2'].to(device)

            gt = batch['gt'].to(device)
            pred = model(opt_img, sar_img)
            print("gt", gt.detach().cpu().numpy())
            print("pred", pred.detach().cpu().numpy())
            break
