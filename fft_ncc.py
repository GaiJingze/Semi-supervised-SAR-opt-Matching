import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def window_sum(x, window_size):
        """
        Computes the sum over a sliding window using separable convolutions.
        
        Instead of a full 2D convolution, we first convolve vertically and then horizontally.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            window_size (int): Size of the sliding window
            
        Returns:
            torch.Tensor: Sum over the window, with shape (B, C, H - window_size + 1, W - window_size + 1)
        """
        
        B, C, H, W = x.shape
        # First, convolve vertically with a kernel of ones (window_size x 1)
        vertical_kernel = torch.ones((C, 1, window_size, 1), dtype=x.dtype, device=x.device)
        out_vert = F.conv2d(x, vertical_kernel, bias=None, stride=1, padding=0, groups=C)
        # Then, convolve horizontally with a kernel of ones (1 x window_size)
        horizontal_kernel = torch.ones((C, 1, 1, window_size), dtype=x.dtype, device=x.device)
        out = F.conv2d(out_vert, horizontal_kernel, bias=None, stride=1, padding=0, groups=C)
        return out
    

class FFTCrossCorrelation(nn.Module):
    def __init__(self):
        super(FFTCrossCorrelation, self).__init__()
    def forward(self, feat_1, feat_2):
        # Determine output shape: (H1 + H2 - 1, W1 + W2 - 1)
        B, C, H1, W1 = feat_1.shape
        _, _, H2, W2 = feat_2.shape
        fft_height = H1 + H2 - 1
        fft_width  = W1 + W2 - 1
        fft_shape = (fft_height, fft_width)
        
        # Reverse second feature map spatially (to mimic cross-correlation)
        feat_2_rev = torch.flip(feat_2, dims=[-2, -1])
        
        # Compute 2D FFTs with zero-padding to fft_shape
        fft1 = torch.fft.rfft2(feat_1, s=fft_shape)
        fft2 = torch.fft.rfft2(feat_2_rev, s=fft_shape)
        
        # Element-wise multiplication in frequency domain
        fft_prod = fft1 * fft2
        
        # Inverse FFT to get cross-correlation map
        out = torch.fft.irfft2(fft_prod, s=fft_shape)
        return out
    
class NormalizationLayer(nn.Module):
    def __init__(self, normalize_by_window=False):
        self.normalize_by_window = normalize_by_window
        self.floor = torch.finfo(torch.float32).eps
        self.ceil = torch.finfo(torch.float32).max

        super(NormalizationLayer, self).__init__()
    
    def forward(self, feat_1, feat_2, xcorr):
        # For demonstration, perform min-max normalization over the spatial dimensions.
        if self.normalize_by_window == False: #use simple normalizaion, faster training but lower performance
            x_min = xcorr.amin(dim=(2,3), keepdim=True)
            x_max = xcorr.amax(dim=(2,3), keepdim=True)
            xcorr_normalized = (xcorr - x_min) / (x_max - x_min + 1e-8)
            return xcorr_normalized
        else:

            B, C, H1, W1 = feat_1.shape
            _, _, H2, W2 = feat_2.shape
            template_size = H2
            # Instead of padding by template_size, pad by template_size - 1
            pad = template_size - 1  # For example, 192 - 1 = 191
            # Padding order for F.pad: (left, right, top, bottom)
            opt_padded = F.pad(feat_1, (pad, pad, pad, pad), mode='constant', value=0)
            
            sar_volume = float(H2 * W2)
            
            # Compute local sums over a window of size template_size.
            winsum  = window_sum(opt_padded, template_size)   # Expected shape: (B, C, H_out, W_out)
            winsum2 = window_sum(opt_padded**2, template_size)
            
            # Now, winsum should have the same spatial dimensions as xcorr.
            numerator = xcorr - winsum * feat_2.mean(dim=(2,3), keepdim=True)
            winsum2_corr = winsum2 - (winsum**2) / sar_volume
            sar_ssd = ((feat_2 - feat_2.mean(dim=(2,3), keepdim=True))**2).sum(dim=(2,3), keepdim=True)
            denom = torch.sqrt(torch.clamp(winsum2_corr * sar_ssd, min=0))
            
            numerator_clipped = torch.clamp(numerator, min=self.floor, max=self.ceil)
            denom_clipped = torch.clamp(denom, min=self.floor, max=self.ceil)
            
            mask = denom > self.floor
            xcorr_normalized = torch.where(mask, numerator_clipped / denom_clipped, torch.zeros_like(xcorr))

            return xcorr_normalized