import torch
import torch.nn as nn
import torch.nn.functional as F
from mutual_information_loss import mutual_independence_loss
import matplotlib.pyplot as plt
import os

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

def conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()

        self.conv_k1 = conv2D(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv_k3 = conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_k5 = conv2D(in_channels, out_channels, kernel_size=5, padding=2)
        self.activation = nn.PReLU()

    def forward(self, x):
        x_k1=self.conv_k1(x)
        x_k3=self.conv_k3(x)
        x_k5=self.conv_k5(x)

        return self.activation(x_k1+x_k3+x_k5)

    
class AttentionLayer(nn.Module):
    def __init__(self, channels, nhead=8):
        """
        channels: channel count of features.
        nhead: number of attention heads.
        """
        super(AttentionLayer, self).__init__()
        self.nhead = nhead
        self.dim = channels // nhead  # dimension per head

        # Projection layers for query, key, and value
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.attention = LinearAttention()

        self.merge = nn.Linear(channels, channels, bias=False)

        self.norm = nn.LayerNorm(channels)

    def forward(self, query_feature, key_feature):
        """
        Adjusted forward method for image inputs.
        
        Args:
            feat1 (torch.Tensor): Feature map from modality 1 with shape (N, C, H1, W1)
            feat2 (torch.Tensor): Feature map from modality 2 with shape (N, C, H2, W2)
            feat1_mask, feat2_mask: Optional masks (not used in this example)
            
        Returns:
            torch.Tensor: Updated feature map for feat1, reshaped to (N, C, H1, W1)
        """
        B, C, H1, W1 = query_feature.shape
        B, C, H2, W2 = key_feature.shape
        
        # Flatten spatial dimensions:
        # Reshape feat1 to (N, L, C) and feat2 to (N, S, C)
        query = query_feature.view(B, C, -1).permute(0, 2, 1)      # (N, L, C)
        key = key_feature.view(B, C, -1).permute(0, 2, 1)   # (N, S, C)

        query = self.q_proj(query)
        key   = self.k_proj(key)
        value = self.v_proj(key)

        # Apply linear projections
        query = query.view(B, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = key.view(B, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = value.view(B, -1, self.nhead, self.dim)

        out = self.attention(query, key, value)  # [N, L, (H, D)]
        out = self.merge(out.view(B, -1, self.nhead*self.dim))  # [N, L, C]
        out = self.norm(out)
        out = out.permute(0, 2, 1).view(B, C, H1, W1)

        return out
    
class FeatureEnhancementLayer(nn.Module):
    def __init__(self, channels=32):
        super(FeatureEnhancementLayer, self).__init__()


        self.self_attention_modality_1 = AttentionLayer(channels)
        self.self_attention_modality_2 = AttentionLayer(channels)
        self.m_conv_1 = MultiScaleConv(channels, channels)

        self.cross_attention = AttentionLayer(channels)
        self.m_conv_2 = MultiScaleConv(channels, channels)


    def forward(self, feature_1 ,feature_2):


        modal_feature_1 = self.self_attention_modality_1(feature_1, feature_1)
        modal_feature_2 = self.self_attention_modality_2(feature_2, feature_2)

        feature_1 = feature_1 - modal_feature_1
        feature_2 = feature_2 - modal_feature_2

        feature_1 = self.m_conv_1(feature_1)
        feature_2 = self.m_conv_1(feature_2)

        common_feature_1 = self.cross_attention(feature_1, feature_2)
        common_feature_2 = self.cross_attention(feature_2, feature_1)

        feature_1 = feature_1 + common_feature_1
        feature_2 = feature_2 + common_feature_2

        feature_1 = self.m_conv_2(feature_1)
        feature_2 = self.m_conv_2(feature_2)

        return feature_1, feature_2

class CrossModalFeatureEnhancer(nn.Module):
    def __init__(self, channels=32, num_layers=1):
        super(CrossModalFeatureEnhancer, self).__init__()
        self.layers = nn.ModuleList(
            [FeatureEnhancementLayer(channels) for _ in range(num_layers)]
        )
        self.saved_fig_index = 0

    def visualize_feature(self, feature, title='Feature Map', savepath = './figures'):
        if savepath is not None:
            os.makedirs(savepath, exist_ok=True)
        img = feature[0].mean(dim=0).detach().cpu().numpy() 
        plt.imshow(img, cmap='viridis')
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        plt.show()
        plt.savefig(f"{savepath}/{title}.png")
        plt.close()

    def forward(self, feature_1, feature_2, visualize = False):

        # Apply each feature enhancement layer sequentially
        for layer in self.layers:
            feature_1, feature_2 = layer(feature_1, feature_2)

        if self.training:
            feature_1_raw = feature_1.detach()
            feature_2_raw = feature_2.detach()

            # Apply each feature enhancement layer sequentially
            for layer in self.layers:
                feature_1, feature_2 = layer(feature_1, feature_2)
                
            modal_feature_1 = feature_1_raw - feature_1
            modal_feature_2 = feature_2_raw - feature_2
            
            loss = (mutual_independence_loss(feature_1, modal_feature_1) +
                    mutual_independence_loss(feature_2, modal_feature_2))
            return feature_1, feature_2, loss
        elif visualize:
            feature_1_raw = feature_1.detach()
            feature_2_raw = feature_2.detach()

            # Apply each feature enhancement layer sequentially
            for layer in self.layers:
                feature_1, feature_2 = layer(feature_1, feature_2)
                
            modal_feature_1 = feature_1_raw - feature_1
            modal_feature_2 = feature_2_raw - feature_2

            savepath = f'./figures/feature_maps_{self.saved_fig_index}'
            self.visualize_feature(feature_1, title='enhanced_m1', savepath=savepath)
            self.visualize_feature(modal_feature_1, title='specific_m1', savepath=savepath)
            self.visualize_feature(feature_1_raw, title='raw_m1', savepath=savepath)

            self.visualize_feature(feature_2, title='enhanced_m2', savepath=savepath)
            self.visualize_feature(modal_feature_2, title='specific_m2', savepath=savepath)
            self.visualize_feature(feature_2_raw, title='raw_m2', savepath=savepath)

            self.saved_fig_index += 1
            return feature_1, feature_2, 0
        else:
            for layer in self.layers:
                feature_1, feature_2 = layer(feature_1, feature_2)
            return feature_1, feature_2, 0