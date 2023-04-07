import torch
import torch.nn as nn
from models.layers.attention import AttentionLayer
from models.mlp import MLP

"""
"""

class TransformerEncoder(nn.Module):

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, qkv_bias: bool, proj_drop_prob:float , attn_drop_p: float):
        """
        Vision transformer encoder implementation
        Inputs:
            >>  dim: (int) Number of embedding dimensions.
            >>  n_heads: (int) Number of attention heads
            >>  mlp_ratio: (float) Dtermines the hidden dimension size of th MLP module respect to 'dim'.
            >>  qkv_biar: (bool) If True then we inlude biar to the query, key and value projections
            >> attn_drop_prob: (float) Dropout probability applied to the query, key and value tensors.
            >> proj_drop_prob: (float) Dropout probability applied to the output tensor.
        Attributes:
            >> layer_norm1: (nn.LayerNorm)
            >> layer_norm2: (nn.LayerNorm)
            >> attn: (AttentionLayer)
            >> mlp: (MLP)
        """
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6) # 1e-6 to match the pretrained model and be able to check if if works.
        self.attn_layer = AttentionLayer(dim,n_heads,qkv_bias,attn_drop_p,proj_drop_prob)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6) # 1e-6 to match the pretrained model and be able to check if if works.
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP( in_feat=dim, hidden_feat=hidden_features, out_feat=dim, drop_prob=0.0)
        return

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            >> x: (torch.Tensor [n_samples, n_patches+1, dim])
        Outputs:
            >> x: (torch.Tensor [n_samples, n_patches+1, dim])
        """
        # Check this image for better understanding: https://www.researchgate.net/profile/Jacob-Heilmann-Clausen/publication/357885173/figure/fig1/AS:1113907477389319@1642587646516/Vision-Transformer-architecture-main-blocks-First-image-is-split-into-fixed-size.png
        x = x + self.attn_layer(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x
