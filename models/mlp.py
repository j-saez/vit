import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, out_feat: int, drop_prob: float):
        """
        Inputs:
            >> in_feat: (int) Number of input features.
            >> hidden_feat: (int) Number of features for the hidden layer.
            >> out_feat: (int) Number of features for the output layer
            >> drop_prob: (float) Dropout probability.
        Outputs:
            >>
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_feat, hidden_feat),
            nn.GELU(),
            nn.Linear(hidden_feat, out_feat),
            nn.Dropout(drop_prob))
        return

    def forward(self, x):
        return self.model(x) # (B, n_patches+1, in_feat) -> (B, n_patches+1, out_feat)
