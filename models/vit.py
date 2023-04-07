import torch
import torch.nn as nn

from models.layers.patch_embedding import PatchEmbedder
from models.transformer_encoder import TransformerEncoder

class ViT(nn.Module):

    def __init__(self,
         img_size: int,
         patch_size: int,
         in_chs: int, 
         n_classes:int, 
         total_embeddings: int,
         transformer_blocks: int,
         n_heads: int,
         mlp_ratio:float,
         qkv_bias: bool,
         p: float,
         attn_drop_prob:float):
        """
        Vision transformer simplified implementation
        Inputs:
            >> img_size: (int) Both height and the widht of the image (it is a square).
            >> patch_size: (int) Both height and the widht of the image (it is a square).
            >> in_chs: (int) Number of channels that the input images have.
            >> n_classes: (int) Number of classes.
            >> total_embeddings: (int) Dimensionalirty fo the token/patch emebddings.
            >> transformer_blocks: (int) Number of transformer_blocks
            >> n_heads: (int) Number of heads multi-head attetion in each block.
            >> mlp_ratio: (float) Determine the hiden dimension of the MLP.
            >> qkv_bias: (bool) If True then we include biat to the query, key and value projections.
            >> p: (float) Dropout probability for the projection in each transformer block
            >> attn_drop_prob: (float) Dropout probability for the attention mechanism in each transformer block

        Attributes:
            >> patch_embedder: (PatchEmbedder) Instance of PatchEmbedder layer.
            >> cls_token (nn.Parameter) Learneable parameter that will represent the first token in the sequence. It has 'embed_dim' elements.
            >> pos_emb (nn.Parameter) Poistional embedding of the cls toker + all the patches. It has (n_patches+1)*embed_dim elements
            >> pos_drop: (nn.Dropout) Dropout layer
            >> blocks: (nn.ModuleList) Contains transformer_blocks TransformerEncoders
            >> layer_normalization: (nn.LayerNorm) Layer normalization.
        """
        super().__init__()
        self.patch_embedder = PatchEmbedder(img_size,patch_size,in_chs,total_embeddings)
        self.class_token = nn.Parameter(torch.zeros(1,1,total_embeddings))
        self.pos_emb = nn.Parameter(torch.zeros(1,1+self.patch_embedder.n_patches,total_embeddings)) # Img + class embeddings
        self.pos_drop = nn.Dropout(p)

        self.encoders = nn.ModuleList([
            TransformerEncoder(total_embeddings,n_heads,mlp_ratio,qkv_bias,p,attn_drop_prob) for _ in range(transformer_blocks)
        ])
        self.layer_normalization = nn.LayerNorm(total_embeddings, eps=1e-6)
        self.head = nn.Linear(total_embeddings, n_classes)
        return

    def forward(self,imgs):
        B = imgs.size()[0]
        patch_embeddings = self.patch_embedder(imgs) # (B,n_patches,total_embeddings)
        cls_token = self.class_token.repeat(B,1,1) # (1,1,total_embeddings) -> (B,1,total_embeddings)
        patch_embeddings = torch.cat(tensors=(cls_token, patch_embeddings), dim=1) # (B,1+n_patches,total_embeddings)
        patch_embeddings = patch_embeddings + self.pos_emb # (B,1+n_patches,total_embeddings)
        patch_embeddings = self.pos_drop(patch_embeddings)

        for encoder in self.encoders:
            patch_embeddings = encoder(patch_embeddings)

        patch_embeddings = self.layer_normalization(patch_embeddings)
        class_embedding_final = patch_embeddings[:,0] # (B, total_embeddings)
        return self.head(class_embedding_final) # (B, total_classes)
