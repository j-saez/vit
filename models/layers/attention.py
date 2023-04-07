import torch
import torch.nn as nn

QKV = 3

class AttentionLayer(nn.Module):

    def __init__(self, dim: int, n_heads: int, qkv_bias: bool, attn_prob: float, proj_prob: float):
        """
        Implementation of the attention mechanism.
        Parameters:
            >> dim: (int) The input and out dimension of per token features.
            >> n_heads: (int) Number of attention heads for the attention mechanism.
            >> qkv_bias: (bool) If True then we include bias to the query, key and value projections.
            >> attn_prob: (float) Dropout probability applied to the query, key and value tensors.
            >> proj_prob: (float) Dropout probability applied to the output tensor.

        Attributes:
            >> scale: (float) Normalizing constant for the the dot product.
            >> qkv_lin_projector: (nn.Linear) Liner pojecton for the query, key and value.
            >> attn_heads_mapper: (nn.Linear) Liner mapping that takes in the concatenated output of all attention heads and maps it into a new space.
            >> attn_drop: (nn.Dropout) Dropout layer for the attention mechanism.
            >> qkv_proj_drop: (nn.Dropout) Dropout layer for the qkv linar projections.
        """
        super().__init__()

        # When all the attention heads are concatenated we will have a tensor with the same dim as the input.
        self.per_head_dim = dim // n_heads
        self.n_heads = n_heads 
        self.dim = dim

        # From the attention is all you need paper to not to feed too big values into the softmax which could lead into small gradients
        self.scale = self.per_head_dim ** -0.5

        # Take in one token embedding and generate q,k,v
        self.qkv_lin_projector = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_prob)
        # Takes the concatenated heads and maps them into a new space.
        self.attn_heads_mapper = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_prob)

        return

    def forward(self, embeddings: torch.Tensor):
        """
        Important: Input and output tensors' size is going to be the same.
                   The +1 in n_patches+1 is bc we will always have the class token
                   as the first token in the sequence.
        Inputs:
            >> embeddings: (torch.Tensor [B, n_patches+1, dim]) 
        Outputs:
            >> embeddings: (torch.Tensor [B, n_patches+1, dim]) 
        """
        B, n_tokens, dim = embeddings.size()
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv_lin_projector(embeddings) # (B, n_patches+1, dim) -> (B, n_patches+1, dim*3)
        qkv = qkv.reshape(B, n_tokens, QKV, self.n_heads, self.per_head_dim).permute(2,0,3,1,4) # (B, n_patches+1, dim*3) -> (3, B, n_heads, n_patches+1, per_head_dim)
        q,k,v = qkv[0],qkv[1],qkv[2]

        dot_product = self.scale * (q @ k.transpose(-2,-1)) # (B, n_heads, n_patches+1, n_patches+1)
        attention = dot_product.softmax(dim=-1) # (B, n_heads, n_patches+1, n_patches+1)
        attention = self.attn_drop(attention) # (B, n_heads, n_patches+1, n_patches+1)

        weighted_avg = attention @ v # (B, n_heads, N_patches+1, per_head_dim
        weighted_avg = weighted_avg.transpose(1,2) # (B, n_patches+1, n_heads, per_head_dim)
        weighted_avg = weighted_avg.flatten(start_dim=2) # (B, n_patches+1, dim)

        embeddings = self.attn_heads_mapper(weighted_avg) # (B, n_patches+1,3*dim)
        return embeddings

if __name__ == "__main__":
    from patch_embedding import PatchEmbedder

    img_size = 32
    patch_size = 8
    in_chs = 3
    total_embeddings = 60
    batch_size = 4

    imgs = torch.rand(batch_size,in_chs, img_size,img_size)
    patch_embedder = PatchEmbedder(img_size,patch_size,in_chs,total_embeddings)
    attention_layer = AttentionLayer(
        dim=total_embeddings,
        n_heads=12,
        qkv_bias=True,
        attn_prob=0.5,
        proj_prob=0.2)

    output = attention_layer(patch_embedder(imgs))
    print(f'output.size() = {output.size()}')
    assert output.size() == (batch_size, (img_size // patch_size) ** 2, 3*total_embeddings)
