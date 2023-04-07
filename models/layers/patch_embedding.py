import torch
import torch.nn as nn

"""
"""

class PatchEmbedder(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_chs: int, total_embeddings: int):
        """
        Takes a CHSxHxW image as input, splits it into patches and embeds those patches.
        Parameters:
            >> img_size: (int) Size of the image. Notice that the image must be an square image.
            >> patch_size: (int) Size of the image patches. Notice that the patches will be squares.
            >> in_chs: (int) Number of channels for the input images.
            >> total_embeddings: (int) Will determine the number of embeddings to get from each patch.

        Attributes:
            >> img_size: (int) Size of input img. Notice that the img must be a square image.
            >> patch_size: (int) Size of img patches. Notice that the patches will be sqares.
            >> n_patches: (int) Quantity of patches.
            >> projection: (nn.Conv2d) Splits the image into patches and embeds them.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
                in_chs,
                total_embeddings,
                kernel_size=patch_size,
                stride=patch_size)
        
        return

    def forward(self, imgs):
        embeddings = self.projection(imgs) # (B,CHS,H,W) -> (B,total_embeddings,n_patches**0.5,n_patches**0.5)
        flattened_embedding = embeddings.flatten(start_dim=2) # (B,total_embeddings,n_patches**0.5,n_patches**0.5) -> (B,total_embeddings,n_patches)
        flattened_embedding = flattened_embedding.permute(0,2,1) # (B,total_embeddings,n_patches) -> (B,n_patches,total_embeddings) 
        return flattened_embedding

if __name__ == "__main__":
    path_embedder = PatchEmbedder(img_size=32, patch_size=4, in_chs=3, total_embeddings=15)
    imgs = torch.rand(4, 3, 32, 32)
    flattened_patches=path_embedder(imgs)
    assert flattened_patches.size() == (4, 64, 15)

