import torch
import argparse
import torchvision.transforms as transforms
from imagenet_labels import labels
from   models import ViT
from PIL import Image

MAX_IDXS = 1
TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384,384)),
    transforms.Normalize([0.5 for _ in range(3)],[0.5 for _ in range(3)],)])

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Inference from Imagenet pics using ViT')
    parser.add_argument( '--file', type=str, required=True, help='Path to the image.')
    parser.add_argument( '--model-weights', type=str, default='./weights/pretrained.pth', help='Path to the image.')
    args = parser.parse_args()

    vit = ViT(
            img_size=384,
            patch_size=16,
            in_chs=3,
            n_classes=1000,
            total_embeddings=768,
            transformer_blocks=12,
            n_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            p=0,
            attn_drop_prob=0).eval().to(device)
    vit.load_state_dict(torch.load(args.model_weights))

    pil_img = Image.open(args.file)
    torch_img = TRANSFORMS(pil_img).unsqueeze(dim=0).to(device)

    with torch.no_grad():
        logits = vit(torch_img)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label = probs.max(dim=-1)[MAX_IDXS]
        print(f'class_label = {labels[predicted_label.item()]}')
