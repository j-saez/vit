# ViT pytorch implementation

This is a ViT pytorch implementation based on the implementation done in this [repository](https://github.com/huggingface/pytorch-image-models/tree/main/timm/models).
This implementation uses the pretrained weights on ImageNet, available on the repository mentioned above.

Download this pretrained weights from [here](https://drive.google.com/file/d/1FymJaDmEJdzw5MXmV7v7XZkmMg6mx31J/view?usp=sharing) and run the following commands:

```bash
pip install gdown
mkdir weights
gdown --id 1FymJaDmEJdzw5MXmV7v7XZkmMg6mx31J
```

## Inference

There is a script to inference from the pretrained model. To infere from it run the following commands: 

```bash
cd <root of the repo>
python inference.py --file <path/to/img/>
```

The command above will load the default pretrained weights that come with this repository. If you want to load another weight file run:

```bash
cd <root of the repo>
python inference.py --file <path/to/img/> --weight <path/to/weights>
```

## Train

Not implemented yet.
