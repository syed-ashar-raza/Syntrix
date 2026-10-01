from .cnn import ConvNet
from .vit import TinyViT


def build_model(name: str):
    if name == "cnn": return ConvNet()
    if name == "vit": return TinyViT()
    raise ValueError(f"Unknown model: {name}")
