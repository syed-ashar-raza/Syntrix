import torch
from syntrix.models import ConvNet,TinyViT

def test_cnn_shape(): assert ConvNet()(torch.randn(4,1,28,28)).shape==(4,10)
def test_vit_shape(): assert TinyViT()(torch.randn(4,1,28,28)).shape==(4,10)
