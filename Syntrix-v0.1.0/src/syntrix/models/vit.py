import torch
import torch.nn as nn


class TinyViT(nn.Module):
    def __init__(
        self,
        classes=10,
        image_size=28,
        patch_size=7,
        dim=128,
        depth=3,
        heads=4,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        patches = (image_size // patch_size) ** 2

        self.patch = nn.Conv2d(
            1,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(
            torch.zeros(1, patches + 1, dim)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)

        cls = self.cls.expand(x.size(0), -1, -1)

        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.size(1)]

        encoded = self.encoder(x)

        return self.head(
            self.norm(encoded[:, 0])
        )
