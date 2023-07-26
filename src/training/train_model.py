import os
from pathlib import Path

import torch
import yaml

from src.model.losses import StyleLoss, ContentLoss
from src.model.autoencoder import build_encoder, Decoder
from src.model.custom_layers import AdaIN


def train():
    n_epochs = 30
    path = os.path.join(Path.cwd(), "src/parameters/params.yaml")
    print(path)

    # Read YAML file
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    print(config)

    style_loss_fn = StyleLoss()
    content_loss_fn = ContentLoss()

    encoder = build_encoder()
    decoder = Decoder()
    ada_in = AdaIN()

    for epoch in range(n_epochs):
        for batch_idx, data in enumerate(data):
            content_image, style_image = data

            style = encoder(style_image)
            content = encoder(content_image)

            t = ada_in(content, style)

            print(f"Return shape of adain: {t.shape}")

            generated_image = decoder(t)

            generated_style = encoder(generated_image)

            style_loss = style_loss_fn(style, generated_style)

            content_loss = content_loss_fn()

    pass


if __name__ == "__main__":
    train()
