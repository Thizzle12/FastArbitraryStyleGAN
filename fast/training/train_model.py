import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from fast.datahandling.DataReader import Datareader
from fast.model.autoencoder import Decoder, build_encoder
from fast.model.custom_layers import AdaIN
from fast.model.losses import ContentLoss, StyleLoss


def devider(spacing: int = 80, title: str = ""):
    print(
        "\n",
        (spacing // 2 - len(title) // 2) * "-",
        title,
        (spacing // 2 - len(title) // 2) * "-",
        "\n",
    )


def train():
    """_summary_"""
    devider(title="Config and Device")
    config_path = os.path.join(Path.cwd(), "fast/parameters/params.yaml")
    print(f"Config Path: {config_path}")

    # Read YAML file
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    backbone = config["backbone"]
    file_path = config["file_path"]
    style_path = config["style_path"]

    [print(f"{key}: {value}") for key, value in config.items()]

    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and config["device"] == "gpu") else "cpu"
    )

    devider(title="Parameter Specs")

    # device = "cpu"
    print(f"Device: {device}")

    # Data
    dataset = Datareader(
        # files_path=r"D:\Monet\gan-getting-started\photo_jpg",
        files_path=r"C:\Users\Theis\Pictures\Camera Roll\Ny mappe",
        style_path=r"C:\Users\Theis\Pictures\art",
        image_size=(256, 256),
    )
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    # Losses
    style_loss_fn = StyleLoss()
    content_loss_fn = ContentLoss()

    # Models.
    encoder = build_encoder()
    decoder = Decoder(
        in_channels=512,
    )

    encoder.to(device)
    decoder.to(device)

    # print(next(encoder.parameters()).is_cuda)  # returns a boolean
    # print(next(decoder.parameters()).is_cuda)  # returns a boolean

    ada_in = AdaIN()
    ada_in.to(device)

    optimizer = torch.optim.Adamax(decoder.parameters(), lr=config["lr"])

    decoder.train()

    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    print(f"Number of trainable parameters for the Encoder: {encoder_params}")
    print(f"Number of trainable parameters for the Decoder: {decoder_params}")

    devider(title="Training")

    for epoch in range(epochs):
        print(epoch)

        running_loss = 0
        last_loss = 0.0

        for batch_idx, (content_image, style_image) in enumerate(train_data):
            content_image = content_image.to(device)
            style_image = style_image.to(device)

            style = encoder(style_image)
            content = encoder(content_image)

            t = ada_in(content["layer4"], style["layer4"])

            generated_image = decoder(t)

            generated_style = encoder(generated_image)

            style_loss = style_loss_fn(style, generated_style)

            content_loss = content_loss_fn(generated_style["layer4"], t)

            loss = style_loss + content_loss

            running_loss += loss.item()
            print(loss.item())

            loss.backward()
            optimizer.step()

            if batch_idx % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print("  batch {} loss: {}".format(batch_idx + 1, last_loss))
                running_loss = 0.0

        print(last_loss)


if __name__ == "__main__":
    train()
