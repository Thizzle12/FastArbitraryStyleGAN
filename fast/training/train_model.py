import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast.datahandling.DataReader import DataReader
from fast.model.autoencoder import Decoder, build_encoder, Decoder2
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
    style_weight = config["style_weight"]

    [print(f"{key}: {value}") for key, value in config.items()]

    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and config["device"] == "gpu") else "cpu"
    )

    devider(title="Parameter Specs")

    # device = "cpu"
    print(f"Device: {device}")

    # Data
    dataset = DataReader(
        # files_path=r"D:\31296_39911_bundle_archive\flickr30k_images\flickr30k_images",
        files_path=r"C:\Users\Theis\Pictures\Camera Roll\Ny mappe",
        style_path=r"C:\Users\Theis\Pictures\art",
        image_size=(256, 256),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
    )

    # Losses
    style_loss_fn = StyleLoss()
    content_loss_fn = ContentLoss()

    # Models.
    encoder = build_encoder()
    # decoder = Decoder(
    #     in_channels=512,
    # )
    # decoder = Decoder(
    #     in_channels=256,
    # )
    decoder = Decoder2(in_channels=256)

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

    for epoch in range(1, epochs):
        print(f"Epoch: #{epoch}")

        running_loss = 0
        last_loss = 0.0

        with tqdm(train_loader, unit="Batch") as tepoch:
            for batch_idx, (content_image, style_image) in enumerate(tepoch):
                tepoch.set_description(f"Batch: #{batch_idx}")

                content_image = content_image.to(device)
                style_image = style_image.to(device)

                style = encoder(style_image)
                content = encoder(content_image)

                t = ada_in(content["layer3"], style["layer3"])

                generated_image = decoder(t)

                generated_style = encoder(generated_image)

                style_loss = style_weight * style_loss_fn(style, generated_style)

                content_loss = content_loss_fn(generated_style["layer3"], t)

                loss = style_loss + content_loss

                # print("")
                # print(f"Style loss: {style_loss}, Content_loss: {content_loss}")

                running_loss += loss.item()

                # Backward pass.
                optimizer.zero_grad()
                loss.backward()
                # Update weights.
                optimizer.step()

                # if batch_idx % 1000 == 999:
                #     last_loss = running_loss / 1000  # loss per batch
                #     print("  batch {} loss: {}".format(batch_idx + 1, last_loss))
                #     running_loss = 0.0

                tepoch.set_postfix(loss=loss.item())

        # Save model.
        torch.save(
            decoder.state_dict(),
            os.path.join(Path.cwd(), f"fast/model_dicts/decoder_{epoch}.pt"),
        )


if __name__ == "__main__":
    train()
