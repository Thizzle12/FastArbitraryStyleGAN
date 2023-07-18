import os
from pathlib import Path

import torch
import yaml


def train():
    n_epochs = 30
    path = os.path.join(Path.cwd(), "src/parameters/params.yaml")
    print(path)

    # Read YAML file
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    print(config)

    # for epoch in range(n_epochs):
    #     for batch_idx, data in enumerate(data):
    #         continue

    # pass


if __name__ == "__main__":
    train()
