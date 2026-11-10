"""Prepare explicitly named public datasets once before starting pooled trials."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="Comma-separated mnist,fashionmnist")
    args = parser.parse_args()
    names = args.datasets.split(",")
    if not names or set(names) - {"mnist", "fashionmnist"}:
        parser.error("Supported public datasets: mnist,fashionmnist")
    root = os.environ.get("DATASET")
    if not root:
        raise RuntimeError("DATASET must point to the shared dataset directory")
    from torchvision import datasets
    for name in names:
        dataset_class = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        sizes = {}
        for training in (True, False):
            data = dataset_class(root=root, train=training, download=True)
            sizes["training" if training else "test"] = len(data)
        print(json.dumps({"dataset": name, "root": root, "sizes": sizes}), flush=True)


if __name__ == "__main__":
    main()
