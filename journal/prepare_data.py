"""Prepare explicitly named public datasets once before starting pooled trials."""

import argparse
import json
import os
import socket


# The maintained torchvision release still points FashionMNIST at an HTTP-only
# S3 website. The dataset authors also publish the same checksum-verified gzip
# files in their official repository; prefer HTTPS and retain installed mirrors
# as fallbacks. Use the original class, preserving DATASET/FashionMNIST/raw.
FASHION_HTTPS_MIRRORS = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/",
)


def fashion_mirrors(installed_mirrors):
    return list(dict.fromkeys([*FASHION_HTTPS_MIRRORS, *installed_mirrors]))


def prepare_fashion(root, timeout):
    """Try official HTTPS mirrors with torchvision's resource MD5 validation.

    Trying one mirror at a time also allows fallback after a truncated/corrupt
    archive, for which torchvision raises RuntimeError rather than URLError.
    Previously downloaded valid resources are retained and checked by torchvision.
    """
    from torchvision.datasets import FashionMNIST
    installed = list(FashionMNIST.mirrors)
    mirrors = fashion_mirrors(installed)
    previous_timeout = socket.getdefaulttimeout()
    failures = []
    socket.setdefaulttimeout(timeout)
    try:
        for mirror in mirrors:
            FashionMNIST.mirrors = [mirror]
            try:
                sizes = {}
                for training in (True, False):
                    data = FashionMNIST(root=root, train=training, download=True)
                    sizes["training" if training else "test"] = len(data)
                return sizes, {"download_mirror": mirror,
                               "resource_md5_checks": dict(FashionMNIST.resources),
                               "previous_mirror_failures": failures}
            except (OSError, RuntimeError) as error:
                failures.append({"mirror": mirror, "error": str(error)})
                print(json.dumps({"dataset": "fashionmnist", "mirror_failed": failures[-1]}), flush=True)
        raise RuntimeError("FashionMNIST download failed on all official/installed mirrors: " + json.dumps(failures))
    finally:
        FashionMNIST.mirrors = installed
        socket.setdefaulttimeout(previous_timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="Comma-separated mnist,fashionmnist")
    parser.add_argument("--timeout", type=float, default=60., help="Per-network-operation timeout in seconds")
    args = parser.parse_args()
    names = list(dict.fromkeys(name.strip() for name in args.datasets.split(",")))
    if not names or set(names) - {"mnist", "fashionmnist"}:
        parser.error("Supported public datasets: mnist,fashionmnist")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    root = os.environ.get("DATASET")
    if not root:
        raise RuntimeError("DATASET must point to the shared dataset directory")
    from torchvision import datasets
    for name in names:
        if name == "fashionmnist":
            sizes, provenance = prepare_fashion(root, args.timeout)
            print(json.dumps({"dataset": name, "root": root, "sizes": sizes, **provenance}), flush=True)
            continue
        dataset_class = datasets.MNIST
        sizes = {}
        for training in (True, False):
            data = dataset_class(root=root, train=training, download=True)
            sizes["training" if training else "test"] = len(data)
        print(json.dumps({"dataset": name, "root": root, "sizes": sizes}), flush=True)


if __name__ == "__main__":
    main()
