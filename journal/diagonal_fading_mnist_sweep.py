"""Reproduce the MNIST bank-size experiment with spatially correlated diagonal gains."""
from .diagonal_fading import DEPLOYMENT_RHO
from .full_fading_mnist_sweep import main


LAW = {"kind": "diagonal_fading", "rho_c": DEPLOYMENT_RHO}


if __name__ == "__main__":
    main(LAW)
